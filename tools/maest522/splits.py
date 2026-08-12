"""Duplicate grouping and immutable train/validation/test split assignment."""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from .annotation_db import AnnotationStore
from .confirmed_labels import current_confirmed_states, get_label_progress
from .constants import NEW_LABELS, SPLITS


IGNORED_GROUP_VALUES = {
    "",
    "unknown",
    "unknown artist",
    "various",
    "various artists",
    "va",
    "n/a",
    "none",
}


@dataclass(frozen=True)
class TrackIdentity:
    track_id: str
    exact_sha256: str
    acoustic_fingerprint: str | None
    artist: str | None
    release_id: str | None


@dataclass(frozen=True)
class SplitSummary:
    assignments: dict[int, str]
    split_counts: dict[str, int]
    group_counts: dict[str, int]


@dataclass(frozen=True)
class LeakageAudit:
    clean: bool
    issues: tuple[str, ...]
    digest_sha256: str


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            self.parent[second_root] = first_root
        else:
            self.parent[first_root] = second_root


def _normalize_group_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in IGNORED_GROUP_VALUES:
        return None
    return normalized


def build_duplicate_groups(tracks: Iterable[TrackIdentity]) -> dict[str, str]:
    """Build deterministic connected components across duplicate/group signals."""
    track_list = list(tracks)
    track_ids = [track.track_id for track in track_list]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("Track IDs must be unique when building duplicate groups.")
    union_find = _UnionFind(track_ids)
    seen: dict[tuple[str, str], str] = {}

    for track in track_list:
        if not track.exact_sha256:
            raise ValueError(f"Track has no exact SHA-256: {track.track_id}")
        signals = (
            ("exact", track.exact_sha256),
            ("fingerprint", _normalize_group_value(track.acoustic_fingerprint)),
            ("release", _normalize_group_value(track.release_id)),
            ("artist", _normalize_group_value(track.artist)),
        )
        for signal_kind, signal_value in signals:
            if signal_value is None:
                continue
            key = (signal_kind, signal_value)
            previous_track_id = seen.get(key)
            if previous_track_id is None:
                seen[key] = track.track_id
            else:
                union_find.union(previous_track_id, track.track_id)

    component_members: dict[str, list[TrackIdentity]] = {}
    for track in track_list:
        component_members.setdefault(union_find.find(track.track_id), []).append(track)

    assignments: dict[str, str] = {}
    for members in component_members.values():
        identity_payload = "\n".join(
            sorted(member.exact_sha256 for member in members)
        ).encode("utf-8")
        group_id = hashlib.sha256(identity_payload).hexdigest()
        for member in members:
            assignments[member.track_id] = group_id
    return assignments


def _load_existing_summary(
    store: AnnotationStore,
    project_id: int,
) -> SplitSummary:
    with store.connection() as connection:
        current = current_confirmed_states(connection, project_id)
        supervised_ids = {
            track_id
            for (track_id, _label), state in current.items()
            if state in {"positive", "negative"}
        }
        rows = connection.execute(
            "SELECT id, group_id, split FROM tracks "
            "WHERE project_id = ? ORDER BY id",
            (project_id,),
        ).fetchall()
    rows = [row for row in rows if int(row["id"]) in supervised_ids]
    if any(row["group_id"] is None or row["split"] is None for row in rows):
        raise RuntimeError("Frozen project has incomplete split assignments.")
    assignments = {int(row["id"]): str(row["split"]) for row in rows}
    split_counts = {split: 0 for split in SPLITS}
    groups_by_split = {split: set() for split in SPLITS}
    for row in rows:
        split = str(row["split"])
        split_counts[split] += 1
        groups_by_split[split].add(str(row["group_id"]))
    return SplitSummary(
        assignments,
        split_counts,
        {split: len(groups_by_split[split]) for split in SPLITS},
    )


def _split_group_indices(label_matrix: numpy.ndarray, seed: int) -> dict[int, str]:
    group_count = label_matrix.shape[0]
    if group_count < 3:
        raise ValueError("At least three duplicate groups are required to freeze splits.")
    features = numpy.zeros((group_count, 1), dtype=numpy.float32)
    first_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed,
    )
    train_indices, holdout_indices = next(first_splitter.split(features, label_matrix))
    holdout_labels = label_matrix[holdout_indices]
    holdout_features = features[holdout_indices]
    second_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed + 1,
    )
    validation_local, test_local = next(
        second_splitter.split(holdout_features, holdout_labels)
    )
    result = {int(index): "train" for index in train_indices}
    result.update(
        {int(holdout_indices[index]): "val" for index in validation_local}
    )
    result.update({int(holdout_indices[index]): "test" for index in test_local})
    return result


def freeze_group_splits(
    store: AnnotationStore,
    project_id: int,
    seed: int = 522,
    proportions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> SplitSummary:
    """Assign deterministic group-disjoint splits and freeze the project."""
    if proportions != (0.70, 0.15, 0.15):
        raise ValueError("Only the approved 70/15/15 split is supported.")
    if store.is_split_frozen(project_id):
        return _load_existing_summary(store, project_id)

    incomplete = [
        progress.label
        for progress in get_label_progress(store, project_id)
        if not progress.complete
    ]
    if incomplete:
        raise RuntimeError(
            "Cannot freeze splits before all label goals are complete: "
            + ", ".join(incomplete)
        )

    with store.connection() as connection:
        current = current_confirmed_states(connection, project_id)
        supervised_ids = {
            track_id
            for (track_id, _label), state in current.items()
            if state in {"positive", "negative"}
        }
        rows = connection.execute(
            "SELECT id, exact_sha256, acoustic_fingerprint, artist, release_id "
            "FROM tracks WHERE project_id = ? ORDER BY id",
            (project_id,),
        ).fetchall()
        audit_rows = connection.execute(
            "SELECT track_id, status FROM fingerprint_audit "
            "WHERE track_id IN (SELECT id FROM tracks WHERE project_id = ?)",
            (project_id,),
        ).fetchall()
    rows = [row for row in rows if int(row["id"]) in supervised_ids]
    audited_status = {
        int(row["track_id"]): str(row["status"])
        for row in audit_rows
    }
    unaudited = [
        row
        for row in rows
        if row["acoustic_fingerprint"] is None
        and audited_status.get(int(row["id"])) not in {"unavailable", "error"}
    ]

    if not rows:
        raise ValueError("Cannot freeze splits for an empty annotation project.")
    if unaudited:
        track_ids = ", ".join(str(row["id"]) for row in unaudited[:10])
        raise RuntimeError(
            "Fingerprint audit is incomplete for track IDs: " + track_ids
        )

    identities = [
        TrackIdentity(
            track_id=str(row["id"]),
            exact_sha256=str(row["exact_sha256"]),
            acoustic_fingerprint=row["acoustic_fingerprint"],
            artist=row["artist"],
            release_id=row["release_id"],
        )
        for row in rows
    ]
    groups_by_track = build_duplicate_groups(identities)
    group_ids = sorted(set(groups_by_track.values()))
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    label_matrix = numpy.zeros(
        (len(group_ids), len(NEW_LABELS) * 2),
        dtype=numpy.int8,
    )
    for row in rows:
        track_id = int(row["id"])
        group_id = groups_by_track[str(track_id)]
        for label_index, label in enumerate(NEW_LABELS):
            state = current.get((track_id, label))
            if state == "positive":
                label_matrix[group_index[group_id], label_index * 2] = 1
            elif state == "negative":
                label_matrix[group_index[group_id], label_index * 2 + 1] = 1

    split_by_group_index = _split_group_indices(label_matrix, seed)
    split_by_group = {
        group_id: split_by_group_index[index]
        for group_id, index in group_index.items()
    }
    frozen_at = datetime.now(timezone.utc).isoformat()
    with store.connection() as connection:
        for row in rows:
            track_id = int(row["id"])
            group_id = groups_by_track[str(track_id)]
            connection.execute(
                "UPDATE tracks SET group_id = ?, split = ? WHERE id = ?",
                (group_id, split_by_group[group_id], track_id),
            )
        connection.execute(
            "UPDATE projects SET split_frozen_at = ? WHERE id = ?",
            (frozen_at, project_id),
        )

    return _load_existing_summary(store, project_id)


def audit_split_leakage(
    store: AnnotationStore,
    project_id: int,
) -> LeakageAudit:
    """Audit group and identity signals for cross-split leakage."""
    with store.connection() as connection:
        current = current_confirmed_states(connection, project_id)
        supervised_ids = {
            track_id
            for (track_id, _label), state in current.items()
            if state in {"positive", "negative"}
        }
        rows = connection.execute(
            "SELECT id, exact_sha256, acoustic_fingerprint, artist, release_id, "
            "group_id, split FROM tracks WHERE project_id = ? ORDER BY id",
            (project_id,),
        ).fetchall()
    rows = [row for row in rows if int(row["id"]) in supervised_ids]

    issues: list[str] = []
    signal_splits: dict[tuple[str, str], set[str]] = {}
    digest_rows = []
    for row in rows:
        split = row["split"]
        group_id = row["group_id"]
        if split not in SPLITS or group_id is None:
            issues.append(f"Track {row['id']} has no frozen group/split assignment.")
            continue
        signals = (
            ("group", str(group_id)),
            ("exact", str(row["exact_sha256"])),
            ("fingerprint", _normalize_group_value(row["acoustic_fingerprint"])),
            ("release", _normalize_group_value(row["release_id"])),
            ("artist", _normalize_group_value(row["artist"])),
        )
        for signal_kind, signal_value in signals:
            if signal_value is not None:
                signal_splits.setdefault((signal_kind, signal_value), set()).add(split)
        digest_rows.append(
            {
                "track_id": int(row["id"]),
                "exact_sha256": str(row["exact_sha256"]),
                "acoustic_fingerprint": _normalize_group_value(
                    row["acoustic_fingerprint"]
                ),
                "artist": _normalize_group_value(row["artist"]),
                "release_id": _normalize_group_value(row["release_id"]),
                "group_id": str(group_id),
                "split": str(split),
            }
        )

    for (signal_kind, signal_value), splits in sorted(signal_splits.items()):
        if len(splits) > 1:
            issues.append(
                f"{signal_kind}={signal_value} crosses splits: "
                + ", ".join(sorted(splits))
            )
    digest_payload = json.dumps(
        digest_rows,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest_sha256 = hashlib.sha256(digest_payload).hexdigest()
    return LeakageAudit(not issues, tuple(issues), digest_sha256)
