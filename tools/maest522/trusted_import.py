"""Read-only trusted-playlist preflight and digest-bound atomic commit."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .annotation_db import AnnotationStore
from .confirmed_labels import (
    ConfirmedBatchSummary,
    current_confirmed_states,
)
from .constants import NEW_LABELS
from .library import AudioIdentity, inspect_audio_identity
from .playlists import parse_playlist


TRUSTED_STATES = {"positive", "negative"}


@dataclass(frozen=True)
class TrustedImportPreflight:
    playlist_path: str
    playlist_sha256: str
    label: str
    state: str
    discovered: int
    new_count: int
    existing_count: int
    missing_paths: tuple[str, ...]
    duplicate_paths: tuple[str, ...]
    invalid_paths: tuple[str, ...]
    conflict_paths: tuple[str, ...]
    identities: tuple[AudioIdentity, ...] = field(repr=False)

    @property
    def clean(self) -> bool:
        return not (
            self.missing_paths
            or self.duplicate_paths
            or self.invalid_paths
            or self.conflict_paths
        )


def _playlist_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_contract(
    store: AnnotationStore,
    project_id: int,
    playlist_path: Path,
    label: str,
    state: str,
) -> Path:
    if label not in NEW_LABELS:
        raise ValueError(f"Unknown MAEST 522 extension label: {label}")
    if state not in TRUSTED_STATES:
        raise ValueError(f"Trusted playlist state must be positive or negative: {state}")
    if store.is_split_frozen(project_id):
        raise RuntimeError("Cannot import trusted labels after project split freeze.")
    resolved_playlist = Path(playlist_path).resolve()
    if not resolved_playlist.is_file():
        raise ValueError(f"Trusted playlist does not exist: {resolved_playlist}")
    if resolved_playlist.suffix.lower() not in {".m3u", ".m3u8"}:
        raise ValueError("Trusted playlist must use .m3u or .m3u8")
    return resolved_playlist


def preflight_trusted_playlist(
    store: AnnotationStore,
    project_id: int,
    playlist_path: Path,
    label: str,
    state: str,
) -> TrustedImportPreflight:
    """Validate a trusted playlist completely without writing database rows."""
    resolved_playlist = _validate_contract(
        store,
        project_id,
        playlist_path,
        label,
        state,
    )
    entries = parse_playlist(resolved_playlist)
    path_counts = Counter(str(path).casefold() for path in entries)
    duplicate_paths = {
        str(path)
        for path in entries
        if path_counts[str(path).casefold()] > 1
    }
    missing_paths: list[str] = []
    invalid_paths: list[str] = []
    identity_by_sha: dict[str, AudioIdentity] = {}
    paths_by_sha: dict[str, list[str]] = defaultdict(list)
    inspected_paths: set[str] = set()
    for entry in entries:
        path_key = str(entry).casefold()
        if path_key in inspected_paths:
            continue
        inspected_paths.add(path_key)
        if not entry.is_file():
            missing_paths.append(str(entry))
            continue
        try:
            identity = inspect_audio_identity(entry)
        except ValueError as error:
            invalid_paths.append(f"{entry}: {error}")
            continue
        paths_by_sha[identity.exact_sha256].append(str(identity.path))
        identity_by_sha.setdefault(identity.exact_sha256, identity)
    for physical_paths in paths_by_sha.values():
        if len(physical_paths) > 1:
            duplicate_paths.update(physical_paths)

    with store.connection() as connection:
        existing_rows = connection.execute(
            "SELECT id, exact_sha256 FROM tracks WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        current = current_confirmed_states(connection, project_id)
    track_by_sha = {
        str(row["exact_sha256"]): int(row["id"])
        for row in existing_rows
    }
    new_count = 0
    existing_count = 0
    conflict_paths: set[str] = set()
    for sha256, identity in identity_by_sha.items():
        track_id = track_by_sha.get(sha256)
        current_state = None if track_id is None else current.get((track_id, label))
        if current_state == state:
            existing_count += 1
        elif current_state in TRUSTED_STATES and current_state != state:
            conflict_paths.update(paths_by_sha[sha256])
        else:
            new_count += 1
    return TrustedImportPreflight(
        playlist_path=str(resolved_playlist),
        playlist_sha256=_playlist_sha256(resolved_playlist),
        label=label,
        state=state,
        discovered=len(entries),
        new_count=new_count,
        existing_count=existing_count,
        missing_paths=tuple(sorted(set(missing_paths), key=str.casefold)),
        duplicate_paths=tuple(sorted(duplicate_paths, key=str.casefold)),
        invalid_paths=tuple(sorted(set(invalid_paths), key=str.casefold)),
        conflict_paths=tuple(sorted(conflict_paths, key=str.casefold)),
        identities=tuple(
            identity_by_sha[key]
            for key in sorted(identity_by_sha)
        ),
    )


def _preflight_error(preflight: TrustedImportPreflight) -> str:
    problems: list[str] = []
    if preflight.missing_paths:
        problems.append(f"missing={len(preflight.missing_paths)}")
    if preflight.duplicate_paths:
        problems.append(f"duplicate={len(preflight.duplicate_paths)}")
    if preflight.invalid_paths:
        problems.append(f"invalid={len(preflight.invalid_paths)}")
    if preflight.conflict_paths:
        problems.append(f"conflict={len(preflight.conflict_paths)}")
    return ", ".join(problems)


def commit_trusted_playlist(
    store: AnnotationStore,
    project_id: int,
    playlist_path: Path,
    label: str,
    state: str,
    expected_playlist_sha256: str,
) -> ConfirmedBatchSummary:
    """Commit one trusted batch atomically after repeating its full preflight."""
    preflight = preflight_trusted_playlist(
        store,
        project_id,
        playlist_path,
        label,
        state,
    )
    if preflight.playlist_sha256 != expected_playlist_sha256:
        raise ValueError(
            "Trusted playlist SHA-256 changed after preflight: "
            f"expected {expected_playlist_sha256}, got {preflight.playlist_sha256}"
        )
    if not preflight.clean:
        raise ValueError(
            "Trusted playlist preflight contains missing, duplicate, invalid, "
            "or conflict entries: " + _preflight_error(preflight)
        )
    created_at = datetime.now(timezone.utc).isoformat()
    source_kind = Path(preflight.playlist_path).suffix.lower()[1:]
    event_track_ids: list[int] = []
    existing_count = 0
    with store.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        project = connection.execute(
            "SELECT split_frozen_at FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        if project is None:
            raise ValueError(f"Unknown annotation project ID: {project_id}")
        if project["split_frozen_at"] is not None:
            raise RuntimeError("Cannot import trusted labels after project split freeze.")
        for identity in preflight.identities:
            connection.execute(
                "INSERT OR IGNORE INTO tracks("
                "project_id, path, exact_sha256, duration_seconds, artist, "
                "release_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    project_id,
                    str(identity.path),
                    identity.exact_sha256,
                    identity.duration_seconds,
                    identity.artist,
                    identity.release_id,
                    created_at,
                ),
            )
            track = connection.execute(
                "SELECT id FROM tracks WHERE project_id = ? AND exact_sha256 = ?",
                (project_id, identity.exact_sha256),
            ).fetchone()
            if track is None:
                raise RuntimeError(f"Could not persist trusted track: {identity.path}")
            track_id = int(track["id"])
            current = connection.execute(
                "SELECT state FROM confirmed_label_events "
                "WHERE project_id = ? AND track_id = ? AND label = ? "
                "ORDER BY id DESC LIMIT 1",
                (project_id, track_id, label),
            ).fetchone()
            current_state = None if current is None else str(current["state"])
            if current_state == state:
                existing_count += 1
                continue
            if current_state in TRUSTED_STATES and current_state != state:
                raise ValueError(
                    f"Trusted playlist conflict for {identity.path}: "
                    f"current={current_state}, requested={state}"
                )
            event_track_ids.append(track_id)
        cursor = connection.execute(
            "INSERT INTO confirmed_label_batches("
            "project_id, label, state, source_kind, source_path, playlist_sha256, "
            "discovered_count, new_count, existing_count, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                project_id,
                label,
                state,
                source_kind,
                preflight.playlist_path,
                preflight.playlist_sha256,
                preflight.discovered,
                len(event_track_ids),
                existing_count,
                created_at,
            ),
        )
        batch_id = int(cursor.lastrowid)
        for track_id in event_track_ids:
            connection.execute(
                "INSERT INTO confirmed_label_events("
                "project_id, track_id, label, state, event_kind, batch_id, note, "
                "created_at) VALUES (?, ?, ?, ?, 'trusted_import', ?, '', ?)",
                (project_id, track_id, label, state, batch_id, created_at),
            )
    return ConfirmedBatchSummary(
        batch_id=batch_id,
        label=label,
        state=state,
        source_kind=source_kind,
        source_path=preflight.playlist_path,
        playlist_sha256=preflight.playlist_sha256,
        discovered_count=preflight.discovered,
        new_count=len(event_track_ids),
        existing_count=existing_count,
        created_at=created_at,
    )
