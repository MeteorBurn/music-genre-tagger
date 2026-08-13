"""Split-aware annotation queue construction."""

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

from .annotation_db import AnnotationStore
from .constants import (
    MAX_CANDIDATES_PER_LABEL,
    NEW_LABELS,
    ROUND_SIZE_PER_LABEL,
    SPLITS,
)


HARD_NEGATIVE_SLOTS_PER_LABEL = 25


@dataclass(frozen=True)
class QueueSummary:
    label: str
    round_number: int
    split: str
    unique_tracks: int
    source_count: int
    hard_negative_count: int
    queue_item_ids: tuple[int, ...]


def _stable_order_key(
    track_id: int,
    label: str,
    round_number: int,
    seed: int,
    student_scores: Mapping[int, float] | None,
) -> tuple[float, str]:
    digest = hashlib.sha256(
        f"{seed}:{round_number}:{label}:{track_id}".encode("utf-8")
    ).hexdigest()
    if student_scores is None:
        return 0.0, digest
    return -float(student_scores.get(track_id, float("-inf"))), digest


def _select_label_candidates(
    candidates_by_role: dict[str, set[int]],
    quota: int,
    label: str,
    round_number: int,
    seed: int,
    student_scores: Mapping[int, float] | None,
) -> list[tuple[int, str]]:
    def ordered(role: str) -> list[int]:
        return sorted(
            candidates_by_role.get(role, set()),
            key=lambda track_id: _stable_order_key(
                track_id,
                label,
                round_number,
                seed,
                student_scores,
            ),
        )

    hard_candidates = ordered("hard_negative_candidate")
    positive_candidates = ordered("positive_candidate")
    unlabeled_candidates = ordered("unlabeled_pool")
    selected: list[tuple[int, str]] = []
    selected_ids: set[int] = set()

    hard_target = min(HARD_NEGATIVE_SLOTS_PER_LABEL, quota, len(hard_candidates))
    for track_id in hard_candidates[:hard_target]:
        selected.append((track_id, "hard_negative_candidate"))
        selected_ids.add(track_id)

    for role, role_candidates in (
        ("positive_candidate", positive_candidates),
        ("hard_negative_candidate", hard_candidates[hard_target:]),
        ("unlabeled_pool", unlabeled_candidates),
    ):
        for track_id in role_candidates:
            if len(selected) >= quota:
                break
            if track_id in selected_ids:
                continue
            selected.append((track_id, role))
            selected_ids.add(track_id)
        if len(selected) >= quota:
            break
    return selected


def create_round(
    store: AnnotationStore,
    project_id: int,
    label: str,
    round_number: int,
    split: str = "train",
    student_scores: Mapping[int, float] | None = None,
    seed: int = 522,
) -> QueueSummary:
    """Create one immutable quota-based queue round."""
    if label not in NEW_LABELS:
        raise ValueError(f"Unknown MAEST 522 extension label: {label}")
    if round_number < 1:
        raise ValueError("Round number must be at least 1.")
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if split in {"val", "test"} and student_scores is not None:
        raise ValueError("Student scores cannot rank blind validation/test queues.")
    if not store.is_split_frozen(project_id):
        raise RuntimeError("Project splits must be frozen before queue creation.")

    with store.connection() as connection:
        existing_round = connection.execute(
            "SELECT 1 FROM queue_rounds "
            "WHERE project_id = ? AND label = ? AND round_number = ? "
            "AND split = ? LIMIT 1",
            (project_id, label, round_number, split),
        ).fetchone()
        if existing_round is not None:
            raise ValueError(
                f"Queue round {round_number} already exists for split {split}."
            )
        if split == "train" and student_scores is not None:
            holdout_rows = connection.execute(
                "SELECT DISTINCT split FROM queue_rounds "
                "WHERE project_id = ? AND label = ? "
                "AND split IN ('val', 'test')",
                (project_id, label),
            ).fetchall()
            holdout_splits = {str(row["split"]) for row in holdout_rows}
            if holdout_splits != {"val", "test"}:
                raise RuntimeError(
                    "Create blind val and test queues before active learning."
                )
        rows = connection.execute(
            "SELECT DISTINCT tracks.id, track_sources.suggested_label, "
            "sources.candidate_role "
            "FROM tracks "
            "JOIN track_sources ON track_sources.track_id = tracks.id "
            "JOIN sources ON sources.id = track_sources.source_id "
            "WHERE tracks.project_id = ? AND tracks.split = ? "
            "AND track_sources.suggested_label = ? "
            "AND NOT EXISTS ("
            "    SELECT 1 FROM queue_items WHERE queue_items.track_id = tracks.id "
            "    AND queue_items.project_id = tracks.project_id "
            "    AND queue_items.label = ?"
            ")",
            (project_id, split, label, label),
        ).fetchall()
        credit_rows = connection.execute(
            "SELECT COUNT(*) AS credit_count "
            "FROM queue_credits "
            "JOIN queue_items ON queue_items.id = queue_credits.queue_item_id "
            "WHERE queue_items.project_id = ? AND queue_credits.label = ?",
            (project_id, label),
        ).fetchall()

    prior_credits = int(credit_rows[0]["credit_count"]) if credit_rows else 0
    candidate_pool: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        candidate_pool[str(row["candidate_role"])].add(int(row["id"]))
    remaining_cap = max(0, MAX_CANDIDATES_PER_LABEL - prior_credits)
    quota = min(ROUND_SIZE_PER_LABEL, remaining_cap)
    selections = _select_label_candidates(
        candidate_pool,
        quota,
        label,
        round_number,
        seed,
        student_scores,
    )
    selected_track_ids = sorted(track_id for track_id, _role in selections)
    acquisition_kind = (
        "blind_holdout"
        if split in {"val", "test"}
        else "active_learning"
        if student_scores is not None
        else "random_seed"
    )
    created_at = datetime.now(timezone.utc).isoformat()
    queue_item_by_track: dict[int, int] = {}
    with store.connection() as connection:
        connection.execute(
            "INSERT INTO queue_rounds("
            "project_id, label, round_number, split, acquisition_kind, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                project_id,
                label,
                round_number,
                split,
                acquisition_kind,
                created_at,
            ),
        )
        for track_id in selected_track_ids:
            score = None if student_scores is None else student_scores.get(track_id)
            cursor = connection.execute(
                "INSERT INTO queue_items("
                "project_id, track_id, label, round_number, acquisition_kind, "
                "acquisition_score, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    project_id,
                    track_id,
                    label,
                    round_number,
                    acquisition_kind,
                    score,
                    created_at,
                ),
            )
            queue_item_by_track[track_id] = int(cursor.lastrowid)
        for track_id, role in selections:
            connection.execute(
                "INSERT INTO queue_credits(queue_item_id, label, candidate_role) "
                "VALUES (?, ?, ?)",
                (queue_item_by_track[track_id], label, role),
            )

    return QueueSummary(
        label=label,
        round_number=round_number,
        split=split,
        unique_tracks=len(selected_track_ids),
        source_count=len(selections),
        hard_negative_count=sum(
            role == "hard_negative_candidate" for _track_id, role in selections
        ),
        queue_item_ids=tuple(queue_item_by_track[track_id] for track_id in selected_track_ids),
    )
