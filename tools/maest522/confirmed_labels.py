"""Confirmed-label ledger, goals, progress, and append-only corrections."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from .annotation_db import AnnotationStore
from .constants import NEW_LABELS


CONFIRMED_STATES = {"positive", "negative", "uncertain"}


@dataclass(frozen=True)
class LabelGoal:
    label: str
    positive_target: int
    negative_target: int
    updated_at: str


@dataclass(frozen=True)
class LabelProgress:
    label: str
    positive_count: int
    positive_target: int
    negative_count: int
    negative_target: int
    uncertain_count: int
    complete: bool


@dataclass(frozen=True)
class ConfirmedBatchSummary:
    batch_id: int
    label: str
    state: str
    source_kind: str
    source_path: str
    playlist_sha256: str
    discovered_count: int
    new_count: int
    existing_count: int
    created_at: str


@dataclass(frozen=True)
class CorrectionResult:
    event_id: int
    track_id: int
    label: str
    previous_state: str
    state: str
    reason: str
    created_at: str


def append_manual_review(
    store: AnnotationStore,
    project_id: int,
    queue_item_id: int,
    label: str,
    state: str,
    note: str = "",
) -> int:
    """Append one queue-bound human review for the queue item's exact label."""
    _require_label(label)
    if state not in CONFIRMED_STATES:
        raise ValueError(f"Unknown confirmed annotation state: {state}")
    created_at = datetime.now(timezone.utc).isoformat()
    with store.connection() as connection:
        row = connection.execute(
            "SELECT queue_items.track_id, queue_items.label "
            "FROM queue_items WHERE queue_items.id = ? "
            "AND queue_items.project_id = ?",
            (queue_item_id, project_id),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Unknown queue item ID for project {project_id}: {queue_item_id}"
            )
        queue_label = str(row["label"])
        if queue_label != label:
            raise ValueError(
                f"Queue item label is {queue_label}, not requested label {label}"
            )
        cursor = connection.execute(
            "INSERT INTO confirmed_label_events("
            "project_id, track_id, label, state, event_kind, batch_id, note, "
            "created_at) VALUES (?, ?, ?, ?, 'manual_review', NULL, ?, ?)",
            (
                project_id,
                int(row["track_id"]),
                label,
                state,
                note.strip(),
                created_at,
            ),
        )
    return int(cursor.lastrowid)


def _require_label(label: str) -> None:
    if label not in NEW_LABELS:
        raise ValueError(f"Unknown MAEST 522 extension label: {label}")


def _project_split_frozen(
    connection: sqlite3.Connection,
    project_id: int,
) -> bool:
    row = connection.execute(
        "SELECT split_frozen_at FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown annotation project ID: {project_id}")
    return row["split_frozen_at"] is not None


def current_confirmed_states(
    connection: sqlite3.Connection,
    project_id: int,
) -> dict[tuple[int, str], str]:
    """Return the latest confirmed state for every track-label pair."""
    rows = connection.execute(
        "SELECT events.track_id, events.label, events.state "
        "FROM confirmed_label_events AS events "
        "JOIN ("
        "    SELECT track_id, label, MAX(id) AS latest_id "
        "    FROM confirmed_label_events WHERE project_id = ? "
        "    GROUP BY track_id, label"
        ") AS latest ON latest.latest_id = events.id "
        "WHERE events.project_id = ?",
        (project_id, project_id),
    ).fetchall()
    return {
        (int(row["track_id"]), str(row["label"])): str(row["state"])
        for row in rows
    }


def get_label_goals(
    store: AnnotationStore,
    project_id: int,
) -> tuple[LabelGoal, ...]:
    """Return all three project goals in canonical classifier order."""
    with store.connection() as connection:
        _project_split_frozen(connection, project_id)
        rows = connection.execute(
            "SELECT label, positive_target, negative_target, updated_at "
            "FROM label_goals WHERE project_id = ?",
            (project_id,),
        ).fetchall()
    by_label = {str(row["label"]): row for row in rows}
    if set(by_label) != set(NEW_LABELS):
        raise RuntimeError(f"Project {project_id} does not have exactly three label goals")
    return tuple(
        LabelGoal(
            label=label,
            positive_target=int(by_label[label]["positive_target"]),
            negative_target=int(by_label[label]["negative_target"]),
            updated_at=str(by_label[label]["updated_at"]),
        )
        for label in NEW_LABELS
    )


def update_label_goal(
    store: AnnotationStore,
    project_id: int,
    label: str,
    positive_target: int,
    negative_target: int,
) -> LabelGoal:
    """Update one label's soft targets without changing confirmed events."""
    _require_label(label)
    if positive_target <= 0:
        raise ValueError("positive target must be greater than zero")
    if negative_target <= 0:
        raise ValueError("negative target must be greater than zero")
    updated_at = datetime.now(timezone.utc).isoformat()
    with store.connection() as connection:
        if _project_split_frozen(connection, project_id):
            raise RuntimeError("Cannot update label goals after project split freeze.")
        cursor = connection.execute(
            "UPDATE label_goals SET positive_target = ?, negative_target = ?, "
            "updated_at = ? WHERE project_id = ? AND label = ?",
            (
                positive_target,
                negative_target,
                updated_at,
                project_id,
                label,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Project {project_id} has no goal row for {label}")
    return LabelGoal(label, positive_target, negative_target, updated_at)


def get_label_progress(
    store: AnnotationStore,
    project_id: int,
) -> tuple[LabelProgress, ...]:
    """Count latest positive, negative, and uncertain states against soft goals."""
    goals = get_label_goals(store, project_id)
    with store.connection() as connection:
        states = current_confirmed_states(connection, project_id)
    counts = {
        label: {"positive": 0, "negative": 0, "uncertain": 0}
        for label in NEW_LABELS
    }
    for (_track_id, label), state in states.items():
        if label in counts and state in CONFIRMED_STATES:
            counts[label][state] += 1
    return tuple(
        LabelProgress(
            label=goal.label,
            positive_count=counts[goal.label]["positive"],
            positive_target=goal.positive_target,
            negative_count=counts[goal.label]["negative"],
            negative_target=goal.negative_target,
            uncertain_count=counts[goal.label]["uncertain"],
            complete=(
                counts[goal.label]["positive"] >= goal.positive_target
                and counts[goal.label]["negative"] >= goal.negative_target
            ),
        )
        for goal in goals
    )


def list_confirmed_batches(
    store: AnnotationStore,
    project_id: int,
    label: str,
    limit: int = 20,
) -> tuple[ConfirmedBatchSummary, ...]:
    """Return recent trusted batches for one label."""
    _require_label(label)
    if limit <= 0:
        raise ValueError("batch history limit must be greater than zero")
    with store.connection() as connection:
        _project_split_frozen(connection, project_id)
        rows = connection.execute(
            "SELECT id, label, state, source_kind, source_path, playlist_sha256, "
            "discovered_count, new_count, existing_count, created_at "
            "FROM confirmed_label_batches WHERE project_id = ? AND label = ? "
            "ORDER BY id DESC LIMIT ?",
            (project_id, label, limit),
        ).fetchall()
    return tuple(
        ConfirmedBatchSummary(
            batch_id=int(row["id"]),
            label=str(row["label"]),
            state=str(row["state"]),
            source_kind=str(row["source_kind"]),
            source_path=str(row["source_path"]),
            playlist_sha256=str(row["playlist_sha256"]),
            discovered_count=int(row["discovered_count"]),
            new_count=int(row["new_count"]),
            existing_count=int(row["existing_count"]),
            created_at=str(row["created_at"]),
        )
        for row in rows
    )


def append_correction(
    store: AnnotationStore,
    project_id: int,
    track_id: int,
    label: str,
    state: str,
    reason: str,
) -> CorrectionResult:
    """Append an explicit correction while preserving the complete event history."""
    _require_label(label)
    if state not in CONFIRMED_STATES:
        raise ValueError(f"Unknown confirmed annotation state: {state}")
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("correction reason must not be empty")
    created_at = datetime.now(timezone.utc).isoformat()
    with store.connection() as connection:
        if _project_split_frozen(connection, project_id):
            raise RuntimeError("Cannot correct labels after project split freeze.")
        track = connection.execute(
            "SELECT id FROM tracks WHERE id = ? AND project_id = ?",
            (track_id, project_id),
        ).fetchone()
        if track is None:
            raise ValueError(f"Unknown track ID for project {project_id}: {track_id}")
        current = connection.execute(
            "SELECT state FROM confirmed_label_events "
            "WHERE project_id = ? AND track_id = ? AND label = ? "
            "ORDER BY id DESC LIMIT 1",
            (project_id, track_id, label),
        ).fetchone()
        if current is None:
            raise ValueError("track has no current confirmed state to correct")
        previous_state = str(current["state"])
        if previous_state == state:
            raise ValueError("correction state must differ from the current state")
        cursor = connection.execute(
            "INSERT INTO confirmed_label_events("
            "project_id, track_id, label, state, event_kind, batch_id, note, "
            "created_at) VALUES (?, ?, ?, ?, 'correction', NULL, ?, ?)",
            (
                project_id,
                track_id,
                label,
                state,
                normalized_reason,
                created_at,
            ),
        )
    return CorrectionResult(
        event_id=int(cursor.lastrowid),
        track_id=track_id,
        label=label,
        previous_state=previous_state,
        state=state,
        reason=normalized_reason,
        created_at=created_at,
    )
