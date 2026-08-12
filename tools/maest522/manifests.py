"""Portable reviewed-manifest exports for MAEST 522 training."""

import json
from dataclasses import dataclass
from pathlib import Path

from .annotation_db import AnnotationStore
from .constants import NEW_LABELS, SPLITS
from .splits import audit_split_leakage


@dataclass(frozen=True)
class ManifestExportReport:
    rows_written: int
    output_path: Path
    summary_path: Path
    split_audit_sha256: str


def select_window_offsets(
    duration_seconds: float,
    window_seconds: float = 30.0,
    positions: tuple[float, ...] = (0.2, 0.5, 0.8),
) -> list[float]:
    """Select centered, clamped, deduplicated window offsets."""
    if duration_seconds <= window_seconds:
        return [0.0]
    maximum_start = duration_seconds - window_seconds
    offsets: list[float] = []
    for position in positions:
        start = min(maximum_start, max(0.0, duration_seconds * position - 15.0))
        rounded = round(start, 6)
        if rounded not in offsets:
            offsets.append(rounded)
    return offsets


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def export_training_manifest(
    store: AnnotationStore,
    project_id: int,
    output_path: Path,
    portable: bool = True,
) -> ManifestExportReport:
    """Export fully reviewed tracks and an auditable dataset summary."""
    audit = audit_split_leakage(store, project_id)
    if not audit.clean:
        raise RuntimeError("Cannot export a manifest with split leakage: " + "; ".join(audit.issues))

    with store.connection() as connection:
        track_rows = connection.execute(
            "SELECT queue_items.id AS queue_item_id, tracks.path, "
            "tracks.exact_sha256, tracks.group_id, tracks.split, "
            "tracks.duration_seconds "
            "FROM queue_items JOIN tracks ON tracks.id = queue_items.track_id "
            "WHERE queue_items.project_id = ? ORDER BY tracks.exact_sha256",
            (project_id,),
        ).fetchall()
        annotation_rows = connection.execute(
            "SELECT events.queue_item_id, events.label, events.state "
            "FROM annotation_events AS events "
            "JOIN ("
            "    SELECT queue_item_id, label, MAX(id) AS latest_id "
            "    FROM annotation_events "
            "    WHERE queue_item_id IN ("
            "        SELECT id FROM queue_items WHERE project_id = ?"
            "    ) GROUP BY queue_item_id, label"
            ") AS latest ON latest.latest_id = events.id",
            (project_id,),
        ).fetchall()
        source_rows = connection.execute(
            "SELECT queue_items.id AS queue_item_id, "
            "track_sources.suggested_label "
            "FROM queue_items "
            "JOIN track_sources ON track_sources.track_id = queue_items.track_id "
            "WHERE queue_items.project_id = ? "
            "AND track_sources.suggested_label <> ''",
            (project_id,),
        ).fetchall()
        credit_rows = connection.execute(
            "SELECT queue_credits.queue_item_id, queue_credits.label, "
            "queue_credits.candidate_role FROM queue_credits "
            "JOIN queue_items ON queue_items.id = queue_credits.queue_item_id "
            "WHERE queue_items.project_id = ?",
            (project_id,),
        ).fetchall()

    annotations: dict[int, dict[str, str]] = {}
    for row in annotation_rows:
        annotations.setdefault(int(row["queue_item_id"]), {})[
            str(row["label"])
        ] = str(row["state"])
    source_labels: dict[int, set[str]] = {}
    for row in source_rows:
        source_labels.setdefault(int(row["queue_item_id"]), set()).add(
            str(row["suggested_label"])
        )
    candidate_roles: dict[int, dict[str, str]] = {}
    for row in credit_rows:
        candidate_roles.setdefault(int(row["queue_item_id"]), {})[
            str(row["label"])
        ] = str(row["candidate_role"])

    output_rows = []
    state_counts = {
        label: {state: 0 for state in ("positive", "negative", "uncertain")}
        for label in NEW_LABELS
    }
    rows_by_split = {split: 0 for split in SPLITS}
    for row in track_rows:
        queue_item_id = int(row["queue_item_id"])
        current = annotations.get(queue_item_id, {})
        if any(
            current.get(label) not in {"positive", "negative", "uncertain"}
            for label in NEW_LABELS
        ):
            continue
        exact_sha256 = str(row["exact_sha256"])
        group_id = str(row["group_id"])
        split = str(row["split"])
        audio_path = Path(str(row["path"]))
        exported = {
            "track_id": f"sha256:{exact_sha256}",
            "group_id": f"group:{group_id}",
            "audio_ref": f"audio/{exact_sha256}{audio_path.suffix.lower()}",
            "split": split,
            "duration_seconds": float(row["duration_seconds"]),
            "window_offsets_seconds": select_window_offsets(
                float(row["duration_seconds"])
            ),
            "source_labels": sorted(source_labels.get(queue_item_id, set())),
            "candidate_roles": dict(
                sorted(candidate_roles.get(queue_item_id, {}).items())
            ),
            "labels": {label: current[label] for label in NEW_LABELS},
        }
        if not portable:
            exported["path"] = str(audio_path)
        output_rows.append(exported)
        rows_by_split[split] += 1
        for label in NEW_LABELS:
            state_counts[label][current[label]] += 1

    resolved_output = Path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = resolved_output.with_suffix(resolved_output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in output_rows:
            output_file.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary_output.replace(resolved_output)

    summary_path = resolved_output.with_name("dataset_summary.json")
    _write_json_atomic(
        summary_path,
        {
            "rows_written": len(output_rows),
            "rows_by_split": rows_by_split,
            "label_state_counts": state_counts,
            "split_audit_sha256": audit.digest_sha256,
        },
    )
    return ManifestExportReport(
        rows_written=len(output_rows),
        output_path=resolved_output,
        summary_path=summary_path,
        split_audit_sha256=audit.digest_sha256,
    )
