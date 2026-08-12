"""Portable confirmed-label manifest exports for MAEST 522 training."""

import json
from dataclasses import dataclass
from pathlib import Path

from .annotation_db import AnnotationStore
from .confirmed_labels import current_confirmed_states
from .constants import NEW_LABELS, SPLITS
from .splits import audit_split_leakage


TRAINING_STATES = {"positive", "negative"}
MANIFEST_STATES = ("positive", "negative", "uncertain", "unreviewed")


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


def _validate_split_coverage(
    state_counts_by_split: dict[str, dict[str, dict[str, int]]],
) -> None:
    missing = []
    for split in SPLITS:
        for label in NEW_LABELS:
            for state in ("positive", "negative"):
                if state_counts_by_split[split][label][state] == 0:
                    missing.append(f"{split}:{label}:{state}")
    if missing:
        raise RuntimeError(
            "Cannot export training manifest without positive and negative "
            "coverage in every split: " + ", ".join(missing)
        )


def export_training_manifest(
    store: AnnotationStore,
    project_id: int,
    output_path: Path,
    portable: bool = True,
) -> ManifestExportReport:
    """Export current confirmed supervision with explicit per-label masks."""
    audit = audit_split_leakage(store, project_id)
    if not audit.clean:
        raise RuntimeError(
            "Cannot export a manifest with split leakage: "
            + "; ".join(audit.issues)
        )

    with store.connection() as connection:
        current = current_confirmed_states(connection, project_id)
        supervised_ids = {
            track_id
            for (track_id, _label), state in current.items()
            if state in TRAINING_STATES
        }
        track_rows = connection.execute(
            "SELECT id, path, exact_sha256, group_id, split, duration_seconds "
            "FROM tracks WHERE project_id = ? ORDER BY exact_sha256",
            (project_id,),
        ).fetchall()
        event_rows = connection.execute(
            "SELECT events.track_id, events.label, events.state, "
            "events.event_kind, events.batch_id FROM confirmed_label_events "
            "AS events JOIN ("
            "    SELECT track_id, label, MAX(id) AS latest_id "
            "    FROM confirmed_label_events WHERE project_id = ? "
            "    GROUP BY track_id, label"
            ") AS latest ON latest.latest_id = events.id "
            "WHERE events.project_id = ?",
            (project_id, project_id),
        ).fetchall()
        source_rows = connection.execute(
            "SELECT track_sources.track_id, track_sources.suggested_label, "
            "sources.candidate_role FROM track_sources "
            "JOIN sources ON sources.id = track_sources.source_id "
            "WHERE sources.project_id = ?",
            (project_id,),
        ).fetchall()

    source_labels: dict[int, set[str]] = {}
    candidate_roles: dict[int, dict[str, str]] = {}
    provenance: dict[tuple[int, str], dict[str, object]] = {}
    for row in source_rows:
        track_id = int(row["track_id"])
        label = str(row["suggested_label"])
        if label:
            source_labels.setdefault(track_id, set()).add(label)
            candidate_roles.setdefault(track_id, {})[label] = str(
                row["candidate_role"]
            )
    for row in event_rows:
        track_id = int(row["track_id"])
        label = str(row["label"])
        annotation_state = str(row["state"])
        provenance[(track_id, label)] = {
            "event_kind": str(row["event_kind"]),
            "batch_id": (
                None if row["batch_id"] is None else int(row["batch_id"])
            ),
        }
        candidate_roles.setdefault(track_id, {})[label] = {
            "positive": "positive_candidate",
            "negative": "hard_negative_candidate",
            "uncertain": "unlabeled_pool",
        }[annotation_state]

    output_rows = []
    state_counts = {
        label: {state: 0 for state in MANIFEST_STATES}
        for label in NEW_LABELS
    }
    state_counts_by_split = {
        split: {
            label: {state: 0 for state in MANIFEST_STATES}
            for label in NEW_LABELS
        }
        for split in SPLITS
    }
    rows_by_split = {split: 0 for split in SPLITS}
    for row in track_rows:
        track_id = int(row["id"])
        if track_id not in supervised_ids:
            continue
        exact_sha256 = str(row["exact_sha256"])
        group_id = row["group_id"]
        split = row["split"]
        if group_id is None or split not in SPLITS:
            raise RuntimeError(f"Confirmed track {track_id} has no frozen split.")
        split = str(split)
        audio_path = Path(str(row["path"]))
        labels = {
            label: current.get((track_id, label), "unreviewed")
            for label in NEW_LABELS
        }
        label_mask = {
            label: 1 if labels[label] in TRAINING_STATES else 0
            for label in NEW_LABELS
        }
        label_provenance = {
            label: provenance.get((track_id, label))
            for label in NEW_LABELS
        }
        exported = {
            "track_id": f"sha256:{exact_sha256}",
            "group_id": f"group:{group_id}",
            "audio_ref": f"audio/{exact_sha256}{audio_path.suffix.lower()}",
            "split": split,
            "duration_seconds": float(row["duration_seconds"]),
            "window_offsets_seconds": select_window_offsets(
                float(row["duration_seconds"])
            ),
            "source_labels": sorted(source_labels.get(track_id, set())),
            "candidate_roles": dict(
                sorted(candidate_roles.get(track_id, {}).items())
            ),
            "labels": labels,
            "label_mask": label_mask,
            "label_provenance": label_provenance,
        }
        if not portable:
            exported["path"] = str(audio_path)
        output_rows.append(exported)
        rows_by_split[split] += 1
        for label in NEW_LABELS:
            label_state = labels[label]
            state_counts[label][label_state] += 1
            state_counts_by_split[split][label][label_state] += 1

    if not output_rows:
        raise RuntimeError("Cannot export an empty confirmed-label manifest.")
    _validate_split_coverage(state_counts_by_split)

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
            "label_state_counts_by_split": state_counts_by_split,
            "split_audit_sha256": audit.digest_sha256,
        },
    )
    return ManifestExportReport(
        rows_written=len(output_rows),
        output_path=resolved_output,
        summary_path=summary_path,
        split_audit_sha256=audit.digest_sha256,
    )
