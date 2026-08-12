"""New-label quality and legacy-regression evaluation for MAEST 522."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

import numpy

from .metrics import (
    BinaryMetrics,
    binary_classification_metrics,
    select_label_threshold,
)


@dataclass(frozen=True)
class AggregatedTracks:
    track_ids: numpy.ndarray
    group_ids: numpy.ndarray
    probabilities: numpy.ndarray
    targets: numpy.ndarray
    target_mask: numpy.ndarray
    window_variance: numpy.ndarray


@dataclass(frozen=True)
class NewLabelEvaluation:
    thresholds: tuple[float, ...]
    per_label: tuple[BinaryMetrics, ...]
    macro_average_precision: float
    macro_f1: float
    micro_f1: float
    hard_negative_counts: tuple[int, ...]
    hard_negative_false_positive_rates: tuple[float, ...]


@dataclass(frozen=True)
class LegacyRegressionReport:
    mean_probability_drift: float
    mean_top10_overlap: float
    teacher_top3_in_student_top5: float
    embedding_cosine: float
    frequently_active_indices: tuple[int, ...]
    frequently_active_counts: tuple[int, ...]
    frequently_active_mean_drift: float
    frequently_active_p95_drift: float
    gates: dict[str, bool]
    gates_passed: bool


class EvaluationPermissionError(RuntimeError):
    """Raised when a locked test evaluation lacks explicit authorization."""


class EvaluationAlreadyRunError(RuntimeError):
    """Raised when a run attempts to evaluate its locked test twice."""


def aggregate_track_windows(
    track_ids: numpy.ndarray,
    group_ids: numpy.ndarray,
    probabilities: numpy.ndarray,
    targets: numpy.ndarray,
    target_mask: numpy.ndarray,
) -> AggregatedTracks:
    """Mean-aggregate window probabilities and preserve group-level identities."""
    resolved_track_ids = numpy.asarray(track_ids)
    resolved_group_ids = numpy.asarray(group_ids)
    resolved_probabilities = numpy.asarray(probabilities, dtype=numpy.float64)
    resolved_targets = numpy.asarray(targets, dtype=numpy.float64)
    resolved_mask = numpy.asarray(target_mask, dtype=numpy.float64)
    row_count = resolved_probabilities.shape[0]
    if any(
        values.shape[0] != row_count
        for values in (
            resolved_track_ids,
            resolved_group_ids,
            resolved_targets,
            resolved_mask,
        )
    ):
        raise ValueError("window arrays must share their first dimension")
    if resolved_probabilities.shape != resolved_targets.shape:
        raise ValueError("probabilities and targets must have identical shapes")
    if resolved_mask.shape != resolved_targets.shape:
        raise ValueError("target mask must match targets")
    unique_track_ids = list(dict.fromkeys(str(value) for value in resolved_track_ids))
    output_probabilities: list[numpy.ndarray] = []
    output_variance: list[numpy.ndarray] = []
    output_targets: list[numpy.ndarray] = []
    output_masks: list[numpy.ndarray] = []
    output_groups: list[str] = []
    for track_id in unique_track_ids:
        members = resolved_track_ids.astype(str) == track_id
        groups = numpy.unique(resolved_group_ids[members].astype(str))
        if groups.size != 1:
            raise ValueError(f"track {track_id!r} has multiple group IDs")
        track_targets = resolved_targets[members]
        track_masks = resolved_mask[members]
        if not numpy.all(track_targets == track_targets[0]) or not numpy.all(
            track_masks == track_masks[0]
        ):
            raise ValueError(f"track {track_id!r} has inconsistent window labels")
        output_groups.append(str(groups[0]))
        output_probabilities.append(resolved_probabilities[members].mean(axis=0))
        output_variance.append(resolved_probabilities[members].var(axis=0))
        output_targets.append(track_targets[0])
        output_masks.append(track_masks[0])
    return AggregatedTracks(
        track_ids=numpy.asarray(unique_track_ids),
        group_ids=numpy.asarray(output_groups),
        probabilities=numpy.stack(output_probabilities),
        targets=numpy.stack(output_targets),
        target_mask=numpy.stack(output_masks),
        window_variance=numpy.stack(output_variance),
    )


def evaluate_new_labels(
    targets: numpy.ndarray,
    target_mask: numpy.ndarray,
    probabilities: numpy.ndarray,
    minimum_precision: float = 0.8,
    thresholds: Sequence[float] | None = None,
    hard_negative_mask: numpy.ndarray | None = None,
) -> NewLabelEvaluation:
    """Evaluate each new label, masking uncertainty and freezing thresholds."""
    resolved_targets = numpy.asarray(targets)
    resolved_mask = numpy.asarray(target_mask).astype(bool)
    resolved_probabilities = numpy.asarray(probabilities, dtype=numpy.float64)
    if (
        resolved_targets.shape != resolved_mask.shape
        or resolved_targets.shape != resolved_probabilities.shape
        or resolved_targets.ndim != 2
    ):
        raise ValueError("targets, mask, and probabilities must be matching matrices")
    label_count = resolved_targets.shape[1]
    if thresholds is not None and len(thresholds) != label_count:
        raise ValueError("frozen thresholds must match the number of labels")
    selected_thresholds: list[float] = []
    per_label: list[BinaryMetrics] = []
    hard_negative_counts: list[int] = []
    hard_negative_false_positive_rates: list[float] = []
    all_targets: list[numpy.ndarray] = []
    all_probabilities: list[numpy.ndarray] = []
    all_thresholds: list[numpy.ndarray] = []
    for label_index in range(label_count):
        supervised = resolved_mask[:, label_index]
        label_targets = resolved_targets[supervised, label_index]
        label_probabilities = resolved_probabilities[supervised, label_index]
        threshold = (
            float(thresholds[label_index])
            if thresholds is not None
            else select_label_threshold(
                label_targets,
                label_probabilities,
                minimum_precision=minimum_precision,
            )
        )
        metrics = binary_classification_metrics(
            label_targets,
            label_probabilities,
            threshold,
        )
        selected_thresholds.append(threshold)
        per_label.append(metrics)
        all_targets.append(label_targets.astype(numpy.int64))
        all_probabilities.append(label_probabilities)
        all_thresholds.append(numpy.full(label_targets.shape, threshold))

        if hard_negative_mask is None:
            hard_members = numpy.zeros(resolved_targets.shape[0], dtype=bool)
        else:
            resolved_hard_negative = numpy.asarray(hard_negative_mask).astype(bool)
            if resolved_hard_negative.shape != resolved_targets.shape:
                raise ValueError("hard-negative mask must match targets")
            hard_members = (
                resolved_hard_negative[:, label_index]
                & supervised
                & (resolved_targets[:, label_index] == 0)
            )
        hard_count = int(hard_members.sum())
        hard_negative_counts.append(hard_count)
        hard_negative_false_positive_rates.append(
            float(
                numpy.mean(
                    resolved_probabilities[hard_members, label_index] >= threshold
                )
            )
            if hard_count
            else float("nan")
        )
    flat_targets = numpy.concatenate(all_targets)
    flat_probabilities = numpy.concatenate(all_probabilities)
    flat_thresholds = numpy.concatenate(all_thresholds)
    flat_predictions = flat_probabilities >= flat_thresholds
    true_positive = int(numpy.sum(flat_predictions & (flat_targets == 1)))
    false_positive = int(numpy.sum(flat_predictions & (flat_targets == 0)))
    false_negative = int(numpy.sum(~flat_predictions & (flat_targets == 1)))
    micro_precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    micro_recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return NewLabelEvaluation(
        thresholds=tuple(selected_thresholds),
        per_label=tuple(per_label),
        macro_average_precision=float(
            numpy.mean([metric.average_precision for metric in per_label])
        ),
        macro_f1=float(numpy.mean([metric.f1 for metric in per_label])),
        micro_f1=micro_f1,
        hard_negative_counts=tuple(hard_negative_counts),
        hard_negative_false_positive_rates=tuple(
            hard_negative_false_positive_rates
        ),
    )


def _sigmoid(values: numpy.ndarray) -> numpy.ndarray:
    clipped = numpy.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + numpy.exp(-clipped))


def evaluate_legacy_regression(
    teacher_logits: numpy.ndarray,
    student_logits: numpy.ndarray,
    teacher_embeddings: numpy.ndarray,
    student_embeddings: numpy.ndarray,
    frequently_active_probability: float = 0.10,
    minimum_active_windows: int = 25,
) -> LegacyRegressionReport:
    """Apply every release gate to old probabilities, ranks, and embeddings."""
    teacher = numpy.asarray(teacher_logits, dtype=numpy.float64)
    student = numpy.asarray(student_logits, dtype=numpy.float64)
    teacher_embedding = numpy.asarray(teacher_embeddings, dtype=numpy.float64)
    student_embedding = numpy.asarray(student_embeddings, dtype=numpy.float64)
    if teacher.shape != student.shape or teacher.ndim != 2:
        raise ValueError("teacher and student logits must be matching matrices")
    if teacher.shape[1] != 519:
        raise ValueError("legacy regression requires exactly 519 logits")
    if teacher_embedding.shape != student_embedding.shape or (
        teacher_embedding.shape[0] != teacher.shape[0]
    ):
        raise ValueError("teacher and student embeddings must match logit rows")
    teacher_probability = _sigmoid(teacher)
    student_probability = _sigmoid(student)
    drift = numpy.abs(teacher_probability - student_probability)
    mean_probability_drift = float(drift.mean())
    teacher_top10 = numpy.argpartition(teacher_probability, -10, axis=1)[:, -10:]
    student_top10 = numpy.argpartition(student_probability, -10, axis=1)[:, -10:]
    top10_overlap = numpy.mean(
        [
            len(set(left).intersection(right)) / 10.0
            for left, right in zip(teacher_top10, student_top10)
        ]
    )
    teacher_top3 = numpy.argpartition(teacher_probability, -3, axis=1)[:, -3:]
    student_top5 = numpy.argpartition(student_probability, -5, axis=1)[:, -5:]
    top3_containment = numpy.mean(
        [
            set(left).issubset(set(right))
            for left, right in zip(teacher_top3, student_top5)
        ]
    )
    dot = numpy.sum(teacher_embedding * student_embedding, axis=1)
    denominator = numpy.linalg.norm(teacher_embedding, axis=1) * numpy.linalg.norm(
        student_embedding,
        axis=1,
    )
    cosine = numpy.divide(
        dot,
        denominator,
        out=numpy.zeros_like(dot),
        where=denominator > 0,
    )
    identical_zero_vectors = (denominator == 0) & numpy.all(
        teacher_embedding == student_embedding,
        axis=1,
    )
    cosine[identical_zero_vectors] = 1.0
    embedding_cosine = float(cosine.mean())
    active_counts = numpy.sum(
        teacher_probability >= frequently_active_probability,
        axis=0,
    )
    active_indices = numpy.flatnonzero(active_counts >= minimum_active_windows)
    if active_indices.size:
        active_drift = drift[:, active_indices]
        active_mean = float(active_drift.mean())
        active_p95 = float(numpy.percentile(active_drift, 95))
    else:
        active_mean = 0.0
        active_p95 = 0.0
    gates = {
        "mean_probability_drift": mean_probability_drift <= 0.01,
        "mean_top10_overlap": float(top10_overlap) >= 0.90,
        "teacher_top3_in_student_top5": float(top3_containment) >= 0.98,
        "embedding_cosine": embedding_cosine >= 0.98,
        "frequently_active_mean_drift": active_mean <= 0.03,
        "frequently_active_p95_drift": active_p95 <= 0.10,
    }
    return LegacyRegressionReport(
        mean_probability_drift=mean_probability_drift,
        mean_top10_overlap=float(top10_overlap),
        teacher_top3_in_student_top5=float(top3_containment),
        embedding_cosine=embedding_cosine,
        frequently_active_indices=tuple(int(index) for index in active_indices),
        frequently_active_counts=tuple(int(active_counts[index]) for index in active_indices),
        frequently_active_mean_drift=active_mean,
        frequently_active_p95_drift=active_p95,
        gates=gates,
        gates_passed=all(gates.values()),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def record_evaluation(
    run_dir: Path,
    split: str,
    report: Mapping[str, Any],
    allow_test: bool = False,
) -> None:
    """Persist JSON/CSV/Markdown and enforce single-use locked-test access."""
    resolved_run = Path(run_dir)
    manifest_path = resolved_run / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split == "test":
        if not allow_test:
            raise EvaluationPermissionError(
                "locked test evaluation requires explicit --allow-test authorization"
            )
        if manifest.get("test_evaluated"):
            raise EvaluationAlreadyRunError("locked test was already evaluated")
        manifest["test_evaluated"] = True
    payload = {"split": split, "report": _json_safe(dict(report))}
    _atomic_text(
        resolved_run / "evaluation.json",
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    flat_rows: list[tuple[str, Any]] = []
    for key, value in payload["report"].items():
        if not isinstance(value, (dict, list)):
            flat_rows.append((str(key), value))
    csv_lines = ["metric,value"]
    csv_lines.extend(f"{key},{value}" for key, value in flat_rows)
    _atomic_text(resolved_run / "metrics.csv", "\n".join(csv_lines) + "\n")
    markdown = [f"# MAEST 522 evaluation: {split}", ""]
    markdown.extend(f"- {key}: {value}" for key, value in flat_rows)
    _atomic_text(resolved_run / "evaluation.md", "\n".join(markdown) + "\n")
    _atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persist a prepared MAEST 522 evaluation report."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--allow-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("evaluation report must be a JSON object")
    record_evaluation(
        args.run_dir,
        args.split,
        report,
        allow_test=args.allow_test,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
