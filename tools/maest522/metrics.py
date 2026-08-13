"""Deterministic binary classification and calibration metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    average_precision: float
    expected_calibration_error: float


def _validated_vectors(
    targets: numpy.ndarray,
    probabilities: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    resolved_targets = numpy.asarray(targets).reshape(-1)
    resolved_probabilities = numpy.asarray(probabilities, dtype=numpy.float64).reshape(-1)
    if resolved_targets.shape != resolved_probabilities.shape:
        raise ValueError("targets and probabilities must have the same shape")
    if resolved_targets.size == 0:
        raise ValueError("metrics require at least one supervised example")
    if not numpy.isin(resolved_targets, (0, 1)).all():
        raise ValueError("classification targets must be binary")
    if not numpy.isfinite(resolved_probabilities).all():
        raise ValueError("probabilities must be finite")
    if numpy.any(resolved_probabilities < 0) or numpy.any(resolved_probabilities > 1):
        raise ValueError("probabilities must be between zero and one")
    return resolved_targets.astype(numpy.int64), resolved_probabilities


def expected_calibration_error(
    targets: numpy.ndarray,
    probabilities: numpy.ndarray,
    bin_count: int = 15,
) -> float:
    """Compute standard equal-width expected calibration error."""
    resolved_targets, resolved_probabilities = _validated_vectors(
        targets,
        probabilities,
    )
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    boundaries = numpy.linspace(0.0, 1.0, bin_count + 1)
    result = 0.0
    for index in range(bin_count):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index == bin_count - 1:
            members = (resolved_probabilities >= lower) & (
                resolved_probabilities <= upper
            )
        else:
            members = (resolved_probabilities >= lower) & (
                resolved_probabilities < upper
            )
        if not numpy.any(members):
            continue
        confidence = float(resolved_probabilities[members].mean())
        accuracy = float(resolved_targets[members].mean())
        result += float(members.mean()) * abs(accuracy - confidence)
    return result


def binary_classification_metrics(
    targets: numpy.ndarray,
    probabilities: numpy.ndarray,
    threshold: float,
) -> BinaryMetrics:
    """Evaluate one label at a frozen sigmoid threshold."""
    resolved_targets, resolved_probabilities = _validated_vectors(
        targets,
        probabilities,
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    predictions = resolved_probabilities >= threshold
    positives = resolved_targets == 1
    negatives = ~positives
    true_positive = int(numpy.sum(predictions & positives))
    false_positive = int(numpy.sum(predictions & negatives))
    true_negative = int(numpy.sum(~predictions & negatives))
    false_negative = int(numpy.sum(~predictions & positives))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    average_precision = (
        float(average_precision_score(resolved_targets, resolved_probabilities))
        if numpy.unique(resolved_targets).size == 2
        else float("nan")
    )
    return BinaryMetrics(
        threshold=float(threshold),
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        average_precision=average_precision,
        expected_calibration_error=expected_calibration_error(
            resolved_targets,
            resolved_probabilities,
        ),
    )


def select_label_threshold(
    targets: numpy.ndarray,
    probabilities: numpy.ndarray,
    minimum_precision: float = 0.8,
) -> float:
    """Maximize F1 under a precision floor, with deterministic tie breaking."""
    resolved_targets, resolved_probabilities = _validated_vectors(
        targets,
        probabilities,
    )
    if numpy.unique(resolved_targets).size != 2:
        raise ValueError("threshold selection requires both positive and negative examples")
    if not 0.0 <= minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be between zero and one")
    candidates = numpy.unique(
        numpy.concatenate((resolved_probabilities, numpy.array([0.0, 1.0])))
    )
    eligible: list[BinaryMetrics] = []
    fallback: list[BinaryMetrics] = []
    for candidate in candidates:
        metrics = binary_classification_metrics(
            resolved_targets,
            resolved_probabilities,
            float(candidate),
        )
        fallback.append(metrics)
        if metrics.precision >= minimum_precision:
            eligible.append(metrics)
    pool = eligible or fallback
    selected = max(
        pool,
        key=lambda metric: (
            metric.f1,
            metric.precision,
            metric.recall,
            metric.threshold,
        ),
    )
    return selected.threshold
