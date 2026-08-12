"""Train-only active-learning acquisition for the next blind annotation round."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy

MAX_LABEL_CREDITS = 1_000


@dataclass(frozen=True)
class AcquisitionCandidate:
    track_id: str
    group_id: str
    split: str
    reviewed: bool
    label_credit_counts: tuple[int, int, int]
    probabilities: numpy.ndarray
    hard_negative_score: float
    embedding: numpy.ndarray


@dataclass(frozen=True)
class AcquisitionResult:
    scores: dict[str, float]
    components: dict[str, dict[str, float]]
    excluded_counts: dict[str, int]
    seed: int


def _validate_candidate(candidate: AcquisitionCandidate) -> None:
    probabilities = numpy.asarray(candidate.probabilities, dtype=numpy.float64)
    embedding = numpy.asarray(candidate.embedding, dtype=numpy.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(
            f"candidate {candidate.track_id!r} probabilities must have shape (windows, 3)"
        )
    if probabilities.shape[0] == 0 or not numpy.isfinite(probabilities).all():
        raise ValueError(f"candidate {candidate.track_id!r} probabilities are invalid")
    if numpy.any(probabilities < 0) or numpy.any(probabilities > 1):
        raise ValueError(f"candidate {candidate.track_id!r} probabilities must be in [0, 1]")
    if embedding.ndim != 1 or embedding.size == 0 or not numpy.isfinite(embedding).all():
        raise ValueError(f"candidate {candidate.track_id!r} embedding is invalid")
    if len(candidate.label_credit_counts) != 3 or any(
        count < 0 for count in candidate.label_credit_counts
    ):
        raise ValueError(f"candidate {candidate.track_id!r} label credits are invalid")
    if not 0.0 <= candidate.hard_negative_score <= 1.0:
        raise ValueError("hard_negative_score must be between zero and one")


def _normalized_entropy(probabilities: numpy.ndarray) -> float:
    mean_probability = probabilities.mean(axis=0)
    clipped = numpy.clip(mean_probability, 1e-8, 1.0 - 1e-8)
    binary_entropy = -(
        clipped * numpy.log(clipped)
        + (1.0 - clipped) * numpy.log(1.0 - clipped)
    )
    return float(binary_entropy.mean() / math.log(2.0))


def _window_disagreement(probabilities: numpy.ndarray) -> float:
    if probabilities.shape[0] < 2:
        return 0.0
    return float(numpy.clip(probabilities.std(axis=0).mean() / 0.5, 0.0, 1.0))


def _normalized_embeddings(
    candidates: list[AcquisitionCandidate],
) -> numpy.ndarray:
    matrix = numpy.stack(
        [numpy.asarray(candidate.embedding, dtype=numpy.float64) for candidate in candidates]
    )
    norms = numpy.linalg.norm(matrix, axis=1, keepdims=True)
    return numpy.divide(matrix, norms, out=numpy.zeros_like(matrix), where=norms > 0)


def _farthest_first_diversity(
    candidates: list[AcquisitionCandidate],
    base_scores: numpy.ndarray,
    limit: int,
    seed: int,
) -> tuple[list[int], numpy.ndarray]:
    if not candidates:
        return [], numpy.empty(0, dtype=numpy.float64)
    embeddings = _normalized_embeddings(candidates)
    rng = numpy.random.default_rng(seed)
    tie_break = rng.random(len(candidates))
    first = max(
        range(len(candidates)),
        key=lambda index: (base_scores[index], tie_break[index]),
    )
    selected = [first]
    diversity = numpy.zeros(len(candidates), dtype=numpy.float64)
    maximum_selected = min(limit, len(candidates))
    while len(selected) < maximum_selected:
        selected_matrix = embeddings[selected]
        cosine = embeddings @ selected_matrix.T
        distance = 1.0 - numpy.max(cosine, axis=1)
        distance = numpy.clip(distance / 2.0, 0.0, 1.0)
        diversity = numpy.maximum(diversity, distance)
        remaining = [index for index in range(len(candidates)) if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: (
                base_scores[index] + 0.15 * diversity[index],
                tie_break[index],
            ),
        )
        selected.append(next_index)
    if len(selected) == 1:
        diversity[first] = 1.0
    else:
        diversity[first] = max(diversity[selected[1:]], default=1.0)
    return selected, diversity


def rank_training_candidates(
    candidates: Iterable[AcquisitionCandidate],
    reviewed_group_ids: set[str],
    limit: int = 300,
    seed: int = 522,
) -> AcquisitionResult:
    """Rank only eligible train tracks using uncertainty and diversity."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    excluded_counts = {
        "reviewed": 0,
        "non_train": 0,
        "reviewed_group_sibling": 0,
        "label_cap": 0,
    }
    eligible: list[AcquisitionCandidate] = []
    seen_track_ids: set[str] = set()
    for candidate in candidates:
        _validate_candidate(candidate)
        if candidate.track_id in seen_track_ids:
            raise ValueError(f"duplicate candidate track ID: {candidate.track_id}")
        seen_track_ids.add(candidate.track_id)
        if candidate.reviewed:
            excluded_counts["reviewed"] += 1
            continue
        if candidate.split != "train":
            excluded_counts["non_train"] += 1
            continue
        if candidate.group_id in reviewed_group_ids:
            excluded_counts["reviewed_group_sibling"] += 1
            continue
        if all(count >= MAX_LABEL_CREDITS for count in candidate.label_credit_counts):
            excluded_counts["label_cap"] += 1
            continue
        eligible.append(candidate)

    base_components: list[dict[str, float]] = []
    base_scores: list[float] = []
    for candidate in eligible:
        probabilities = numpy.asarray(candidate.probabilities, dtype=numpy.float64)
        entropy = _normalized_entropy(probabilities)
        disagreement = _window_disagreement(probabilities)
        hard_negative = float(candidate.hard_negative_score)
        base_score = 0.35 * entropy + 0.25 * disagreement + 0.25 * hard_negative
        base_components.append(
            {
                "normalized_entropy": entropy,
                "window_disagreement": disagreement,
                "hard_negative_score": hard_negative,
            }
        )
        base_scores.append(base_score)
    selected_indices, diversity = _farthest_first_diversity(
        eligible,
        numpy.asarray(base_scores),
        limit,
        seed,
    )
    score_rows: list[tuple[str, float, dict[str, float]]] = []
    for index in selected_indices:
        component = dict(base_components[index])
        component["diversity_score"] = float(diversity[index])
        final_score = float(
            numpy.clip(base_scores[index] + 0.15 * diversity[index], 0.0, 1.0)
        )
        score_rows.append((eligible[index].track_id, final_score, component))
    score_rows.sort(key=lambda row: (-row[1], row[0]))
    return AcquisitionResult(
        scores={track_id: score for track_id, score, _ in score_rows},
        components={track_id: component for track_id, _, component in score_rows},
        excluded_counts=excluded_counts,
        seed=seed,
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def export_acquisition_scores(
    result: AcquisitionResult,
    output_path: Path,
) -> Path:
    """Export the queue contract plus a separate diagnostic report."""
    resolved_output = Path(output_path)
    _atomic_json(resolved_output, result.scores)
    diagnostic_path = resolved_output.with_name(
        f"{resolved_output.stem}-diagnostic.json"
    )
    _atomic_json(
        diagnostic_path,
        {
            "seed": result.seed,
            "selected_count": len(result.scores),
            "excluded_counts": result.excluded_counts,
            "components": result.components,
        },
    )
    return diagnostic_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank prepared train-only MAEST active-learning candidates."
    )
    parser.add_argument("--candidates-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=522)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("candidate input must be a JSON array")
    candidates = [
        AcquisitionCandidate(
            track_id=str(row["track_id"]),
            group_id=str(row["group_id"]),
            split=str(row["split"]),
            reviewed=bool(row["reviewed"]),
            label_credit_counts=tuple(int(value) for value in row["label_credit_counts"]),
            probabilities=numpy.asarray(row["probabilities"], dtype=numpy.float64),
            hard_negative_score=float(row["hard_negative_score"]),
            embedding=numpy.asarray(row["embedding"], dtype=numpy.float64),
        )
        for row in payload
    ]
    result = rank_training_candidates(
        candidates,
        reviewed_group_ids=set(),
        limit=args.limit,
        seed=args.seed,
    )
    export_acquisition_scores(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
