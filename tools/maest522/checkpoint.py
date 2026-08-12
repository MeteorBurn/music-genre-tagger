"""Deterministic MAEST classifier expansion from 519 to 522 labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .model_labels import NEW_LABELS, build_522_labels, load_official_519_labels

LOGGER = logging.getLogger(__name__)

CLASSIFIER_HEAD_SHAPES_519: dict[str, tuple[int, ...]] = {
    "head.1.weight": (519, 768),
    "head.1.bias": (519,),
    "head_dist.weight": (519, 768),
    "head_dist.bias": (519,),
}


def validate_checkpoint_519(state_dict: Mapping[str, Tensor]) -> None:
    """Validate the exact public MAEST 519-label classifier tensor contract."""
    for key, expected_shape in CLASSIFIER_HEAD_SHAPES_519.items():
        tensor = state_dict.get(key)
        if not isinstance(tensor, Tensor):
            raise ValueError(f"checkpoint is missing tensor {key!r}")
        actual_shape = tuple(tensor.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"checkpoint tensor {key!r} has shape {actual_shape}; "
                f"expected {expected_shape}"
            )
        if not tensor.is_floating_point():
            raise ValueError(f"checkpoint tensor {key!r} must be floating point")


def _new_weight_rows(
    source: Tensor,
    row_count: int,
    generator: torch.Generator,
) -> Tensor:
    standard_deviation = float(source.detach().float().std(unbiased=False).item())
    if not math.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("classifier weight standard deviation must be positive")
    rows = torch.empty(
        (row_count, source.shape[1]),
        dtype=source.dtype,
        device="cpu",
    )
    torch.nn.init.trunc_normal_(
        rows,
        mean=0.0,
        std=standard_deviation,
        a=-2.0 * standard_deviation,
        b=2.0 * standard_deviation,
        generator=generator,
    )
    return rows.to(device=source.device)


def expand_classifier_state_dict(
    state_dict: Mapping[str, Tensor],
    prior_probability: float = 0.01,
    seed: int = 522,
) -> dict[str, Tensor]:
    """Clone a MAEST state dict and append three deterministic classifier rows."""
    validate_checkpoint_519(state_dict)
    if not 0.0 < prior_probability < 1.0:
        raise ValueError("prior_probability must be strictly between 0 and 1")

    expanded: dict[str, Tensor] = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, Tensor):
            raise ValueError(f"checkpoint entry {key!r} is not a tensor")
        expanded[key] = tensor.clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bias_prior = math.log(prior_probability / (1.0 - prior_probability))

    for prefix in ("head.1", "head_dist"):
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"
        source_weight = state_dict[weight_key]
        source_bias = state_dict[bias_key]
        new_weights = _new_weight_rows(source_weight, len(NEW_LABELS), generator)
        new_biases = torch.full(
            (len(NEW_LABELS),),
            bias_prior,
            dtype=source_bias.dtype,
            device=source_bias.device,
        )
        expanded[weight_key] = torch.cat((source_weight.clone(), new_weights), dim=0)
        expanded[bias_key] = torch.cat((source_bias.clone(), new_biases), dim=0)

    return expanded


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_torch_save(state_dict: Mapping[str, Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(state_dict), temporary_path)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def convert_checkpoint(
    source_path: Path,
    output_dir: Path,
    prior_probability: float = 0.01,
    seed: int = 522,
) -> dict[str, Any]:
    """Expand a raw MAEST state dict and atomically write release artifacts."""
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    loaded = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise ValueError("checkpoint must contain a raw tensor state dictionary")
    expanded = expand_classifier_state_dict(
        loaded,
        prior_probability=prior_probability,
        seed=seed,
    )
    labels = build_522_labels(load_official_519_labels())

    checkpoint_path = output_dir / "expanded-init.ckpt"
    labels_path = output_dir / "labels-522.txt"
    report_path = output_dir / "expansion-report.json"
    _atomic_torch_save(expanded, checkpoint_path)
    labels_payload = ("\n".join(labels) + "\n").encode("utf-8")
    _atomic_write_bytes(labels_path, labels_payload)

    report: dict[str, Any] = {
        "format_version": 1,
        "source_checkpoint": source_path.name,
        "source_sha256": _sha256_file(source_path),
        "output_checkpoint": checkpoint_path.name,
        "output_sha256": _sha256_file(checkpoint_path),
        "labels_file": labels_path.name,
        "labels_sha256": hashlib.sha256(labels_payload).hexdigest(),
        "legacy_label_count": 519,
        "label_count": 522,
        "new_labels": list(NEW_LABELS),
        "new_label_indices": [519, 520, 521],
        "prior_probability": prior_probability,
        "seed": seed,
        "classifier_shapes": {
            key: list(expanded[key].shape)
            for key in CLASSIFIER_HEAD_SHAPES_519
        },
    }
    report_payload = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(report_path, report_payload)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expand a raw MAEST 519-label checkpoint to 522 labels."
    )
    parser.add_argument("source_checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--prior-probability", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=522)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = convert_checkpoint(
        args.source_checkpoint,
        args.output_directory,
        prior_probability=args.prior_probability,
        seed=args.seed,
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOGGER.info(
        "Expanded %s to %s labels: %s",
        report["legacy_label_count"],
        report["label_count"],
        report["output_checkpoint"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
