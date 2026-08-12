"""Merge a trained continual model into a native MAEST 522 release artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .checkpoint import CLASSIFIER_HEAD_SHAPES_519, validate_checkpoint_519
from .model_labels import build_522_labels, load_official_519_labels


@dataclass(frozen=True)
class NativeExportReport:
    output_path: Path
    output_sha256: str
    label_path: Path
    label_sha256: str
    teacher_sha256: str
    trained_sha256: str
    evaluation_sha256: str | None


def _require_tensor(
    state: Mapping[str, Any],
    key: str,
    shape: tuple[int, ...],
) -> Tensor:
    tensor = state.get(key)
    if not isinstance(tensor, Tensor):
        raise ValueError(f"trained checkpoint is missing tensor {key!r}")
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"trained tensor {key!r} has shape {tuple(tensor.shape)}; expected {shape}"
        )
    return tensor


def merge_native_state_dict(
    teacher_state: Mapping[str, Tensor],
    student_state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Merge trained backbone/new rows while sourcing legacy rows from teacher."""
    validate_checkpoint_519(teacher_state)
    merged = {
        key: value.clone()
        for key, value in teacher_state.items()
        if isinstance(value, Tensor)
    }
    for key, value in student_state.items():
        if key.startswith("backbone.") and isinstance(value, Tensor):
            native_key = key.removeprefix("backbone.")
            if native_key in merged and not native_key.startswith(("head.", "head_dist")):
                if merged[native_key].shape != value.shape:
                    raise ValueError(
                        f"trained backbone tensor {key!r} changed shape"
                    )
                merged[native_key] = value.clone()

    for suffix in ("weight", "bias"):
        student_key = f"head_norm.{suffix}"
        teacher_key = f"head.0.{suffix}"
        if teacher_key in merged:
            trained_norm = _require_tensor(
                student_state,
                student_key,
                tuple(merged[teacher_key].shape),
            )
            merged[teacher_key] = trained_norm.clone()

    extension_weight = _require_tensor(
        student_state,
        "extension_head.weight",
        (3, 768),
    )
    extension_bias = _require_tensor(
        student_state,
        "extension_head.bias",
        (3,),
    )
    extension_dist_weight = _require_tensor(
        student_state,
        "extension_dist_head.weight",
        (3, 768),
    )
    extension_dist_bias = _require_tensor(
        student_state,
        "extension_dist_head.bias",
        (3,),
    )
    merged["head.1.weight"] = torch.cat(
        (teacher_state["head.1.weight"].clone(), extension_weight.clone())
    )
    merged["head.1.bias"] = torch.cat(
        (teacher_state["head.1.bias"].clone(), extension_bias.clone())
    )
    merged["head_dist.weight"] = torch.cat(
        (teacher_state["head_dist.weight"].clone(), extension_dist_weight.clone())
    )
    merged["head_dist.bias"] = torch.cat(
        (teacher_state["head_dist.bias"].clone(), extension_dist_bias.clone())
    )
    return merged


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_student_state(path: Path) -> tuple[dict[str, Tensor], Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("trained checkpoint must be a mapping")
    raw_state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(raw_state, Mapping):
        raise ValueError("trained checkpoint has no model state dictionary")
    student = {
        str(key): value
        for key, value in raw_state.items()
        if isinstance(key, str) and isinstance(value, Tensor)
    }
    return student, payload


def export_native_release(
    teacher_checkpoint: Path,
    trained_checkpoint: Path,
    output_checkpoint: Path,
    labels_path: Path,
    evaluation_path: Path | None = None,
) -> NativeExportReport:
    """Write one auditable native checkpoint with both 522-label heads."""
    teacher_checkpoint = Path(teacher_checkpoint)
    trained_checkpoint = Path(trained_checkpoint)
    output_checkpoint = Path(output_checkpoint)
    labels_path = Path(labels_path)
    teacher = torch.load(
        teacher_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(teacher, Mapping):
        raise ValueError("teacher checkpoint must contain a raw state dictionary")
    student, trained_payload = _load_student_state(trained_checkpoint)
    merged = merge_native_state_dict(teacher, student)
    labels = build_522_labels(load_official_519_labels())
    labels_payload = ("\n".join(labels) + "\n").encode("utf-8")
    label_sha256 = hashlib.sha256(labels_payload).hexdigest()
    evaluation_sha256 = (
        _sha256_file(Path(evaluation_path)) if evaluation_path is not None else None
    )
    metadata = {
        "format_version": 1,
        "architecture": "maest_522l_pytorch",
        "base_architecture": "discogs-maest-30s-pw-129e-519l",
        "label_count": 522,
        "label_sha256": label_sha256,
        "teacher_sha256": _sha256_file(teacher_checkpoint),
        "training_run_sha256": _sha256_file(trained_checkpoint),
        "evaluation_sha256": evaluation_sha256,
        "annotation_manifest_sha256": dict(
            trained_payload.get("input_digests", {})
        ).get("annotation"),
    }
    _atomic_torch_save(
        {"state_dict": merged, "metadata": metadata},
        output_checkpoint,
    )
    _atomic_bytes(labels_path, labels_payload)
    return NativeExportReport(
        output_path=output_checkpoint,
        output_sha256=_sha256_file(output_checkpoint),
        label_path=labels_path,
        label_sha256=label_sha256,
        teacher_sha256=metadata["teacher_sha256"],
        trained_sha256=metadata["training_run_sha256"],
        evaluation_sha256=evaluation_sha256,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a trained continual checkpoint as native MAEST 522."
    )
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    export_native_release(
        args.teacher,
        args.trained,
        args.output,
        args.labels,
        evaluation_path=args.evaluation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
