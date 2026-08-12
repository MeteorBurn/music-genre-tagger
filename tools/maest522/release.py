"""Allowlisted, audited local release staging for MAEST 522."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence

import safetensors
import torch
import transformers
from safetensors.torch import load_file

from .checkpoint import validate_checkpoint_519
from .model_labels import build_522_labels, load_official_519_labels
from .native_model import load_native_state


MODEL_ALLOWLIST = {
    "config.json",
    "model.safetensors",
    "native-maest-522.ckpt",
    "labels-522.txt",
    "feature_extraction_maest.py",
    "preprocessor_config.json",
    "README.md",
    "evaluation.json",
    "parity.json",
    "provenance.json",
    "SHA256SUMS",
}
SOURCE_REQUIRED = MODEL_ALLOWLIST - {"provenance.json", "SHA256SUMS"}
TEXT_SUFFIXES = {".json", ".md", ".py", ".txt"}
PRIVATE_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/][^\s`\"']+"),
    re.compile(r"/(?:home|Users)/[^\s`\"']+"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
)


@dataclass(frozen=True)
class ReleaseInputs:
    source_dir: Path
    teacher_checkpoint: Path
    trained_checkpoint: Path
    dataset_manifest: Path
    split_audit_sha256: str
    git_commit: str
    version: str


@dataclass(frozen=True)
class ReleaseReport:
    output_dir: Path
    files: set[str]
    file_count: int
    provenance_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid release JSON {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"release JSON {path.name} must contain an object")
    return payload


def _validate_source(inputs: ReleaseInputs) -> None:
    source_dir = Path(inputs.source_dir)
    if not source_dir.is_dir():
        raise ValueError(f"release source directory is unavailable: {source_dir}")
    actual_names = {path.name for path in source_dir.iterdir() if path.is_file()}
    missing = sorted(SOURCE_REQUIRED - actual_names)
    unexpected = sorted(actual_names - SOURCE_REQUIRED)
    if missing:
        raise ValueError("release source is missing files: " + ", ".join(missing))
    if unexpected:
        raise ValueError("release source contains unrecognized files: " + ", ".join(unexpected))

    evaluation = _json(source_dir / "evaluation.json")
    if not evaluation.get("legacy_gates_passed") or not evaluation.get("test_evaluated"):
        raise ValueError("release evaluation has not passed gates and locked test")
    parity = _json(source_dir / "parity.json")
    if (
        not parity.get("passed")
        or int(parity.get("windows", 0)) < 30
        or int(parity.get("tracks", 0)) < 10
    ):
        raise ValueError("release parity must pass on at least 30 windows and 10 tracks")

    expected_labels = build_522_labels(load_official_519_labels())
    actual_labels = tuple(
        (source_dir / "labels-522.txt").read_text(encoding="utf-8").splitlines()
    )
    if actual_labels != expected_labels:
        raise ValueError("release labels do not match the canonical 522-label order")
    config = _json(source_dir / "config.json")
    id2label = config.get("id2label")
    if not isinstance(id2label, dict) or tuple(
        str(id2label.get(str(index), id2label.get(index, "")))
        for index in range(522)
    ) != expected_labels:
        raise ValueError("config id2label does not match labels-522.txt")

    teacher = torch.load(
        Path(inputs.teacher_checkpoint),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(teacher, dict):
        raise ValueError("teacher checkpoint must be a raw state dictionary")
    validate_checkpoint_519(teacher)
    native_state = load_native_state(source_dir / "native-maest-522.ckpt")
    for key, expected_shape in {
        "head.1.weight": (522, 768),
        "head_dist.weight": (522, 768),
    }.items():
        if tuple(native_state[key].shape) != expected_shape:
            raise ValueError(f"native release tensor {key!r} has an invalid shape")
    hf_state = load_file(source_dir / "model.safetensors", device="cpu")
    for key, expected_shape in {
        "classifier.dense.weight": (522, 768),
        "classifier.dense.bias": (522,),
    }.items():
        tensor = hf_state.get(key)
        if tensor is None or tuple(tensor.shape) != expected_shape:
            raise ValueError(f"Hugging Face release tensor {key!r} has an invalid shape")

    for path in source_dir.iterdir():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in PRIVATE_PATTERNS:
            match = pattern.search(text)
            if match:
                raise ValueError(
                    f"private path/token pattern found in {path.name}: {match.group(0)[:40]}"
                )
        if "annotation note" in text.casefold():
            raise ValueError(f"private annotation notes found in {path.name}")

    if len(inputs.split_audit_sha256) != 64:
        raise ValueError("split_audit_sha256 must contain 64 hexadecimal characters")
    for required_path in (
        inputs.trained_checkpoint,
        inputs.dataset_manifest,
    ):
        if not Path(required_path).is_file():
            raise ValueError(f"provenance input is unavailable: {required_path}")


def _atomic_write(path: Path, payload: bytes) -> None:
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


def stage_release(inputs: ReleaseInputs, output_dir: Path) -> ReleaseReport:
    """Validate source inputs and atomically publish one local staging directory."""
    _validate_source(inputs)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"release output must not already exist: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        for name in sorted(SOURCE_REQUIRED):
            source = Path(inputs.source_dir) / name
            temporary_destination = staging_dir / f".{name}.tmp"
            shutil.copyfile(source, temporary_destination)
            os.replace(temporary_destination, staging_dir / name)
        provenance = {
            "format_version": 1,
            "version": inputs.version,
            "git_commit": inputs.git_commit,
            "build_time_utc": datetime.now(timezone.utc).isoformat(),
            "tools": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "safetensors": safetensors.__version__,
            },
            "source_checkpoint_sha256": _sha256_file(Path(inputs.teacher_checkpoint)),
            "trained_checkpoint_sha256": _sha256_file(Path(inputs.trained_checkpoint)),
            "native_artifact_sha256": _sha256_file(
                staging_dir / "native-maest-522.ckpt"
            ),
            "hf_artifact_sha256": _sha256_file(staging_dir / "model.safetensors"),
            "label_sha256": _sha256_file(staging_dir / "labels-522.txt"),
            "dataset_manifest_sha256": _sha256_file(Path(inputs.dataset_manifest)),
            "split_audit_sha256": inputs.split_audit_sha256,
            "evaluation_sha256": _sha256_file(staging_dir / "evaluation.json"),
            "parity_sha256": _sha256_file(staging_dir / "parity.json"),
        }
        _atomic_write(
            staging_dir / "provenance.json",
            (
                json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            ).encode("utf-8"),
        )
        hash_lines = [
            f"{_sha256_file(path)}  {path.name}"
            for path in sorted(staging_dir.iterdir(), key=lambda item: item.name)
            if path.name != "SHA256SUMS"
        ]
        _atomic_write(
            staging_dir / "SHA256SUMS",
            ("\n".join(hash_lines) + "\n").encode("utf-8"),
        )
        actual = {path.name for path in staging_dir.iterdir() if path.is_file()}
        if actual != MODEL_ALLOWLIST:
            raise RuntimeError("staged release differs from the model allowlist")
        os.replace(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    provenance_sha = _sha256_file(output_dir / "provenance.json")
    return ReleaseReport(
        output_dir=output_dir,
        files=set(MODEL_ALLOWLIST),
        file_count=len(MODEL_ALLOWLIST),
        provenance_sha256=provenance_sha,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage an audited local MAEST 522 release.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--split-audit-sha256", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    inputs = ReleaseInputs(
        source_dir=args.source_dir,
        teacher_checkpoint=args.teacher,
        trained_checkpoint=args.trained,
        dataset_manifest=args.dataset_manifest,
        split_audit_sha256=args.split_audit_sha256,
        git_commit=args.git_commit,
        version=args.version,
    )
    stage_release(inputs, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
