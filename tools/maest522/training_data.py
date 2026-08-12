"""Immutable manifests and exact audio preprocessing for MAEST 522 training."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile
import torch
import torchaudio
from maest_infer.helpers.melspectrogram import MelSpectrogram
from torch import Tensor

from .constants import NEW_LABELS, SPLITS

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 30.0
WINDOW_POSITIONS = (0.2, 0.5, 0.8)
PREPROCESSING_VERSION = "maest-infer-0.2.0-mel-16khz-mono-30s-v1"
REPLAY_SPLITS = ("replay_train", "regression_holdout")
REVIEWED_STATES = {"positive", "negative", "uncertain"}


@dataclass(frozen=True)
class TrainingSample:
    track_id: str
    group_id: str
    audio_path: Path
    split: str
    duration_seconds: float
    window_offsets_seconds: tuple[float, ...]
    targets: Tensor
    target_mask: Tensor
    candidate_roles: dict[str, str]


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    manifest_sha256: str
    split_audit_sha256: str | None
    samples: tuple[TrainingSample, ...]

    def by_split(self, split: str) -> tuple[TrainingSample, ...]:
        return tuple(sample for sample in self.samples if sample.split == split)


def select_window_offsets(
    duration_seconds: float,
    window_seconds: float = WINDOW_SECONDS,
    positions: tuple[float, ...] = WINDOW_POSITIONS,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[float, ...]:
    """Select centered, clamped starts using production-equivalent sample math."""
    if not duration_seconds > 0:
        raise ValueError("duration_seconds must be positive")
    if not window_seconds > 0:
        raise ValueError("window_seconds must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if len(positions) > 3:
        raise ValueError("at most three window positions are supported")
    if any(position < 0.0 or position > 1.0 for position in positions):
        raise ValueError("window positions must be between 0 and 1")

    total_samples = max(1, int(round(duration_seconds * sample_rate)))
    window_samples = max(1, int(window_seconds * sample_rate))
    if total_samples <= window_samples:
        return (0.0,)
    maximum_start = total_samples - window_samples
    starts = {
        int(
            max(
                0.0,
                min(
                    total_samples * position - window_samples / 2.0,
                    maximum_start,
                ),
            )
        )
        for position in positions
    }
    return tuple(start / sample_rate for start in sorted(starts))


def _resolve_audio_path(row: Mapping[str, Any], manifest_dir: Path) -> Path:
    raw_path = row.get("path")
    if raw_path:
        audio_path = Path(str(raw_path)).expanduser().resolve()
    else:
        raw_reference = str(row.get("audio_ref", "")).strip()
        if not raw_reference:
            raise ValueError("manifest row has no audio_ref or path")
        reference = Path(raw_reference)
        if reference.is_absolute():
            raise ValueError("portable audio_ref must be relative")
        audio_path = (manifest_dir / reference).resolve()
    if not audio_path.is_file():
        raise ValueError(f"manifest audio reference is unavailable: {audio_path}")
    return audio_path


def decode_manifest_row(
    row: Mapping[str, Any],
    manifest_dir: Path,
    require_labels: bool = True,
    allowed_splits: tuple[str, ...] = SPLITS,
) -> TrainingSample:
    """Validate and decode one reviewed or replay manifest row."""
    track_id = str(row.get("track_id", "")).strip()
    group_id = str(row.get("group_id", "")).strip()
    split = str(row.get("split", "")).strip()
    if not track_id or not group_id:
        raise ValueError("manifest rows require non-empty track_id and group_id")
    if split not in allowed_splits:
        raise ValueError(f"unsupported manifest split {split!r}")
    try:
        duration_seconds = float(row["duration_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"track {track_id!r} has invalid duration_seconds") from error
    offsets = select_window_offsets(duration_seconds)
    provided_offsets = row.get("window_offsets_seconds")
    if provided_offsets is not None:
        try:
            decoded_offsets = tuple(float(value) for value in provided_offsets)
        except (TypeError, ValueError) as error:
            raise ValueError(f"track {track_id!r} has invalid window offsets") from error
        if len(decoded_offsets) != len(offsets) or any(
            abs(provided - expected) > (1.0 / SAMPLE_RATE + 1e-6)
            for provided, expected in zip(decoded_offsets, offsets)
        ):
            raise ValueError(
                f"track {track_id!r} window offsets differ from preprocessing contract"
            )
        offsets = decoded_offsets

    targets = torch.zeros(len(NEW_LABELS), dtype=torch.float32)
    target_mask = torch.zeros(len(NEW_LABELS), dtype=torch.float32)
    labels = row.get("labels")
    if require_labels:
        if not isinstance(labels, Mapping):
            raise ValueError(f"track {track_id!r} has no completed labels")
        for index, label in enumerate(NEW_LABELS):
            state = labels.get(label)
            if state not in REVIEWED_STATES:
                raise ValueError(
                    f"track {track_id!r} has no completed state for {label}"
                )
            if state == "positive":
                targets[index] = 1.0
                target_mask[index] = 1.0
            elif state == "negative":
                target_mask[index] = 1.0

    raw_roles = row.get("candidate_roles", {})
    if not isinstance(raw_roles, Mapping):
        raise ValueError(f"track {track_id!r} candidate_roles must be an object")
    candidate_roles = {str(key): str(value) for key, value in raw_roles.items()}
    return TrainingSample(
        track_id=track_id,
        group_id=group_id,
        audio_path=_resolve_audio_path(row, Path(manifest_dir)),
        split=split,
        duration_seconds=duration_seconds,
        window_offsets_seconds=offsets,
        targets=targets,
        target_mask=target_mask,
        candidate_roles=candidate_roles,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on manifest line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError("manifest contains no rows")
    return rows


def _validate_unique_groups(samples: tuple[TrainingSample, ...]) -> None:
    seen_tracks: set[str] = set()
    group_splits: dict[str, str] = {}
    for sample in samples:
        if sample.track_id in seen_tracks:
            raise ValueError(f"duplicate track_id in manifest: {sample.track_id}")
        seen_tracks.add(sample.track_id)
        previous_split = group_splits.setdefault(sample.group_id, sample.split)
        if previous_split != sample.split:
            raise ValueError(
                f"group {sample.group_id!r} occurs in multiple splits: "
                f"{previous_split}, {sample.split}"
            )


def _validate_manifest_digest(
    path: Path,
    expected_manifest_sha256: str | None,
) -> str:
    digest = _sha256_file(path)
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ValueError(
            f"manifest SHA-256 mismatch: expected {expected_manifest_sha256}, got {digest}"
        )
    return digest


def load_manifest(
    path: Path,
    expected_split_audit_sha256: str,
    expected_manifest_sha256: str | None = None,
) -> DatasetManifest:
    """Load a frozen reviewed manifest and reject identity or split drift."""
    resolved_path = Path(path).resolve()
    manifest_digest = _validate_manifest_digest(
        resolved_path,
        expected_manifest_sha256,
    )
    summary_path = resolved_path.with_name("dataset_summary.json")
    if not summary_path.is_file():
        raise ValueError(f"dataset summary is unavailable: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read dataset summary: {error}") from error
    actual_audit = str(summary.get("split_audit_sha256", ""))
    if actual_audit != expected_split_audit_sha256:
        raise ValueError(
            "split audit SHA-256 mismatch: "
            f"expected {expected_split_audit_sha256}, got {actual_audit}"
        )
    samples = tuple(
        decode_manifest_row(row, resolved_path.parent, require_labels=True)
        for row in _read_jsonl(resolved_path)
    )
    _validate_unique_groups(samples)
    return DatasetManifest(
        path=resolved_path,
        manifest_sha256=manifest_digest,
        split_audit_sha256=actual_audit,
        samples=samples,
    )


def load_replay_manifest(
    path: Path,
    expected_manifest_sha256: str,
) -> DatasetManifest:
    """Load a separate unlabeled replay/regression manifest by exact digest."""
    resolved_path = Path(path).resolve()
    manifest_digest = _validate_manifest_digest(
        resolved_path,
        expected_manifest_sha256,
    )
    samples = tuple(
        decode_manifest_row(
            row,
            resolved_path.parent,
            require_labels=False,
            allowed_splits=REPLAY_SPLITS,
        )
        for row in _read_jsonl(resolved_path)
    )
    _validate_unique_groups(samples)
    return DatasetManifest(
        path=resolved_path,
        manifest_sha256=manifest_digest,
        split_audit_sha256=None,
        samples=samples,
    )


def load_audio_window(
    sample: TrainingSample,
    offset_seconds: float,
    sample_rate: int = SAMPLE_RATE,
    window_seconds: float = WINDOW_SECONDS,
    ffmpeg_path: Path = Path("ffmpeg"),
) -> Tensor:
    """Decode one exact mono window and resample it to MAEST's 16 kHz input."""
    if offset_seconds not in sample.window_offsets_seconds:
        raise ValueError(
            f"offset {offset_seconds} is not a frozen window for {sample.track_id}"
        )
    try:
        decoded, source_rate = soundfile.read(
            str(sample.audio_path),
            dtype="float32",
            always_2d=True,
        )
        waveform = torch.from_numpy(decoded).mean(dim=1)
        if source_rate != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                source_rate,
                sample_rate,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as decode_error:
        try:
            completed = subprocess.run(
                [
                    str(ffmpeg_path),
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(sample.audio_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate),
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                check=False,
                capture_output=True,
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired) as ffmpeg_error:
            raise RuntimeError(
                f"could not start ffmpeg for training audio {sample.audio_path}: "
                f"{ffmpeg_error}"
            ) from decode_error
        if completed.returncode != 0 or not completed.stdout:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"could not decode training audio {sample.audio_path}: "
                f"{detail or decode_error}"
            ) from decode_error
        waveform = torch.frombuffer(
            bytearray(completed.stdout),
            dtype=torch.float32,
        ).clone()
    start_sample = int(offset_seconds * sample_rate)
    window_samples = int(window_seconds * sample_rate)
    return waveform[start_sample : start_sample + window_samples].contiguous()


def build_maest_mel(waveform: Tensor) -> Tensor:
    """Run the installed MAEST mel implementation without approximation."""
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("waveform must be a non-empty mono tensor")
    return MelSpectrogram()(waveform)
