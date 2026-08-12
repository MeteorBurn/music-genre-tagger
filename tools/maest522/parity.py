"""Numerical parity checks between native MAEST and Hugging Face AST."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Iterable, Sequence

import torch
from mutagen import File as MutagenFile
from torch import Tensor, nn
from transformers import ASTForAudioClassification

from .native_model import get_maest_522
from .training_data import (
    TrainingSample,
    build_maest_mel,
    load_audio_window,
    select_window_offsets,
)


@dataclass(frozen=True)
class ParityReport:
    windows: int
    max_absolute_logit_error: float
    mean_absolute_logit_error: float
    max_absolute_probability_error: float
    top10_match_rate: float
    passed: bool


def prepare_hf_input(native_mel: Tensor) -> Tensor:
    """Transpose `[batch, mel, time]` to AST without changing numeric values."""
    if native_mel.ndim != 3 or native_mel.shape[1] != 96:
        raise ValueError("native mel batch must have shape [batch, 96, time]")
    return native_mel.transpose(1, 2).contiguous()


def _native_logits(model: nn.Module, mel: Tensor) -> Tensor:
    result = model(mel, melspectrogram_input=True)
    logits = result[0] if isinstance(result, tuple) else result
    if not isinstance(logits, Tensor):
        raise ValueError("native model did not return tensor logits")
    return logits


def _hf_logits(model: nn.Module, mel: Tensor) -> Tensor:
    result = model(input_values=prepare_hf_input(mel))
    logits = getattr(result, "logits", None)
    if not isinstance(logits, Tensor):
        raise ValueError("Hugging Face model did not return tensor logits")
    return logits


def compare_native_and_hf(
    native_model: nn.Module,
    hf_model: nn.Module,
    mel_batches: Iterable[Tensor],
    atol: float = 1e-5,
    probability_atol: float = 1e-6,
) -> ParityReport:
    """Compare float32 raw logits, one sigmoid, and ordered top-10 indices."""
    native_model = native_model.cpu().float().eval()
    hf_model = hf_model.cpu().float().eval()
    absolute_errors: list[Tensor] = []
    probability_errors: list[Tensor] = []
    top10_matches: list[Tensor] = []
    window_count = 0
    with torch.no_grad():
        for mel_batch in mel_batches:
            mel = mel_batch.detach().cpu().float()
            if mel.ndim != 3:
                raise ValueError("parity mel batches must have three dimensions")
            native_logits = _native_logits(native_model, mel.clone())
            hf_logits = _hf_logits(hf_model, mel)
            if native_logits.shape != hf_logits.shape or native_logits.shape[-1] != 522:
                raise ValueError("native and Hugging Face logits must match at 522 outputs")
            absolute_errors.append(torch.abs(native_logits - hf_logits).reshape(-1))
            native_probabilities = torch.sigmoid(native_logits)
            hf_probabilities = torch.sigmoid(hf_logits)
            probability_errors.append(
                torch.abs(native_probabilities - hf_probabilities).reshape(-1)
            )
            native_top10 = torch.topk(native_logits, 10, dim=-1).indices
            hf_top10 = torch.topk(hf_logits, 10, dim=-1).indices
            top10_matches.append(torch.all(native_top10 == hf_top10, dim=-1))
            window_count += int(mel.shape[0])
    if window_count == 0:
        raise ValueError("parity comparison received no windows")
    errors = torch.cat(absolute_errors)
    probability_error = torch.cat(probability_errors)
    top10_match = torch.cat(top10_matches)
    max_logit = float(errors.max().item())
    max_probability = float(probability_error.max().item())
    top10_rate = float(top10_match.float().mean().item())
    return ParityReport(
        windows=window_count,
        max_absolute_logit_error=max_logit,
        mean_absolute_logit_error=float(errors.mean().item()),
        max_absolute_probability_error=max_probability,
        top10_match_rate=top10_rate,
        passed=(
            max_logit <= atol
            and max_probability <= probability_atol
            and top10_rate == 1.0
        ),
    )


def _duration_seconds(path: Path) -> float:
    audio = MutagenFile(str(path))
    duration = getattr(getattr(audio, "info", None), "length", None)
    if duration is None or float(duration) <= 0:
        raise ValueError(f"could not determine audio duration: {path}")
    return float(duration)


def _mel_windows(audio_paths: Sequence[Path]) -> list[Tensor]:
    batches: list[Tensor] = []
    for index, audio_path in enumerate(audio_paths):
        duration = _duration_seconds(audio_path)
        offsets = select_window_offsets(duration)
        sample = TrainingSample(
            track_id=f"parity-{index}",
            group_id=f"parity-{index}",
            audio_path=audio_path.resolve(),
            split="test",
            duration_seconds=duration,
            window_offsets_seconds=offsets,
            targets=torch.zeros(3),
            target_mask=torch.zeros(3),
            candidate_roles={},
        )
        for offset in offsets:
            waveform = load_audio_window(sample, offset)
            mel = build_maest_mel(waveform)
            if mel.shape[1] != 1876:
                raise ValueError(
                    f"publishable parity requires 30-second/1876-frame windows: {audio_path}"
                )
            batches.append(mel.unsqueeze(0))
    return batches


def _atomic_report(path: Path, report: ParityReport, tracks: int) -> None:
    payload = {**asdict(report), "tracks": tracks}
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare native and Hugging Face MAEST 522 on real windows."
    )
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--hf-model-dir", type=Path, required=True)
    parser.add_argument("--audio-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    audio_paths = [
        Path(line.strip())
        for line in args.audio_list.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(audio_paths) < 10:
        raise ValueError("publishable parity requires at least 10 tracks")
    native_model = get_maest_522(args.native_checkpoint)
    hf_model = ASTForAudioClassification.from_pretrained(
        args.hf_model_dir,
        local_files_only=True,
    )
    mel_batches = _mel_windows(audio_paths)
    if sum(batch.shape[0] for batch in mel_batches) < 30:
        raise ValueError("publishable parity requires at least 30 windows")
    report = compare_native_and_hf(native_model, hf_model, mel_batches)
    _atomic_report(args.output, report, tracks=len(audio_paths))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
