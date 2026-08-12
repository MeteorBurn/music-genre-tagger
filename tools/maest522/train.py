"""Staged, resumable pure-PyTorch training primitives for MAEST 522."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy
import torch
from torch import Tensor, nn

from .checkpoint import expand_classifier_state_dict
from .continual_model import (
    ContinualMaest522,
    TrainingStage,
    apply_training_stage,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    teacher_checkpoint: Path
    annotation_manifest: Path
    replay_manifest: Path
    output_dir: Path
    seed: int = 522
    batch_size: int = 8
    accumulation_steps: int = 4
    head_learning_rate: float = 3e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_epochs_per_stage: int = 30
    patience: int = 5
    max_gradient_norm: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "teacher_checkpoint",
            "annotation_manifest",
            "replay_manifest",
            "output_dir",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in (
            "batch_size",
            "accumulation_steps",
            "max_epochs_per_stage",
            "patience",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "head_learning_rate",
            "backbone_learning_rate",
            "max_gradient_norm",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "teacher_checkpoint",
            "annotation_manifest",
            "replay_manifest",
            "output_dir",
        ):
            payload[key] = str(payload[key])
        return payload

    def input_digests(self) -> dict[str, str]:
        return {
            "teacher": _sha256_file(self.teacher_checkpoint),
            "annotation": _sha256_file(self.annotation_manifest),
            "replay": _sha256_file(self.replay_manifest),
        }

    @classmethod
    def from_json(cls, path: Path) -> "TrainingConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("training config must be a JSON object")
        return cls(**payload)


@dataclass(frozen=True)
class TrainingProgress:
    stage: TrainingStage
    epoch: int
    global_step: int
    best_metrics: dict[str, Any]


@dataclass(frozen=True)
class EpochTrainingResult:
    mean_loss: float
    microbatches: int
    optimizer_steps: int


@dataclass
class EarlyStopping:
    patience: int
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive")

    def update(self, improved: bool) -> bool:
        if improved:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


def _sha256_file(path: Path) -> str:
    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(f"training input is unavailable: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    stage: TrainingStage,
) -> torch.optim.AdamW:
    """Build fresh AdamW groups after each named unfreezing transition."""
    resolved_stage = TrainingStage(stage)
    head_parameters: list[nn.Parameter] = []
    backbone_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("extension_head.", "extension_dist_head.")):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    if not head_parameters:
        raise ValueError("training stage has no trainable extension-head parameters")
    if resolved_stage is TrainingStage.EXTENSION_HEADS and backbone_parameters:
        raise ValueError("extension-head stage cannot include backbone parameters")
    if resolved_stage is not TrainingStage.EXTENSION_HEADS and not backbone_parameters:
        raise ValueError("backbone stage has no trainable backbone parameters")

    groups: list[dict[str, Any]] = [
        {
            "params": head_parameters,
            "lr": config.head_learning_rate,
            "weight_decay": config.weight_decay,
            "name": "extension_heads",
        }
    ]
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": config.backbone_learning_rate,
                "weight_decay": config.weight_decay,
                "name": "backbone",
            }
        )
    return torch.optim.AdamW(groups)


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [_move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, dict):
        return {
            key: _move_batch_to_device(value, device)
            for key, value in batch.items()
        }
    return batch


def train_accumulated_batches(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[Any, Tensor], Tensor],
    accumulation_steps: int,
    device: torch.device,
    max_gradient_norm: float,
) -> EpochTrainingResult:
    """Train one epoch with AMP, accumulation, and clipped optimizer steps."""
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda"
    autocast_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_total = 0.0
    microbatch_count = 0
    optimizer_steps = 0
    pending_microbatches = 0

    def optimizer_step(partial_count: int) -> None:
        nonlocal optimizer_steps
        if partial_count < accumulation_steps:
            correction = accumulation_steps / partial_count
            scaler.unscale_(optimizer)
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
        else:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    for inputs, targets in batches:
        resolved_inputs = _move_batch_to_device(inputs, device)
        resolved_targets = _move_batch_to_device(targets, device)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=amp_enabled,
        ):
            output = model(resolved_inputs)
            loss = loss_function(output, resolved_targets)
            scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        loss_total += float(loss.detach().cpu().item())
        microbatch_count += 1
        pending_microbatches += 1
        if pending_microbatches == accumulation_steps:
            optimizer_step(pending_microbatches)
            pending_microbatches = 0
    if pending_microbatches:
        optimizer_step(pending_microbatches)
    if microbatch_count == 0:
        raise ValueError("training epoch received no microbatches")
    return EpochTrainingResult(
        mean_loss=loss_total / microbatch_count,
        microbatches=microbatch_count,
        optimizer_steps=optimizer_steps,
    )


def select_better_validation_result(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any] | None,
) -> bool:
    """Apply the approved regression-first lexicographic selection rule."""
    if incumbent is None:
        return True
    candidate_key = (
        bool(candidate.get("regression_gates_passed", False)),
        float(candidate.get("macro_average_precision", float("-inf"))),
        float(candidate.get("macro_f1", float("-inf"))),
        -float(candidate.get("legacy_probability_drift", float("inf"))),
    )
    incumbent_key = (
        bool(incumbent.get("regression_gates_passed", False)),
        float(incumbent.get("macro_average_precision", float("-inf"))),
        float(incumbent.get("macro_f1", float("-inf"))),
        -float(incumbent.get("legacy_probability_drift", float("inf"))),
    )
    return candidate_key > incumbent_key


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=resolved.parent,
        prefix=f".{resolved.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, resolved)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    progress: TrainingProgress,
    input_digests: Mapping[str, str],
) -> None:
    """Atomically persist exact resume state, including all random generators."""
    _atomic_torch_save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "progress": {
                "stage": progress.stage.value,
                "epoch": progress.epoch,
                "global_step": progress.global_step,
                "best_metrics": progress.best_metrics,
            },
            "input_digests": dict(input_digests),
            "rng_state": _capture_rng_state(),
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_input_digests: Mapping[str, str],
) -> TrainingProgress:
    """Restore a run only when all immutable training inputs still match."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported training checkpoint format")
    stored_digests = payload.get("input_digests")
    if stored_digests != dict(expected_input_digests):
        raise ValueError(
            "training input digest mismatch; refusing to resume with drifted inputs"
        )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _restore_rng_state(payload["rng_state"])
    raw_progress = payload["progress"]
    return TrainingProgress(
        stage=TrainingStage(raw_progress["stage"]),
        epoch=int(raw_progress["epoch"]),
        global_step=int(raw_progress["global_step"]),
        best_metrics=dict(raw_progress["best_metrics"]),
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with path.open("ab") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())


def run_training_stage(
    model: nn.Module,
    batches_for_epoch: Callable[[int], Iterable[tuple[Tensor, Tensor]]],
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[Any, Tensor], Tensor],
    validate: Callable[[nn.Module], Mapping[str, Any]],
    config: TrainingConfig,
    stage: TrainingStage,
    input_digests: Mapping[str, str],
    device: torch.device,
    resume_from: Path | None = None,
) -> TrainingProgress:
    """Run one stage with validation, early stopping, and atomic resume points."""
    resolved_stage = TrainingStage(stage)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_metrics: dict[str, Any] = {}
    starting_epoch = 0
    global_step = 0
    if resume_from is not None:
        resumed = load_training_checkpoint(
            resume_from,
            model,
            optimizer,
            input_digests,
        )
        if resumed.stage is not resolved_stage:
            raise ValueError(
                f"resume checkpoint belongs to {resumed.stage.value}, "
                f"not {resolved_stage.value}"
            )
        starting_epoch = resumed.epoch
        global_step = resumed.global_step
        best_metrics = dict(resumed.best_metrics)

    stopper = EarlyStopping(config.patience)
    progress = TrainingProgress(
        stage=resolved_stage,
        epoch=starting_epoch,
        global_step=global_step,
        best_metrics=best_metrics,
    )
    for epoch in range(starting_epoch + 1, config.max_epochs_per_stage + 1):
        training_result = train_accumulated_batches(
            model=model,
            batches=batches_for_epoch(epoch),
            optimizer=optimizer,
            loss_function=loss_function,
            accumulation_steps=config.accumulation_steps,
            device=device,
            max_gradient_norm=config.max_gradient_norm,
        )
        validation = dict(validate(model))
        improved = select_better_validation_result(
            validation,
            best_metrics or None,
        )
        if improved:
            best_metrics = validation
        global_step += training_result.optimizer_steps
        progress = TrainingProgress(
            stage=resolved_stage,
            epoch=epoch,
            global_step=global_step,
            best_metrics=dict(best_metrics),
        )
        save_training_checkpoint(
            config.output_dir / "last.ckpt",
            model,
            optimizer,
            progress,
            input_digests,
        )
        if improved:
            save_training_checkpoint(
                config.output_dir / "best.ckpt",
                model,
                optimizer,
                progress,
                input_digests,
            )
        _append_jsonl(
            config.output_dir / "metrics.jsonl",
            {
                "stage": resolved_stage.value,
                "epoch": epoch,
                "global_step": global_step,
                "train_mean_loss": training_result.mean_loss,
                "optimizer_steps": training_result.optimizer_steps,
                "improved": improved,
                "validation": validation,
            },
        )
        if stopper.update(improved):
            break
    return progress


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=resolved.parent,
        prefix=f".{resolved.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, resolved)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def initialize_run(config: TrainingConfig) -> dict[str, Any]:
    """Write reproducibility records before any optimizer step is allowed."""
    input_digests = config.input_digests()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(config.output_dir / "resolved-config.json", config.to_dict())
    _atomic_json(
        config.output_dir / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
    )
    run_manifest = {
        "format_version": 1,
        "seed": config.seed,
        "input_digests": input_digests,
        "stages": [stage.value for stage in TrainingStage],
        "test_evaluated": False,
    }
    _atomic_json(config.output_dir / "run-manifest.json", run_manifest)
    return run_manifest


def dry_run_actual_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Run one forward/backward head-only batch through the actual MAEST graph."""
    from maest_infer import get_maest

    raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(raw_state, Mapping):
        raise ValueError("MAEST checkpoint must contain a raw state dictionary")
    if tuple(raw_state["head.1.weight"].shape) == (519, 768):
        expanded_state = expand_classifier_state_dict(raw_state)
    elif tuple(raw_state["head.1.weight"].shape) == (522, 768):
        expanded_state = {key: value.clone() for key, value in raw_state.items()}
    else:
        raise ValueError("checkpoint classifier must contain 519 or 522 rows")
    backbone = get_maest(
        "discogs-maest-30s-pw-129e-519l",
        pretrained=False,
    )
    model = ContinualMaest522(backbone, expanded_state)
    apply_training_stage(model, TrainingStage.EXTENSION_HEADS)
    model.train()
    generator = torch.Generator().manual_seed(522)
    mel = torch.randn((1, 1, 96, 1875), generator=generator)
    output = model(mel)
    (output.new_logits.sum() + output.auxiliary_new_logits.sum()).backward()
    legacy_gradients = [
        parameter.grad
        for parameter in model.legacy_head.parameters()
    ]
    return {
        "logits_shape": list(output.logits.shape),
        "legacy_logits_shape": list(output.legacy_logits.shape),
        "new_logits_shape": list(output.new_logits.shape),
        "legacy_gradients_absent": all(
            gradient is None for gradient in legacy_gradients
        ),
        "extension_gradients_present": all(
            parameter.grad is not None
            for module in (model.extension_head, model.extension_dist_head)
            for parameter in module.parameters()
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or initialize staged hybrid continual MAEST fine-tuning."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument(
        "--dry-run-checkpoint",
        type=Path,
        help="Run one synthetic head-only batch through an actual MAEST checkpoint.",
    )
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="Validate immutable inputs and write reproducibility manifests.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.dry_run_checkpoint is not None:
        report = dry_run_actual_checkpoint(args.dry_run_checkpoint)
        LOGGER.info("MAEST dry run: %s", json.dumps(report, sort_keys=True))
        return 0
    config = TrainingConfig.from_json(args.config)
    run_manifest = initialize_run(config)
    if not args.initialize_only:
        LOGGER.warning(
            "Run initialized; invoke the staged training API with the resolved "
            "train and validation loaders."
        )
    LOGGER.info(
        "Initialized training run in %s with seed %s",
        config.output_dir,
        run_manifest["seed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
