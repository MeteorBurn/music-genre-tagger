"""Continual-learning objective for protecting legacy MAEST behavior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from .continual_model import ContinualOutput


@dataclass(frozen=True)
class LossWeights:
    new: float = 1.0
    distill: float = 2.0
    l2sp: float = 1e-4
    auxiliary: float = 0.25

    def __post_init__(self) -> None:
        if min(self.new, self.distill, self.l2sp, self.auxiliary) < 0.0:
            raise ValueError("loss weights must be non-negative")


@dataclass(frozen=True)
class ContinualLosses:
    total: Tensor
    new_label: Tensor
    legacy_distillation: Tensor
    l2_starting_point: Tensor
    auxiliary: Tensor


def masked_weighted_bce(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    positive_weights: Tensor,
) -> Tensor:
    """Compute BCE only for explicitly positive or negative label cells."""
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("logits, targets, and mask must have identical shapes")
    if logits.ndim < 1 or positive_weights.shape != (logits.shape[-1],):
        raise ValueError("positive_weights must match the final logits dimension")
    if torch.any(mask < 0) or torch.any(mask > 1):
        raise ValueError("supervision mask values must be between zero and one")
    if torch.count_nonzero(mask).item() == 0:
        return logits.sum() * 0.0
    weights = positive_weights.to(device=logits.device, dtype=logits.dtype)
    elementwise = functional.binary_cross_entropy_with_logits(
        logits,
        targets.to(device=logits.device, dtype=logits.dtype),
        pos_weight=weights,
        reduction="none",
    )
    resolved_mask = mask.to(device=logits.device, dtype=logits.dtype)
    return (elementwise * resolved_mask).sum() / resolved_mask.sum()


def compute_positive_weights(targets: Tensor, mask: Tensor) -> Tensor:
    """Compute clipped positive weights from supervised training cells only."""
    if targets.shape != mask.shape or targets.ndim != 2:
        raise ValueError("training targets and masks must be matching matrices")
    resolved_targets = targets.to(dtype=torch.float32)
    resolved_mask = mask.to(dtype=torch.float32)
    positive_count = (resolved_targets * resolved_mask).sum(dim=0)
    negative_count = resolved_mask.sum(dim=0) - positive_count
    fallback = torch.full_like(positive_count, 10.0)
    ratio = torch.where(
        positive_count > 0,
        negative_count / positive_count.clamp_min(1.0),
        fallback,
    )
    return ratio.clamp(min=1.0, max=10.0)


def l2_starting_point(
    trainable_parameters: Mapping[str, Tensor],
    reference_parameters: Mapping[str, Tensor],
) -> Tensor:
    """Mean squared distance from the immutable pretrained starting point."""
    if set(trainable_parameters) != set(reference_parameters):
        raise ValueError("trainable and reference parameters must have the same names")
    if not trainable_parameters:
        return torch.tensor(0.0)
    squared_sum: Tensor | None = None
    element_count = 0
    for name in sorted(trainable_parameters):
        parameter = trainable_parameters[name]
        reference = reference_parameters[name]
        if parameter.shape != reference.shape:
            raise ValueError(
                f"parameter {name!r} shape differs from its L2-SP reference"
            )
        difference = parameter - reference.detach().to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        current_sum = difference.square().sum()
        squared_sum = current_sum if squared_sum is None else squared_sum + current_sum
        element_count += parameter.numel()
    if squared_sum is None or element_count == 0:
        raise ValueError("L2-SP parameters must contain at least one element")
    return squared_sum / element_count


def continual_loss(
    output: ContinualOutput,
    new_targets: Tensor,
    new_mask: Tensor,
    teacher_legacy_logits: Tensor,
    trainable_parameters: Mapping[str, Tensor],
    reference_parameters: Mapping[str, Tensor],
    positive_weights: Tensor | None = None,
    weights: LossWeights = LossWeights(),
) -> ContinualLosses:
    """Combine manual supervision, soft replay, L2-SP, and dist-token loss."""
    if output.legacy_logits.shape != teacher_legacy_logits.shape:
        raise ValueError("student and teacher legacy logits must have identical shapes")
    if positive_weights is None:
        positive_weights = torch.ones(
            output.new_logits.shape[-1],
            device=output.new_logits.device,
            dtype=output.new_logits.dtype,
        )
    new_label_loss = masked_weighted_bce(
        output.new_logits,
        new_targets,
        new_mask,
        positive_weights,
    )
    legacy_distillation = functional.smooth_l1_loss(
        output.legacy_logits,
        teacher_legacy_logits.detach().to(
            device=output.legacy_logits.device,
            dtype=output.legacy_logits.dtype,
        ),
    )
    if trainable_parameters:
        l2sp_loss = l2_starting_point(
            trainable_parameters,
            reference_parameters,
        )
    else:
        if reference_parameters:
            raise ValueError("L2-SP references were provided without trainable parameters")
        l2sp_loss = output.new_logits.sum() * 0.0
    auxiliary_loss = masked_weighted_bce(
        output.auxiliary_new_logits,
        new_targets,
        new_mask,
        positive_weights,
    )
    total = (
        weights.new * new_label_loss
        + weights.distill * legacy_distillation
        + weights.l2sp * l2sp_loss
        + weights.auxiliary * auxiliary_loss
    )
    return ContinualLosses(
        total=total,
        new_label=new_label_loss,
        legacy_distillation=legacy_distillation,
        l2_starting_point=l2sp_loss,
        auxiliary=auxiliary_loss,
    )
