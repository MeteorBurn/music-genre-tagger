"""Native MAEST continual-learning wrapper for the 522-label output space."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn

from .constants import NEW_LABELS

LEGACY_LABEL_COUNT = 519
TOTAL_LABEL_COUNT = LEGACY_LABEL_COUNT + len(NEW_LABELS)


@dataclass(frozen=True)
class ContinualOutput:
    """Outputs used by native inference and continual-learning objectives."""

    logits: Tensor
    legacy_logits: Tensor
    new_logits: Tensor
    auxiliary_new_logits: Tensor
    cls_embedding: Tensor
    dist_embedding: Tensor


class TrainingStage(str, Enum):
    """Named gradual-unfreezing stages for hybrid continual fine-tuning."""

    EXTENSION_HEADS = "extension_heads"
    BLOCKS_10_11 = "blocks_10_11"
    BLOCKS_8_11 = "blocks_8_11"


class ContinualMaest522(nn.Module):
    """MAEST backbone with one ordered native 522-label classifier output."""

    def __init__(
        self,
        backbone: nn.Module,
        expanded_state_dict: Mapping[str, Tensor],
    ) -> None:
        super().__init__()
        self._validate_expanded_heads(expanded_state_dict)
        self.backbone = backbone

        backbone_state = {
            key: value
            for key, value in expanded_state_dict.items()
            if key not in {
                "head.1.weight",
                "head.1.bias",
                "head_dist.weight",
                "head_dist.bias",
            }
        }
        if backbone_state:
            self.backbone.load_state_dict(backbone_state, strict=False)

        backbone_head = getattr(self.backbone, "head", None)
        if not isinstance(backbone_head, nn.Sequential) or len(backbone_head) < 1:
            raise ValueError("MAEST backbone must expose head[0] normalization")
        self.head_norm = deepcopy(backbone_head[0])

        feature_count = int(expanded_state_dict["head.1.weight"].shape[1])
        self.legacy_head = nn.Linear(feature_count, LEGACY_LABEL_COUNT)
        self.extension_head = nn.Linear(feature_count, len(NEW_LABELS))
        self.extension_dist_head = nn.Linear(feature_count, len(NEW_LABELS))
        self._load_classifier_rows(expanded_state_dict)

        self.register_buffer(
            "legacy_dist_weight",
            expanded_state_dict["head_dist.weight"][:LEGACY_LABEL_COUNT].clone(),
        )
        self.register_buffer(
            "legacy_dist_bias",
            expanded_state_dict["head_dist.bias"][:LEGACY_LABEL_COUNT].clone(),
        )

        self.backbone.head = nn.Identity()
        self.backbone.head_dist = nn.Identity()
        apply_training_stage(self, TrainingStage.EXTENSION_HEADS)

    @staticmethod
    def _validate_expanded_heads(state_dict: Mapping[str, Tensor]) -> None:
        expected = {
            "head.1.weight": (TOTAL_LABEL_COUNT, 768),
            "head.1.bias": (TOTAL_LABEL_COUNT,),
            "head_dist.weight": (TOTAL_LABEL_COUNT, 768),
            "head_dist.bias": (TOTAL_LABEL_COUNT,),
        }
        feature_counts: set[int] = set()
        for key, default_shape in expected.items():
            tensor = state_dict.get(key)
            if not isinstance(tensor, Tensor):
                raise ValueError(f"expanded checkpoint is missing tensor {key!r}")
            if key.endswith("weight"):
                if tensor.ndim != 2 or tensor.shape[0] != TOTAL_LABEL_COUNT:
                    raise ValueError(
                        f"expanded tensor {key!r} must have 522 rows; "
                        f"received {tuple(tensor.shape)}"
                    )
                feature_counts.add(int(tensor.shape[1]))
            elif tuple(tensor.shape) != default_shape:
                raise ValueError(
                    f"expanded tensor {key!r} has shape {tuple(tensor.shape)}; "
                    f"expected {default_shape}"
                )
        if len(feature_counts) != 1:
            raise ValueError("expanded classifier weights must share a feature size")

    def _load_classifier_rows(self, state_dict: Mapping[str, Tensor]) -> None:
        with torch.no_grad():
            self.legacy_head.weight.copy_(
                state_dict["head.1.weight"][:LEGACY_LABEL_COUNT]
            )
            self.legacy_head.bias.copy_(
                state_dict["head.1.bias"][:LEGACY_LABEL_COUNT]
            )
            self.extension_head.weight.copy_(
                state_dict["head.1.weight"][LEGACY_LABEL_COUNT:]
            )
            self.extension_head.bias.copy_(
                state_dict["head.1.bias"][LEGACY_LABEL_COUNT:]
            )
            self.extension_dist_head.weight.copy_(
                state_dict["head_dist.weight"][LEGACY_LABEL_COUNT:]
            )
            self.extension_dist_head.bias.copy_(
                state_dict["head_dist.bias"][LEGACY_LABEL_COUNT:]
            )

    def forward(self, mel: Tensor) -> ContinualOutput:
        features = self.backbone.forward_features(mel)
        if not isinstance(features, tuple) or len(features) != 2:
            raise RuntimeError("MAEST backbone must return cls and dist embeddings")
        cls_embedding, dist_embedding = features
        pooled_embedding = self.head_norm((cls_embedding + dist_embedding) / 2.0)
        legacy_logits = self.legacy_head(pooled_embedding)
        new_logits = self.extension_head(pooled_embedding)
        auxiliary_new_logits = self.extension_dist_head(dist_embedding)
        return ContinualOutput(
            logits=torch.cat((legacy_logits, new_logits), dim=-1),
            legacy_logits=legacy_logits,
            new_logits=new_logits,
            auxiliary_new_logits=auxiliary_new_logits,
            cls_embedding=cls_embedding,
            dist_embedding=dist_embedding,
        )

    def export_maest_state_dict(self) -> dict[str, Tensor]:
        """Export the wrapper as a raw MAEST-compatible 522-label state dict."""
        state_dict = {
            key: value.clone()
            for key, value in self.backbone.state_dict().items()
        }
        for key, value in self.head_norm.state_dict().items():
            state_dict[f"head.0.{key}"] = value.clone()
        state_dict["head.1.weight"] = torch.cat(
            (self.legacy_head.weight.detach(), self.extension_head.weight.detach())
        ).clone()
        state_dict["head.1.bias"] = torch.cat(
            (self.legacy_head.bias.detach(), self.extension_head.bias.detach())
        ).clone()
        state_dict["head_dist.weight"] = torch.cat(
            (self.legacy_dist_weight, self.extension_dist_head.weight.detach())
        ).clone()
        state_dict["head_dist.bias"] = torch.cat(
            (self.legacy_dist_bias, self.extension_dist_head.bias.detach())
        ).clone()
        return state_dict


def apply_training_stage(
    model: ContinualMaest522,
    stage: TrainingStage,
) -> None:
    """Apply one exact gradual-unfreezing policy to a continual model."""
    resolved_stage = TrainingStage(stage)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for head in (model.extension_head, model.extension_dist_head):
        for parameter in head.parameters():
            parameter.requires_grad_(True)

    if resolved_stage is TrainingStage.EXTENSION_HEADS:
        return

    first_trainable_block = (
        10 if resolved_stage is TrainingStage.BLOCKS_10_11 else 8
    )
    blocks = getattr(model.backbone, "blocks", None)
    if not isinstance(blocks, (nn.ModuleList, nn.Sequential)) or len(blocks) < 12:
        raise ValueError("MAEST backbone must expose at least 12 transformer blocks")
    for block_index in range(first_trainable_block, 12):
        for parameter in blocks[block_index].parameters():
            parameter.requires_grad_(True)
    final_norm = getattr(model.backbone, "norm", None)
    if not isinstance(final_norm, nn.Module):
        raise ValueError("MAEST backbone must expose final normalization as norm")
    for parameter in final_norm.parameters():
        parameter.requires_grad_(True)
