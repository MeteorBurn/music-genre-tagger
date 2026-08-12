from unittest import TestCase

import torch
from torch import nn

from tools.maest522.continual_model import (
    ContinualMaest522,
    TrainingStage,
    apply_training_stage,
)


class FakeMaestBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])
        self.norm = nn.LayerNorm(4)
        self.head = nn.Sequential(nn.Identity(), nn.Linear(4, 519))
        self.head_dist = nn.Linear(4, 519)

    def forward_features(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return mel, mel * 2.0


def build_expanded_state_dict() -> dict[str, torch.Tensor]:
    legacy_weight = torch.arange(519 * 4, dtype=torch.float32).reshape(519, 4)
    extension_weight = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, -1.0, 1.0, -1.0]]
    )
    legacy_bias = torch.arange(519, dtype=torch.float32)
    extension_bias = torch.tensor([0.5, -0.5, 1.5])
    return {
        "head.1.weight": torch.cat((legacy_weight, extension_weight)),
        "head.1.bias": torch.cat((legacy_bias, extension_bias)),
        "head_dist.weight": torch.cat((legacy_weight + 10.0, extension_weight + 2.0)),
        "head_dist.bias": torch.cat((legacy_bias + 10.0, extension_bias + 2.0)),
    }


class ContinualModelTests(TestCase):
    def test_native_logits_preserve_order_and_isolate_legacy_gradients(self) -> None:
        state_dict = build_expanded_state_dict()
        model = ContinualMaest522(FakeMaestBackbone(), state_dict)
        batch = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            requires_grad=True,
        )

        output = model(batch)

        pooled = batch * 1.5
        expected_legacy = torch.nn.functional.linear(
            pooled,
            state_dict["head.1.weight"][:519],
            state_dict["head.1.bias"][:519],
        )
        expected_new = torch.nn.functional.linear(
            pooled,
            state_dict["head.1.weight"][519:],
            state_dict["head.1.bias"][519:],
        )
        expected_auxiliary = torch.nn.functional.linear(
            batch * 2.0,
            state_dict["head_dist.weight"][519:],
            state_dict["head_dist.bias"][519:],
        )
        self.assertEqual(tuple(output.logits.shape), (2, 522))
        self.assertEqual(tuple(output.legacy_logits.shape), (2, 519))
        self.assertEqual(tuple(output.new_logits.shape), (2, 3))
        torch.testing.assert_close(output.legacy_logits, expected_legacy)
        torch.testing.assert_close(output.new_logits, expected_new)
        torch.testing.assert_close(output.auxiliary_new_logits, expected_auxiliary)
        torch.testing.assert_close(output.logits[:, :519], output.legacy_logits)
        torch.testing.assert_close(output.logits[:, 519:], output.new_logits)

        (output.logits.sum() + output.auxiliary_new_logits.sum()).backward()
        self.assertIsNone(model.legacy_head.weight.grad)
        self.assertIsNotNone(model.extension_head.weight.grad)
        self.assertIsNotNone(model.extension_dist_head.weight.grad)
        self.assertFalse(model.legacy_head.weight.requires_grad)
        torch.testing.assert_close(
            model.legacy_dist_weight,
            state_dict["head_dist.weight"][:519],
            rtol=0,
            atol=0,
        )
        exported = model.export_maest_state_dict()
        self.assertEqual(tuple(exported["head.1.weight"].shape), (522, 4))
        self.assertEqual(tuple(exported["head_dist.weight"].shape), (522, 4))
        torch.testing.assert_close(
            exported["head.1.weight"][:519],
            state_dict["head.1.weight"][:519],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            exported["head_dist.weight"][:519],
            state_dict["head_dist.weight"][:519],
            rtol=0,
            atol=0,
        )

    def test_named_stages_unfreeze_only_intended_parameters(self) -> None:
        model = ContinualMaest522(FakeMaestBackbone(), build_expanded_state_dict())

        self.assertEqual(
            self._trainable_names(model),
            {
                "extension_head.weight",
                "extension_head.bias",
                "extension_dist_head.weight",
                "extension_dist_head.bias",
            },
        )

        apply_training_stage(model, TrainingStage.BLOCKS_10_11)
        stage_two = self._trainable_names(model)
        self.assertTrue(any(name.startswith("backbone.blocks.10.") for name in stage_two))
        self.assertTrue(any(name.startswith("backbone.blocks.11.") for name in stage_two))
        self.assertTrue(any(name.startswith("backbone.norm.") for name in stage_two))
        self.assertFalse(any(name.startswith("backbone.blocks.9.") for name in stage_two))
        self.assertFalse(any(name.startswith("legacy_head.") for name in stage_two))
        self.assertFalse(any(name.startswith("head_norm.") for name in stage_two))

        apply_training_stage(model, TrainingStage.BLOCKS_8_11)
        stage_three = self._trainable_names(model)
        for block_index in range(8, 12):
            self.assertTrue(
                any(
                    name.startswith(f"backbone.blocks.{block_index}.")
                    for name in stage_three
                )
            )
        self.assertFalse(any(name.startswith("backbone.blocks.7.") for name in stage_three))

    @staticmethod
    def _trainable_names(model: nn.Module) -> set[str]:
        return {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
