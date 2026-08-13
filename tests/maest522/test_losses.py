from unittest import TestCase

import torch
from torch.nn import functional as functional

from tools.maest522.continual_model import ContinualOutput
from tools.maest522.losses import (
    LossWeights,
    compute_positive_weights,
    continual_loss,
    l2_starting_point,
    masked_weighted_bce,
)


class ContinualLossTests(TestCase):
    def test_combines_masked_supervision_distillation_l2sp_and_auxiliary(self) -> None:
        new_logits = torch.tensor(
            [[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]],
            requires_grad=True,
        )
        auxiliary_logits = torch.tensor(
            [[0.5, 0.0, -0.5], [1.0, -1.0, 0.0]],
            requires_grad=True,
        )
        legacy_logits = torch.tensor(
            [[0.0, 0.5], [1.0, -0.5]],
            requires_grad=True,
        )
        teacher_logits = torch.tensor([[0.1, 0.4], [0.8, -0.4]])
        targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        positive_weights = torch.tensor([2.0, 3.0, 4.0])
        trainable = {"backbone.weight": torch.tensor([2.0, 4.0], requires_grad=True)}
        reference = {"backbone.weight": torch.tensor([1.0, 2.0])}
        output = ContinualOutput(
            logits=torch.cat((legacy_logits, new_logits), dim=-1),
            legacy_logits=legacy_logits,
            new_logits=new_logits,
            auxiliary_new_logits=auxiliary_logits,
            cls_embedding=torch.zeros(2, 4),
            dist_embedding=torch.zeros(2, 4),
        )
        weights = LossWeights(new=1.0, distill=2.0, l2sp=0.1, auxiliary=0.25)

        losses = continual_loss(
            output=output,
            new_targets=targets,
            new_mask=mask,
            teacher_legacy_logits=teacher_logits,
            trainable_parameters=trainable,
            reference_parameters=reference,
            positive_weights=positive_weights,
            weights=weights,
        )

        elementwise = functional.binary_cross_entropy_with_logits(
            new_logits,
            targets,
            pos_weight=positive_weights,
            reduction="none",
        )
        expected_new = (elementwise * mask).sum() / mask.sum()
        expected_distillation = functional.smooth_l1_loss(
            legacy_logits,
            teacher_logits,
        )
        expected_l2sp = torch.tensor(2.5)
        expected_auxiliary = masked_weighted_bce(
            auxiliary_logits,
            targets,
            mask,
            positive_weights,
        )
        expected_total = (
            expected_new
            + 2.0 * expected_distillation
            + 0.1 * expected_l2sp
            + 0.25 * expected_auxiliary
        )
        torch.testing.assert_close(losses.new_label, expected_new)
        torch.testing.assert_close(losses.legacy_distillation, expected_distillation)
        torch.testing.assert_close(losses.l2_starting_point, expected_l2sp)
        torch.testing.assert_close(losses.auxiliary, expected_auxiliary)
        torch.testing.assert_close(losses.total, expected_total)
        self.assertEqual(losses.total.ndim, 0)

        losses.total.backward()
        self.assertIsNotNone(new_logits.grad)
        self.assertIsNotNone(auxiliary_logits.grad)
        self.assertIsNotNone(legacy_logits.grad)
        self.assertIsNotNone(trainable["backbone.weight"].grad)

    def test_empty_supervision_returns_differentiable_zero(self) -> None:
        logits = torch.randn(2, 3, requires_grad=True)

        loss = masked_weighted_bce(
            logits,
            torch.zeros_like(logits),
            torch.zeros_like(logits),
            torch.ones(3),
        )

        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_positive_weights_use_train_mask_and_clip_to_approved_range(self) -> None:
        targets = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        mask = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        weights = compute_positive_weights(targets, mask)

        torch.testing.assert_close(weights, torch.tensor([3.0, 1.0, 10.0]))

    def test_l2sp_rejects_parameter_contract_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "same names"):
            l2_starting_point({"a": torch.zeros(1)}, {"b": torch.zeros(1)})
        with self.assertRaisesRegex(ValueError, "shape"):
            l2_starting_point({"a": torch.zeros(2)}, {"a": torch.zeros(1)})

    def test_masked_bce_passes_double_precision_gradcheck(self) -> None:
        logits = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        mask = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        positive_weights = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda values: masked_weighted_bce(
                    values,
                    targets,
                    mask,
                    positive_weights,
                ),
                (logits,),
            )
        )
