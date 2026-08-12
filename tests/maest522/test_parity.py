from types import SimpleNamespace
from unittest import TestCase

import torch
from torch import nn

from tools.maest522.parity import compare_native_and_hf, prepare_hf_input


class FakeNative(nn.Module):
    def forward(self, mel, melspectrogram_input=False):
        if not melspectrogram_input:
            raise AssertionError("expected precomputed mel")
        pooled = mel.mean(dim=(1, 2))
        logits = pooled[:, None] + torch.arange(522, dtype=mel.dtype)[None, :]
        return logits, pooled


class FakeHuggingFace(nn.Module):
    def forward(self, input_values):
        pooled = input_values.mean(dim=(1, 2))
        logits = pooled[:, None] + torch.arange(522, dtype=input_values.dtype)[None, :]
        return SimpleNamespace(logits=logits)


class MutatingNative(FakeNative):
    def forward(self, mel, melspectrogram_input=False):
        original = mel
        mel.unsqueeze_(1)
        return super().forward(original.squeeze(1), melspectrogram_input)


class ParityTests(TestCase):
    def test_transposes_native_mel_without_numeric_modification(self) -> None:
        native = torch.arange(2 * 96 * 10, dtype=torch.float32).reshape(2, 96, 10)

        hf = prepare_hf_input(native)

        self.assertEqual(tuple(hf.shape), (2, 10, 96))
        torch.testing.assert_close(hf.transpose(1, 2), native, rtol=0, atol=0)

    def test_exact_models_pass_raw_logit_probability_and_top10_gates(self) -> None:
        batches = [
            torch.randn(2, 96, 20, generator=torch.Generator().manual_seed(1)),
            torch.randn(1, 96, 20, generator=torch.Generator().manual_seed(2)),
        ]

        report = compare_native_and_hf(FakeNative(), FakeHuggingFace(), batches)

        self.assertEqual(report.windows, 3)
        self.assertLessEqual(report.max_absolute_logit_error, 1e-7)
        self.assertLessEqual(report.max_absolute_probability_error, 1e-7)
        self.assertEqual(report.top10_match_rate, 1.0)
        self.assertTrue(report.passed)

    def test_probability_is_sigmoid_of_raw_logits_exactly_once(self) -> None:
        class ShiftedHf(FakeHuggingFace):
            def forward(self, input_values):
                result = super().forward(input_values)
                result.logits = result.logits + 0.01
                return result

        mel = torch.zeros(1, 96, 20)

        report = compare_native_and_hf(
            FakeNative(),
            ShiftedHf(),
            [mel],
        )

        self.assertAlmostEqual(
            report.max_absolute_logit_error,
            float(torch.tensor(521.01, dtype=torch.float32) - torch.tensor(521.0)),
            places=7,
        )
        expected_probability_error = torch.max(
            torch.abs(
                torch.sigmoid(torch.arange(522, dtype=torch.float32))
                - torch.sigmoid(torch.arange(522, dtype=torch.float32) + 0.01)
            )
        ).item()
        self.assertAlmostEqual(
            report.max_absolute_probability_error,
            expected_probability_error,
            places=7,
        )
        self.assertFalse(report.passed)

    def test_native_in_place_shape_change_cannot_corrupt_hf_input(self) -> None:
        mel = torch.zeros(1, 96, 20)

        report = compare_native_and_hf(MutatingNative(), FakeHuggingFace(), [mel])

        self.assertTrue(report.passed)
        self.assertEqual(tuple(mel.shape), (1, 96, 20))
