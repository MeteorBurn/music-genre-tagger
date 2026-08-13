from unittest import TestCase

import torch
from transformers import ASTForAudioClassification

from tools.maest522.hf_mapping import (
    build_ast_config,
    convert_native_to_hf_state,
    load_hf_model_strict,
)
from tools.maest522.model_labels import build_522_labels, load_official_519_labels


def tiny_native_state() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(522)
    hidden = 4
    intermediate = 8
    state = {
        "cls_token": torch.randn(1, 1, hidden, generator=generator),
        "dist_token": torch.randn(1, 1, hidden, generator=generator),
        "new_pos_embed": torch.randn(1, 2, hidden, generator=generator),
        "freq_new_pos_embed": torch.randn(1, hidden, 7, 1, generator=generator),
        "time_new_pos_embed": torch.randn(1, hidden, 1, 12, generator=generator),
        "patch_embed.proj.weight": torch.randn(hidden, 1, 4, 4, generator=generator),
        "patch_embed.proj.bias": torch.randn(hidden, generator=generator),
        "norm.weight": torch.randn(hidden, generator=generator),
        "norm.bias": torch.randn(hidden, generator=generator),
        "head.0.weight": torch.randn(hidden, generator=generator),
        "head.0.bias": torch.randn(hidden, generator=generator),
        "head.1.weight": torch.randn(522, hidden, generator=generator),
        "head.1.bias": torch.randn(522, generator=generator),
        "head_dist.weight": torch.randn(522, hidden, generator=generator),
        "head_dist.bias": torch.randn(522, generator=generator),
    }
    for index in range(2):
        prefix = f"blocks.{index}"
        state.update(
            {
                f"{prefix}.norm1.weight": torch.randn(hidden, generator=generator),
                f"{prefix}.norm1.bias": torch.randn(hidden, generator=generator),
                f"{prefix}.attn.qkv.weight": torch.randn(hidden * 3, hidden, generator=generator),
                f"{prefix}.attn.qkv.bias": torch.randn(hidden * 3, generator=generator),
                f"{prefix}.attn.proj.weight": torch.randn(hidden, hidden, generator=generator),
                f"{prefix}.attn.proj.bias": torch.randn(hidden, generator=generator),
                f"{prefix}.norm2.weight": torch.randn(hidden, generator=generator),
                f"{prefix}.norm2.bias": torch.randn(hidden, generator=generator),
                f"{prefix}.mlp.fc1.weight": torch.randn(intermediate, hidden, generator=generator),
                f"{prefix}.mlp.fc1.bias": torch.randn(intermediate, generator=generator),
                f"{prefix}.mlp.fc2.weight": torch.randn(hidden, intermediate, generator=generator),
                f"{prefix}.mlp.fc2.bias": torch.randn(hidden, generator=generator),
            }
        )
    return state


class HuggingFaceMappingTests(TestCase):
    def _config(self):
        labels = tuple(f"label-{index}" for index in range(522))
        return build_ast_config(
            labels,
            num_mel_bins=16,
            max_length=26,
            patch_size=4,
            frequency_stride=2,
            time_stride=2,
            hidden_size=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=8,
        )

    def test_maps_positions_qkv_blocks_and_classifier(self) -> None:
        native = tiny_native_state()
        config = self._config()

        converted = convert_native_to_hf_state(native, config)

        expected_patch_position = (
            native["freq_new_pos_embed"] + native["time_new_pos_embed"]
        ).flatten(2).transpose(1, 2)
        expected_position = torch.cat(
            (native["new_pos_embed"], expected_patch_position),
            dim=1,
        )
        torch.testing.assert_close(
            converted[
                "audio_spectrogram_transformer.embeddings.position_embeddings"
            ],
            expected_position,
        )
        qkv_weight = native["blocks.0.attn.qkv.weight"].chunk(3, dim=0)
        for name, expected in zip(("query", "key", "value"), qkv_weight):
            torch.testing.assert_close(
                converted[
                    "audio_spectrogram_transformer.encoder.layer.0."
                    f"attention.attention.{name}.weight"
                ],
                expected,
            )
        torch.testing.assert_close(
            converted["classifier.dense.weight"],
            native["head.1.weight"],
        )
        self.assertEqual(tuple(expected_position.shape), (1, 86, 4))

    def test_strict_model_load_has_no_missing_or_unexpected_keys(self) -> None:
        model = load_hf_model_strict(tiny_native_state(), self._config())

        self.assertIsInstance(model, ASTForAudioClassification)
        self.assertEqual(model.config.num_labels, 522)
        self.assertEqual(model.config.problem_type, "multi_label_classification")

    def test_public_config_uses_the_final_extension_label_order(self) -> None:
        labels = build_522_labels(load_official_519_labels())

        config = build_ast_config(labels)

        self.assertEqual(
            tuple(config.id2label[index] for index in (519, 520, 521)),
            (
                "Electronic---Minimal-Deep-Tech",
                "Electronic---Microhouse",
                "Electronic---RoMinimal",
            ),
        )
        self.assertEqual(config.label2id["Electronic---Minimal-Deep-Tech"], 519)
        self.assertEqual(config.label2id["Electronic---Microhouse"], 520)
        self.assertEqual(config.label2id["Electronic---RoMinimal"], 521)

    def test_rejects_missing_unexpected_and_shape_mismatched_native_keys(self) -> None:
        native = tiny_native_state()
        del native["blocks.0.attn.qkv.weight"]
        with self.assertRaisesRegex(ValueError, "qkv.weight"):
            convert_native_to_hf_state(native, self._config())
        native = tiny_native_state()
        native["blocks.0.attn.qkv.weight"] = torch.zeros(11, 4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            convert_native_to_hf_state(native, self._config())
        native = tiny_native_state()
        native["unexpected.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "unexpected native"):
            convert_native_to_hf_state(native, self._config())
