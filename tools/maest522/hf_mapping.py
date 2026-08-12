"""Strict native MAEST to standard Hugging Face AST tensor conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from transformers import ASTConfig, ASTForAudioClassification


def build_ast_config(
    labels: Sequence[str],
    *,
    num_mel_bins: int = 96,
    max_length: int = 1876,
    patch_size: int = 16,
    frequency_stride: int = 10,
    time_stride: int = 10,
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
) -> ASTConfig:
    """Build the exact public 522-label AST representation."""
    resolved_labels = tuple(labels)
    if len(resolved_labels) != 522 or len(set(resolved_labels)) != 522:
        raise ValueError("Hugging Face AST config requires 522 unique labels")
    frequency_patches = (num_mel_bins - patch_size) // frequency_stride + 1
    time_patches = (max_length - patch_size) // time_stride + 1
    if frequency_patches <= 0 or time_patches <= 0:
        raise ValueError("AST convolution settings produce no patches")
    config = ASTConfig(
        num_labels=522,
        num_mel_bins=num_mel_bins,
        max_length=max_length,
        frequency_stride=frequency_stride,
        time_stride=time_stride,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        problem_type="multi_label_classification",
        id2label={index: label for index, label in enumerate(resolved_labels)},
        label2id={label: index for index, label in enumerate(resolved_labels)},
    )
    config.base_model_name_or_path = "mtg-upf/discogs-maest-30s-pw-129e-519l"
    config.maest_expected_patch_count = frequency_patches * time_patches
    return config


def _require(
    state: Mapping[str, Tensor],
    consumed: set[str],
    key: str,
) -> Tensor:
    value = state.get(key)
    if not isinstance(value, Tensor):
        raise ValueError(f"native checkpoint is missing tensor {key!r}")
    consumed.add(key)
    return value


def convert_native_to_hf_state(
    native_state: Mapping[str, Tensor],
    config: ASTConfig,
) -> dict[str, Tensor]:
    """Map every AST-consumed native tensor and reject unrecognized tensors."""
    consumed: set[str] = set()
    output: dict[str, Tensor] = {}
    prefix = "audio_spectrogram_transformer"
    output[f"{prefix}.embeddings.cls_token"] = _require(
        native_state, consumed, "cls_token"
    ).clone()
    output[f"{prefix}.embeddings.distillation_token"] = _require(
        native_state, consumed, "dist_token"
    ).clone()
    token_position = _require(native_state, consumed, "new_pos_embed")
    frequency_position = _require(native_state, consumed, "freq_new_pos_embed")
    time_position = _require(native_state, consumed, "time_new_pos_embed")
    try:
        patch_position = (frequency_position + time_position).flatten(2).transpose(1, 2)
    except RuntimeError as error:
        raise ValueError("native frequency/time positional tensors cannot be composed") from error
    position = torch.cat((token_position, patch_position), dim=1)
    expected_position_count = 2 + int(config.maest_expected_patch_count)
    expected_position_shape = (1, expected_position_count, config.hidden_size)
    if tuple(position.shape) != expected_position_shape:
        raise ValueError(
            f"composed position embedding has shape {tuple(position.shape)}; "
            f"expected {expected_position_shape}"
        )
    output[f"{prefix}.embeddings.position_embeddings"] = position.clone()
    output[f"{prefix}.embeddings.patch_embeddings.projection.weight"] = _require(
        native_state, consumed, "patch_embed.proj.weight"
    ).clone()
    output[f"{prefix}.embeddings.patch_embeddings.projection.bias"] = _require(
        native_state, consumed, "patch_embed.proj.bias"
    ).clone()

    for block_index in range(config.num_hidden_layers):
        native_prefix = f"blocks.{block_index}"
        hf_prefix = f"{prefix}.encoder.layer.{block_index}"
        qkv_weight = _require(
            native_state,
            consumed,
            f"{native_prefix}.attn.qkv.weight",
        )
        qkv_bias = _require(
            native_state,
            consumed,
            f"{native_prefix}.attn.qkv.bias",
        )
        if qkv_weight.shape[0] % 3 or qkv_bias.shape[0] % 3:
            raise ValueError("native QKV output dimension must be divisible by three")
        for name, weight, bias in zip(
            ("query", "key", "value"),
            qkv_weight.chunk(3, dim=0),
            qkv_bias.chunk(3, dim=0),
        ):
            output[f"{hf_prefix}.attention.attention.{name}.weight"] = weight.clone()
            output[f"{hf_prefix}.attention.attention.{name}.bias"] = bias.clone()
        direct_mapping = {
            f"{native_prefix}.attn.proj.weight": f"{hf_prefix}.attention.output.dense.weight",
            f"{native_prefix}.attn.proj.bias": f"{hf_prefix}.attention.output.dense.bias",
            f"{native_prefix}.mlp.fc1.weight": f"{hf_prefix}.intermediate.dense.weight",
            f"{native_prefix}.mlp.fc1.bias": f"{hf_prefix}.intermediate.dense.bias",
            f"{native_prefix}.mlp.fc2.weight": f"{hf_prefix}.output.dense.weight",
            f"{native_prefix}.mlp.fc2.bias": f"{hf_prefix}.output.dense.bias",
            f"{native_prefix}.norm1.weight": f"{hf_prefix}.layernorm_before.weight",
            f"{native_prefix}.norm1.bias": f"{hf_prefix}.layernorm_before.bias",
            f"{native_prefix}.norm2.weight": f"{hf_prefix}.layernorm_after.weight",
            f"{native_prefix}.norm2.bias": f"{hf_prefix}.layernorm_after.bias",
        }
        for native_key, hf_key in direct_mapping.items():
            output[hf_key] = _require(native_state, consumed, native_key).clone()

    final_mapping = {
        "norm.weight": f"{prefix}.layernorm.weight",
        "norm.bias": f"{prefix}.layernorm.bias",
        "head.0.weight": "classifier.layernorm.weight",
        "head.0.bias": "classifier.layernorm.bias",
        "head.1.weight": "classifier.dense.weight",
        "head.1.bias": "classifier.dense.bias",
    }
    for native_key, hf_key in final_mapping.items():
        output[hf_key] = _require(native_state, consumed, native_key).clone()

    ignored = {"head_dist.weight", "head_dist.bias"}
    for key in ignored:
        _require(native_state, consumed, key)
    unexpected = sorted(set(native_state) - consumed)
    if unexpected:
        raise ValueError("unexpected native tensors: " + ", ".join(unexpected))
    return output


def load_hf_model_strict(
    native_state: Mapping[str, Tensor],
    config: ASTConfig,
) -> ASTForAudioClassification:
    """Instantiate standard AST and demand exact converted key/shape coverage."""
    model = ASTForAudioClassification(config)
    converted = convert_native_to_hf_state(native_state, config)
    incompatibility = model.load_state_dict(converted, strict=False)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError(
            "Hugging Face strict mapping failed: "
            f"missing={incompatibility.missing_keys}, "
            f"unexpected={incompatibility.unexpected_keys}"
        )
    return model.eval()
