"""Strict loader for the separately named native MAEST 522 artifact."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from maest_infer import get_maest
from torch import Tensor, nn

from .model_labels import build_522_labels, load_official_519_labels


def load_native_state(checkpoint_path: Path) -> dict[str, Tensor]:
    """Load and validate the state dictionary inside a native release artifact."""
    artifact = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(artifact, Mapping):
        raise ValueError("native MAEST artifact must be a mapping")
    raw_state = artifact.get("state_dict", artifact)
    if not isinstance(raw_state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in raw_state.items()
    ):
        raise ValueError("native MAEST artifact has no valid tensor state dictionary")
    state = {str(key): value for key, value in raw_state.items()}
    expected_shapes = {
        "head.1.weight": (522, 768),
        "head.1.bias": (522,),
        "head_dist.weight": (522, 768),
        "head_dist.bias": (522,),
    }
    for key, shape in expected_shapes.items():
        if key not in state or tuple(state[key].shape) != shape:
            actual = tuple(state[key].shape) if key in state else None
            raise ValueError(
                f"native MAEST tensor {key!r} has shape {actual}; expected {shape}"
            )
    return state


def get_maest_522(
    checkpoint_path: Path,
    device: str = "cpu",
) -> nn.Module:
    """Strictly load a native 522-label MAEST without altering the 519 factory."""
    model = get_maest(
        "discogs-maest-30s-pw-129e-519l",
        pretrained=False,
    )
    if not isinstance(model.head, nn.Sequential) or len(model.head) < 2:
        raise ValueError("installed MAEST model does not expose head[1]")
    model.head[1] = nn.Linear(768, 522)
    model.head_dist = nn.Linear(768, 522)
    model.num_classes = 522
    model.labels = list(build_522_labels(load_official_519_labels()))
    state = load_native_state(checkpoint_path)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def predict_522(model: nn.Module, mel: Tensor) -> Tensor:
    """Return 522 sigmoid probabilities for an already computed MAEST mel batch."""
    with torch.no_grad():
        result = model(mel, melspectrogram_input=True)
        logits = result[0] if isinstance(result, tuple) else result
        if not isinstance(logits, Tensor) or logits.shape[-1] != 522:
            raise RuntimeError("native MAEST 522 model returned an invalid logit shape")
        return torch.sigmoid(logits)
