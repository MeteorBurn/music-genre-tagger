"""Release-file writer for the exact custom MAEST Hugging Face frontend."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


PREPROCESSOR_CONFIG: dict[str, Any] = {
    "auto_map": {
        "AutoFeatureExtractor": "feature_extraction_maest.MAESTFeatureExtractor"
    },
    "do_normalize": True,
    "feature_extractor_type": "MAESTFeatureExtractor",
    "feature_size": 1,
    "hop_length": 256,
    "log_compression": "logC",
    "max_length": 1876,
    "mean": 2.06755686098554,
    "n_fft": 512,
    "num_mel_bins": 96,
    "padding_side": "right",
    "padding_value": 0.0,
    "return_attention_mask": False,
    "sampling_rate": 16_000,
    "std": 1.268292820667291,
}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_feature_extractor_release(output_dir: Path) -> dict[str, Any]:
    """Write deterministic remote code and preprocessor configuration."""
    resolved_output = Path(output_dir)
    source_path = Path(__file__).with_name("feature_extraction_maest.py")
    _atomic_bytes(
        resolved_output / "feature_extraction_maest.py",
        source_path.read_bytes(),
    )
    _atomic_bytes(
        resolved_output / "preprocessor_config.json",
        (
            json.dumps(
                PREPROCESSOR_CONFIG,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return dict(PREPROCESSOR_CONFIG)
