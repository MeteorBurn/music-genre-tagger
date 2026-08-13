"""Deterministic model and metadata-only dataset card generation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from jinja2 import Environment, StrictUndefined

from .constants import NEW_LABELS


REQUIRED_METADATA = (
    "version",
    "source_checkpoint",
    "source_sha256",
    "license_review_status",
    "evaluation",
    "parity",
    "dataset_summary",
    "split_audit_sha256",
    "annotation_protocol",
    "intended_use",
    "limitations",
    "audio_rights",
)


@dataclass(frozen=True)
class CardReport:
    model_card: Path
    dataset_card: Path


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def render_release_cards(
    metadata: Mapping[str, Any],
    output_dir: Path,
) -> CardReport:
    """Render cards from complete release metadata with strict placeholders."""
    for key in REQUIRED_METADATA:
        if key not in metadata or metadata[key] in (None, ""):
            raise ValueError(f"release metadata is missing required field {key}")
    template_dir = Path(__file__).with_name("templates")
    environment = Environment(
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    context = dict(metadata)
    context.update(
        {
            "new_labels": NEW_LABELS,
            "evaluation_json": json.dumps(
                metadata["evaluation"],
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            "parity_json": json.dumps(
                metadata["parity"],
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            "dataset_summary_json": json.dumps(
                metadata["dataset_summary"],
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
        }
    )
    model_template = environment.from_string(
        (template_dir / "model-card.md.j2").read_text(encoding="utf-8")
    )
    dataset_template = environment.from_string(
        (template_dir / "dataset-card.md.j2").read_text(encoding="utf-8")
    )
    model_path = Path(output_dir) / "model" / "README.md"
    dataset_path = Path(output_dir) / "dataset" / "README.md"
    _atomic_text(model_path, model_template.render(**context))
    _atomic_text(dataset_path, dataset_template.render(**context))
    if "{{" in model_path.read_text(encoding="utf-8") or "{{" in dataset_path.read_text(encoding="utf-8"):
        raise ValueError("rendered release card contains an unfilled template value")
    return CardReport(model_card=model_path, dataset_card=dataset_path)
