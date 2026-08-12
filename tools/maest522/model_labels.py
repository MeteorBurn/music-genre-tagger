"""Canonical label handling for the MAEST 519-to-522 extension."""

from collections.abc import Sequence

from maest_infer.discogs_labels import discogs_519labels

from .constants import NEW_LABELS


def load_official_519_labels() -> tuple[str, ...]:
    """Load and validate the label ordering shipped by ``maest-infer``."""
    labels = tuple(discogs_519labels)
    if len(labels) != 519:
        raise ValueError(
            "maest-infer must provide exactly 519 official Discogs labels; "
            f"received {len(labels)}"
        )
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("official Discogs labels must be non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("official Discogs labels must be unique")
    return labels


def build_522_labels(labels_519: Sequence[str]) -> tuple[str, ...]:
    """Append the extension labels without changing the legacy label order."""
    legacy_labels = tuple(labels_519)
    if len(legacy_labels) != 519:
        raise ValueError(
            "the legacy label sequence must contain exactly 519 entries; "
            f"received {len(legacy_labels)}"
        )
    if any(not isinstance(label, str) or not label for label in legacy_labels):
        raise ValueError("legacy labels must be non-empty strings")
    if len(set(legacy_labels)) != len(legacy_labels):
        raise ValueError("legacy labels must be unique")
    overlap = set(legacy_labels).intersection(NEW_LABELS)
    if overlap:
        raise ValueError(
            "legacy label sequence already contains extension labels: "
            + ", ".join(sorted(overlap))
        )
    return legacy_labels + NEW_LABELS
