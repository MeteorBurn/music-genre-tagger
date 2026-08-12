"""Shared constants for the MAEST 522 annotation and training workflows."""

NEW_LABELS = (
    "Electronic---Minimal-Deep-Tech",
    "Electronic---Microhouse",
    "Electronic---RoMinimal",
)

DEFAULT_POSITIVE_TARGET = 1_000
DEFAULT_NEGATIVE_TARGET = 1_000

REVIEW_STATES = {"positive", "negative", "uncertain", "unreviewed"}
CANDIDATE_ROLES = {
    "positive_candidate",
    "hard_negative_candidate",
    "unlabeled_pool",
}
SPLITS = ("train", "val", "test")
ROUND_SIZE_PER_LABEL = 100
MAX_CANDIDATES_PER_LABEL = 1_000
SCHEMA_VERSION = 1
