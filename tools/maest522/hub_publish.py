"""Explicit validation-first Hugging Face Hub publication workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

from huggingface_hub import HfApi

from .release import MODEL_ALLOWLIST

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HubValidationReport:
    release_dir: Path
    file_count: int
    provenance_sha256: str
    parity_windows: int
    parity_tracks: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_for_hub(release_dir: Path) -> HubValidationReport:
    """Validate staged content without constructing a Hub client."""
    resolved = Path(release_dir).resolve()
    if not resolved.is_dir():
        raise ValueError(f"release directory is unavailable: {resolved}")
    actual_names = {path.name for path in resolved.iterdir() if path.is_file()}
    if actual_names != MODEL_ALLOWLIST:
        missing = sorted(MODEL_ALLOWLIST - actual_names)
        unexpected = sorted(actual_names - MODEL_ALLOWLIST)
        raise ValueError(
            f"release allowlist mismatch: missing={missing}, unexpected={unexpected}"
        )
    checksum_lines = (resolved / "SHA256SUMS").read_text(
        encoding="utf-8"
    ).splitlines()
    checksum_entries: dict[str, str] = {}
    for line in checksum_lines:
        digest, separator, filename = line.partition("  ")
        if not separator or len(digest) != 64 or not filename:
            raise ValueError(f"invalid SHA256SUMS entry: {line}")
        checksum_entries[filename] = digest
    expected_names = MODEL_ALLOWLIST - {"SHA256SUMS"}
    if set(checksum_entries) != expected_names:
        raise ValueError("SHA256SUMS file coverage differs from the release allowlist")
    for filename, expected_digest in checksum_entries.items():
        actual_digest = _sha256_file(resolved / filename)
        if actual_digest != expected_digest:
            raise ValueError(
                f"SHA256 mismatch for {filename}: expected {expected_digest}, got {actual_digest}"
            )
    readme = (resolved / "README.md").read_text(encoding="utf-8")
    if "license: other" not in readme:
        raise ValueError("model card must retain license: other until license review")
    evaluation = json.loads(
        (resolved / "evaluation.json").read_text(encoding="utf-8")
    )
    if not evaluation.get("legacy_gates_passed") or not evaluation.get(
        "test_evaluated"
    ):
        raise ValueError("release evaluation gates are incomplete")
    parity = json.loads((resolved / "parity.json").read_text(encoding="utf-8"))
    if (
        not parity.get("passed")
        or int(parity.get("windows", 0)) < 30
        or int(parity.get("tracks", 0)) < 10
    ):
        raise ValueError("release parity is not publishable")
    return HubValidationReport(
        release_dir=resolved,
        file_count=len(actual_names),
        provenance_sha256=_sha256_file(resolved / "provenance.json"),
        parity_windows=int(parity["windows"]),
        parity_tracks=int(parity["tracks"]),
    )


def publish_model(
    release_dir: Path,
    repo_id: str,
    private: bool,
    revision: str,
    version: str,
) -> str:
    """Create/update one model repo without deleting any remote content."""
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id must be provided as NAMESPACE/NAME")
    validation = validate_for_hub(release_dir)
    api = HfApi()
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(validation.release_dir),
        revision=revision,
        commit_message=(
            f"Release MAEST 522 {version} "
            f"({validation.provenance_sha256[:12]})"
        ),
    )
    return str(repo_url)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or explicitly publish an audited MAEST 522 release."
    )
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the repository as public after the required license review.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    validation = validate_for_hub(args.release_dir)
    LOGGER.info(
        "Validated %s files; parity=%s windows/%s tracks",
        validation.file_count,
        validation.parity_windows,
        validation.parity_tracks,
    )
    if not args.push:
        return 0
    if not args.repo_id:
        raise ValueError("--repo-id is required together with --push")
    url = publish_model(
        args.release_dir,
        args.repo_id,
        private=not args.public,
        revision=args.revision,
        version=args.version,
    )
    LOGGER.info("Uploaded release to %s", url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
