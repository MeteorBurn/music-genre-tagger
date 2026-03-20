#!/usr/bin/env python3

import argparse
from pathlib import Path

from config import get_config
from environment import run_environment_checks
from pipeline import apply_cli_overrides
from pipeline import DEFAULT_CHECKPOINT_FILENAME
from pipeline import DEFAULT_MODELS_DIR
from pipeline import run_pipeline
from pipeline import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MusicTagger one-flow pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "analyze", "excel", "tag"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--input-directory",
        default=None,
        help=(
            "Path to your audio library directory. "
            "Required for all stages unless set in src/config.py."
        ),
    )
    parser.add_argument(
        "--output-directory",
        default=None,
        help=(
            "Base directory for generated metadata projects. "
            "Pipeline creates a subfolder named after input directory."
        ),
    )
    parser.add_argument(
        "--file-pattern",
        default=None,
        help=(
            "Optional substring filter for filenames during analysis. "
            "Only matching tracks are processed."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of tracks analyzed in current run (test mode).",
    )
    parser.add_argument(
        "--convert-to-wav",
        action="store_true",
        help="Convert non-WAV tracks to temporary WAV before analysis.",
    )
    parser.add_argument(
        "--tag-yes",
        action="store_true",
        help="For --stage all: force tagging stage on without interactive prompt.",
    )
    parser.add_argument(
        "--tag-no",
        action="store_true",
        help="For --stage all: skip tagging stage without interactive prompt.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable all CLI prompts and require values from config/arguments.",
    )
    parser.add_argument(
        "--loglevel",
        default=None,
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = apply_cli_overrides(get_config(), args)

    setup_logging(config.get("loglevel", "INFO"))
    project_dir = Path(__file__).resolve().parent.parent
    check_ok = run_environment_checks(
        project_dir,
        DEFAULT_MODELS_DIR,
        DEFAULT_CHECKPOINT_FILENAME,
        str(config.get("model_file_path", "")),
    )
    if not check_ok:
        return 1

    return run_pipeline(config, args.stage, project_dir, args.non_interactive)


if __name__ == "__main__":
    raise SystemExit(main())
