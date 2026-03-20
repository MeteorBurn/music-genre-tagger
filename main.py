#!/usr/bin/env python3

import argparse
from pathlib import Path

from config import CONFIG
from environment import run_environment_checks
from pipeline import apply_cli_overrides
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
    parser.add_argument("--input-directory", default=None)
    parser.add_argument("--output-directory", default=None)
    parser.add_argument("--json-directory", default=None)
    parser.add_argument("--excel-path", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--file-pattern", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--convert-to-wav", action="store_true")
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--tag-yes", action="store_true")
    parser.add_argument("--tag-no", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--loglevel", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = apply_cli_overrides(CONFIG, args)

    setup_logging(config.get("loglevel", "INFO"))
    script_dir = Path(__file__).resolve().parent

    check_ok = run_environment_checks(
        script_dir,
        config.get("models_dir", "models"),
        config.get("maest_models", {})
        .get("maest_519l_pytorch", {})
        .get("checkpoint_filename", ""),
    )
    if not check_ok:
        return 1

    return run_pipeline(config, args.stage, script_dir, args.non_interactive)


if __name__ == "__main__":
    raise SystemExit(main())
