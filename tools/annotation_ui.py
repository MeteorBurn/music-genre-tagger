"""Command-line entry point for the local MAEST 522 annotation UI."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.maest522.annotation_api import create_app  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local MAEST 522 audio annotation UI."
    )
    parser.add_argument("--db", type=Path, required=True, help="Annotation SQLite path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8522)
    parser.add_argument("--fpcalc", type=Path, default=Path("fpcalc"))
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    app = create_app(
        args.db,
        fpcalc_path=args.fpcalc,
        ffmpeg_path=args.ffmpeg,
    )
    logging.info("MAEST 522 annotation UI: http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
