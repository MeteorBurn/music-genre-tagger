import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable
from typing import NoReturn
from typing import Optional
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline import run_excel_stage
from pipeline import run_pipeline
from pipeline import run_tag_stage
from storage import init_db
from storage import upsert_track


def create_legacy_database(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE tracks (
                timestamp TEXT NOT NULL,
                hash TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                name TEXT NOT NULL,
                extension TEXT NOT NULL,
                megabytes REAL NOT NULL,
                bytes INTEGER NOT NULL,
                time TEXT NOT NULL,
                seconds REAL NOT NULL,
                labels TEXT NOT NULL,
                confidences TEXT NOT NULL,
                model TEXT NOT NULL,
                error TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO tracks VALUES (
                '2026-07-30T00:00:00',
                'legacy-hash',
                'C:/Music/legacy.flac',
                'legacy',
                '.flac',
                1.0,
                1048576,
                '00:01:00',
                60.0,
                '["House"]',
                '[0.9]',
                'maest_519l_pytorch',
                '',
                'analysis_success',
                '2026-07-30T00:00:00'
            )
            """
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()


def build_partial_track_payload() -> dict[str, object]:
    return {
        "timestamp": "2026-07-30T00:00:00",
        "hash": "partial-hash",
        "file": {
            "path": "C:/Music/partial.flac",
            "name": "partial",
            "extension": ".flac",
            "size": {"megabytes": 1.0, "bytes": 1048576},
            "duration": {"time": "00:01:00", "seconds": 60.0},
        },
        "genres": {
            "labels": ["House"],
            "confidences": [0.9],
            "model": "maest_519l_pytorch",
        },
        "analysis_config": {
            "audio_segment_offsets": [0.0, 15.0, 30.0],
            "audio_segment_duration": 30.0,
            "audio_segment_count": 3,
            "aggregation": "mean",
        },
    }


class PipelineStageSchemaTests(unittest.TestCase):
    def assert_legacy_schema_rejected(
        self, operation: Callable[[], object]
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "incompatible with schema version 2"):
            operation()

    def test_excel_stage_rejects_legacy_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "tracks.db"
            create_legacy_database(db_path)

            self.assert_legacy_schema_rejected(
                lambda: run_excel_stage({}, db_path, root / "genres.xlsx")
            )

    def test_tag_stage_validates_schema_before_tagger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "tracks.db"
            create_legacy_database(db_path)

            with patch(
                "pipeline.run_genre_tagging",
                side_effect=AssertionError("tagger must not run"),
            ):
                self.assert_legacy_schema_rejected(
                    lambda: run_tag_stage(
                        {"tagger": {}},
                        db_path,
                        root / "genres.xlsx",
                    )
                )


class PipelineFailureFinalizationTests(unittest.TestCase):
    def assert_pipeline_writes_original_schema_error(
        self, stage: str, write_json: bool
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "music"
            output_dir = root / "output"
            input_dir.mkdir()
            meta_root = output_dir / input_dir.name
            meta_root.mkdir(parents=True)
            db_path = meta_root / "tracks.db"
            report_path = meta_root / "report.md"
            tracks_json_path = meta_root / "tracks.json"
            create_legacy_database(db_path)
            config = {
                "input_directory": str(input_dir),
                "output_directory": str(output_dir),
                "model_file_path": "",
                "model_key": "",
                "write_json": write_json,
                "tag_mode": "ask",
                "tagger": {},
            }

            with patch(
                "pipeline.run_genre_tagging",
                side_effect=AssertionError("tagger must not run"),
            ):
                exit_code = run_pipeline(
                    config,
                    stage,
                    PROJECT_ROOT,
                    non_interactive=True,
                )

            self.assertEqual(exit_code, 1)
            self.assertTrue(report_path.is_file())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("- status: failed", report_text)
            self.assertIn("- success: false", report_text)
            self.assertIn("## Error", report_text)
            self.assertIn("incompatible with schema version 2", report_text)
            self.assertIn("move or delete", report_text)
            self.assertFalse(tracks_json_path.exists())

    def test_legacy_database_failure_finalization(self) -> None:
        for stage in ("excel", "tag"):
            for write_json in (False, True):
                with self.subTest(stage=stage, write_json=write_json):
                    self.assert_pipeline_writes_original_schema_error(
                        stage,
                        write_json,
                    )

    def test_unrelated_failure_refreshes_json_and_database_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "music"
            output_dir = root / "output"
            input_dir.mkdir()
            meta_root = output_dir / input_dir.name
            report_path = meta_root / "report.md"
            tracks_json_path = meta_root / "tracks.json"
            config = {
                "input_directory": str(input_dir),
                "output_directory": str(output_dir),
                "model_file_path": "",
                "model_key": "",
                "write_json": True,
                "tag_mode": "no",
                "tagger": {},
            }

            def fail_after_partial_analysis(
                stage_config: dict[str, object],
                script_dir: Path,
                selected_input_dir: Optional[Path],
                db_path: Path,
            ) -> NoReturn:
                init_db(db_path)
                upsert_track(db_path, build_partial_track_payload())
                raise RuntimeError("unrelated analysis failure")

            with patch(
                "pipeline.run_analysis_stage",
                side_effect=fail_after_partial_analysis,
            ):
                exit_code = run_pipeline(
                    config,
                    "analyze",
                    PROJECT_ROOT,
                    non_interactive=True,
                )

            self.assertEqual(exit_code, 1)
            self.assertTrue(tracks_json_path.is_file())
            tracks_payload = json.loads(
                tracks_json_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                [track["hash"] for track in tracks_payload["tracks"]],
                ["partial-hash"],
            )
            self.assertTrue(report_path.is_file())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("- status: failed", report_text)
            self.assertIn("- success: false", report_text)
            self.assertIn("unrelated analysis failure", report_text)
            self.assertIn("## Library Summary", report_text)
            self.assertIn("- total_tracks: 1", report_text)
