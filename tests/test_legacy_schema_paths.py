import sqlite3
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline import run_excel_stage
from pipeline import run_pipeline
from pipeline import run_tag_stage
from storage import export_tracks_json
from storage import validate_db_schema


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


def read_journal_mode(db_path: Path) -> str:
    connection = sqlite3.connect(db_path)
    try:
        return str(connection.execute("PRAGMA journal_mode").fetchone()[0])
    finally:
        connection.close()


class LegacySchemaConsumerTests(unittest.TestCase):
    def assert_legacy_schema_rejected(
        self, operation: Callable[[], object]
    ) -> None:
        try:
            operation()
        except Exception as exc:
            self.assertIsInstance(exc, RuntimeError)
            self.assertRegex(str(exc), "incompatible with schema version 2")
            self.assertIn("move or delete", str(exc))
        else:
            self.fail("Legacy database was not rejected")

    def test_standalone_excel_stage_rejects_legacy_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "tracks.db"
            create_legacy_database(db_path)

            self.assert_legacy_schema_rejected(
                lambda: run_excel_stage({}, db_path, root / "genres.xlsx")
            )

    def test_standalone_tag_stage_rejects_legacy_database_before_tagging(
        self,
    ) -> None:
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

    def test_json_export_rejects_legacy_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "tracks.db"
            create_legacy_database(db_path)

            self.assert_legacy_schema_rejected(
                lambda: export_tracks_json(db_path, root / "tracks.json")
            )

    def test_validation_does_not_change_legacy_database_journal_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tracks.db"
            create_legacy_database(db_path)
            original_journal_mode = read_journal_mode(db_path)

            self.assert_legacy_schema_rejected(
                lambda: validate_db_schema(db_path)
            )

            self.assertEqual(read_journal_mode(db_path), original_journal_mode)

    def test_read_only_legacy_database_has_actionable_schema_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tracks.db"
            create_legacy_database(db_path)
            db_path.chmod(stat.S_IREAD)
            try:
                self.assert_legacy_schema_rejected(
                    lambda: validate_db_schema(db_path)
                )
            finally:
                db_path.chmod(stat.S_IREAD | stat.S_IWRITE)


class LegacySchemaPipelineTests(unittest.TestCase):
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
                try:
                    exit_code = run_pipeline(
                        config,
                        stage,
                        PROJECT_ROOT,
                        non_interactive=True,
                    )
                except Exception as exc:
                    self.fail(
                        f"run_pipeline raised {type(exc).__name__}: {exc}"
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

    def test_excel_legacy_failure_finalizes_with_json_disabled(self) -> None:
        self.assert_pipeline_writes_original_schema_error(
            stage="excel",
            write_json=False,
        )

    def test_excel_legacy_failure_finalizes_with_json_enabled(self) -> None:
        self.assert_pipeline_writes_original_schema_error(
            stage="excel",
            write_json=True,
        )

    def test_tag_legacy_failure_finalizes_with_json_disabled(self) -> None:
        self.assert_pipeline_writes_original_schema_error(
            stage="tag",
            write_json=False,
        )

    def test_tag_legacy_failure_finalizes_with_json_enabled(self) -> None:
        self.assert_pipeline_writes_original_schema_error(
            stage="tag",
            write_json=True,
        )
