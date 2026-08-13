import json
import sqlite3
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from storage import export_tracks_json
from storage import init_db
from storage import load_track_records
from storage import upsert_track
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


def build_track_payload() -> dict:
    return {
        "timestamp": "2026-07-30T12:00:00",
        "hash": "0123456789abcdef",
        "file": {
            "path": "C:\\music\\artist\\track.flac",
            "name": "track.flac",
            "extension": ".flac",
            "size": {"megabytes": 12.34, "bytes": 12939427},
            "duration": {"time": "00:03:42", "seconds": 222.0},
        },
        "genres": {
            "labels": ["Breakbeat", "Jungle"],
            "confidences": [0.95, 0.85],
            "model": "maest_519l_pytorch",
        },
        "analysis_config": {
            "audio_segment_offsets": [21.0, 75.0, 129.0],
            "audio_segment_duration": 30.0,
            "audio_segment_count": 3,
            "aggregation": "mean",
        },
    }


class StorageSchemaTests(unittest.TestCase):
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

    def test_creates_and_reopens_schema_v2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tracks.db"
            init_db(db_path)

            connection = sqlite3.connect(db_path)
            try:
                column_types = {
                    row[1]: row[2]
                    for row in connection.execute("PRAGMA table_info(tracks)")
                }
                user_version = connection.execute("PRAGMA user_version").fetchone()[0]
            finally:
                connection.close()

            self.assertEqual(user_version, 2)
            self.assertEqual(column_types["audio_segment_offsets"], "TEXT")
            self.assertEqual(column_types["audio_segment_duration"], "REAL")
            self.assertEqual(column_types["audio_segment_count"], "INTEGER")
            self.assertEqual(column_types["aggregation"], "TEXT")
            init_db(db_path)

    def test_rejects_legacy_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tracks.db"
            connection = sqlite3.connect(db_path)
            try:
                connection.execute("CREATE TABLE tracks (hash TEXT PRIMARY KEY)")
                connection.commit()
            finally:
                connection.close()

            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                init_db(db_path)

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


class StorageSerializationTests(unittest.TestCase):
    def test_round_trips_three_window_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tracks.db"
            init_db(db_path)
            upsert_track(db_path, build_track_payload())
            [record] = load_track_records(db_path)
            self.assertEqual(
                json.loads(record["audio_segment_offsets"]), [21.0, 75.0, 129.0]
            )
            self.assertEqual(record["audio_segment_duration"], 30.0)
            self.assertEqual(record["audio_segment_count"], 3)
            self.assertEqual(record["aggregation"], "mean")

            [exported] = export_tracks_json(db_path, Path(directory) / "tracks.json")
            payload = json.loads(exported.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["tracks"][0]["analysis_config"]["audio_segment_offsets"],
                [21.0, 75.0, 129.0],
            )
