import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from storage import export_tracks_json
from storage import init_db
from storage import load_track_records
from storage import upsert_track


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


class StorageV2Tests(unittest.TestCase):
    def test_round_trips_three_window_metadata(self):
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

    def test_rejects_legacy_database(self):
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
