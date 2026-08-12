import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.constants import NEW_LABELS, REVIEW_STATES


class AnnotationStoreSchemaTests(TestCase):
    def test_initializes_versioned_schema_and_three_labels(self) -> None:
        with TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "annotations.db"
            store = AnnotationStore(database_path)

            store.initialize()
            store.initialize()

            self.assertEqual(
                NEW_LABELS,
                (
                    "Electronic---Microhouse",
                    "Electronic---RoMinimal",
                    "Electronic---DeepTech-Minimal",
                ),
            )
            self.assertEqual(
                REVIEW_STATES,
                {"positive", "negative", "uncertain", "unreviewed"},
            )
            self.assertEqual(store.schema_version(), 1)

            with closing(sqlite3.connect(database_path)) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

            self.assertTrue(
                {
                    "projects",
                    "tracks",
                    "sources",
                    "track_sources",
                    "queue_rounds",
                    "queue_items",
                    "queue_credits",
                    "annotation_events",
                    "fingerprint_audit",
                    "schema_meta",
                }.issubset(tables)
            )
            self.assertEqual(journal_mode.lower(), "wal")

    def test_appends_all_three_review_states_in_one_operation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = AnnotationStore(Path(temp_dir) / "annotations.db")
            store.initialize()
            project_id = store.create_project("review-test")
            now = datetime.now(timezone.utc).isoformat()
            with store.connection() as connection:
                track_cursor = connection.execute(
                    "INSERT INTO tracks("
                    "project_id, path, exact_sha256, duration_seconds, created_at"
                    ") VALUES (?, 'track.wav', ?, 60, ?)",
                    (project_id, "a" * 64, now),
                )
                queue_cursor = connection.execute(
                    "INSERT INTO queue_items("
                    "project_id, track_id, round_number, acquisition_kind, created_at"
                    ") VALUES (?, ?, 1, 'fixture', ?)",
                    (project_id, track_cursor.lastrowid, now),
                )

            event_ids = store.append_review(
                int(queue_cursor.lastrowid),
                {
                    NEW_LABELS[0]: "positive",
                    NEW_LABELS[1]: "negative",
                    NEW_LABELS[2]: "uncertain",
                },
                "one transaction",
            )

            self.assertEqual(len(event_ids), 3)
            current = store.current_annotations(int(queue_cursor.lastrowid))
            self.assertEqual(
                {label: current[label]["state"] for label in NEW_LABELS},
                {
                    NEW_LABELS[0]: "positive",
                    NEW_LABELS[1]: "negative",
                    NEW_LABELS[2]: "uncertain",
                },
            )
