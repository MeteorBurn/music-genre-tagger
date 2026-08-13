import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.constants import (
    DEFAULT_NEGATIVE_TARGET,
    DEFAULT_POSITIVE_TARGET,
    NEW_LABELS,
    REVIEW_STATES,
)


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
                    "Electronic---Minimal-Deep-Tech",
                    "Electronic---Microhouse",
                    "Electronic---RoMinimal",
                ),
            )
            self.assertEqual(
                REVIEW_STATES,
                {"positive", "negative", "uncertain", "unreviewed"},
            )
            self.assertEqual(store.schema_version(), 2)

            project_id = store.create_project("schema-v2")

            with closing(sqlite3.connect(database_path)) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                goal_rows = connection.execute(
                    "SELECT label, positive_target, negative_target "
                    "FROM label_goals WHERE project_id = ? ORDER BY label",
                    (project_id,),
                ).fetchall()
                queue_item_columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(queue_items)")
                }
                queue_round_columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(queue_rounds)")
                }

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
                    "label_goals",
                    "confirmed_label_batches",
                    "confirmed_label_events",
                    "fingerprint_audit",
                    "schema_meta",
                }.issubset(tables)
            )
            self.assertEqual(journal_mode.lower(), "wal")
            self.assertEqual(
                goal_rows,
                sorted(
                    (
                        label,
                        DEFAULT_POSITIVE_TARGET,
                        DEFAULT_NEGATIVE_TARGET,
                    )
                    for label in NEW_LABELS
                ),
            )
            self.assertIn("label", queue_item_columns)
            self.assertIn("label", queue_round_columns)

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
                    "project_id, track_id, label, round_number, acquisition_kind, "
                    "created_at) VALUES (?, ?, ?, 1, 'fixture', ?)",
                    (project_id, track_cursor.lastrowid, NEW_LABELS[0], now),
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
