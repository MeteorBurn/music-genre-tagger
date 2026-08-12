import sqlite3
from contextlib import closing
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
