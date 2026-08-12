import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.constants import NEW_LABELS
from tools.maest522.manifests import export_training_manifest
from tools.maest522.queues import create_round


class ManifestExportTests(TestCase):
    def test_exports_latest_complete_review_without_private_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = AnnotationStore(root / "annotations.db")
            store.initialize()
            project_id = store.create_project("manifest-test")
            now = datetime.now(timezone.utc).isoformat()
            with store.connection() as connection:
                connection.execute(
                    "UPDATE projects SET split_frozen_at = ? WHERE id = ?",
                    (now, project_id),
                )
                source_cursor = connection.execute(
                    "INSERT INTO sources("
                    "project_id, kind, source_path, candidate_role, imported_at"
                    ") VALUES (?, 'folder', 'seed', 'positive_candidate', ?)",
                    (project_id, now),
                )
                track_cursor = connection.execute(
                    "INSERT INTO tracks("
                    "project_id, path, exact_sha256, duration_seconds, group_id, "
                    "split, created_at"
                    ") VALUES (?, ?, ?, 100, ?, 'train', ?)",
                    (
                        project_id,
                        str(root / "private" / "track.wav"),
                        "a" * 64,
                        "b" * 64,
                        now,
                    ),
                )
                connection.execute(
                    "INSERT INTO track_sources(track_id, source_id, suggested_label) "
                    "VALUES (?, ?, ?)",
                    (track_cursor.lastrowid, source_cursor.lastrowid, NEW_LABELS[0]),
                )

            round_summary = create_round(store, project_id, round_number=1)
            queue_item_id = round_summary.queue_item_ids[0]
            store.append_annotation(queue_item_id, NEW_LABELS[0], "negative")
            store.append_annotation(queue_item_id, NEW_LABELS[0], "positive")
            store.append_annotation(queue_item_id, NEW_LABELS[1], "negative")
            store.append_annotation(queue_item_id, NEW_LABELS[2], "uncertain")
            output_path = root / "export" / "training.jsonl"

            report = export_training_manifest(
                store,
                project_id,
                output_path,
                portable=True,
            )

            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(report.rows_written, 1)
            self.assertEqual(len(rows), 1)
            self.assertNotIn("path", rows[0])
            self.assertEqual(rows[0]["track_id"], "sha256:" + "a" * 64)
            self.assertEqual(rows[0]["group_id"], "group:" + "b" * 64)
            self.assertEqual(rows[0]["window_offsets_seconds"], [5.0, 35.0, 65.0])
            self.assertEqual(
                rows[0]["labels"],
                {
                    NEW_LABELS[0]: "positive",
                    NEW_LABELS[1]: "negative",
                    NEW_LABELS[2]: "uncertain",
                },
            )
            self.assertEqual(
                rows[0]["candidate_roles"],
                {NEW_LABELS[0]: "positive_candidate"},
            )
            summary_path = output_path.with_name("dataset_summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["rows_by_split"], {"train": 1, "val": 0, "test": 0})
            self.assertEqual(len(summary["split_audit_sha256"]), 64)
