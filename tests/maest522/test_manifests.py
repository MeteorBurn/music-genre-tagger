import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.constants import NEW_LABELS
from tools.maest522.manifests import export_training_manifest


class ManifestExportTests(TestCase):
    def test_exports_latest_confirmed_states_masks_and_no_private_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = AnnotationStore(root / "annotations.db")
            store.initialize()
            project_id = store.create_project("manifest-test")
            now = datetime.now(timezone.utc).isoformat()
            split_states = (
                ("train", "positive"),
                ("train", "negative"),
                ("val", "positive"),
                ("val", "negative"),
                ("test", "positive"),
                ("test", "negative"),
            )
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
                for index, (split, annotation_state) in enumerate(split_states):
                    cursor = connection.execute(
                        "INSERT INTO tracks("
                        "project_id, path, exact_sha256, duration_seconds, group_id, "
                        "split, created_at) VALUES (?, ?, ?, 100, ?, ?, ?)",
                        (
                            project_id,
                            str(root / "private" / f"coverage-{index}.wav"),
                            f"{index + 1:064x}",
                            f"{index + 101:064x}",
                            split,
                            now,
                        ),
                    )
                    track_id = int(cursor.lastrowid)
                    for label in NEW_LABELS:
                        connection.execute(
                            "INSERT INTO confirmed_label_events("
                            "project_id, track_id, label, state, event_kind, "
                            "batch_id, note, created_at) VALUES (?, ?, ?, ?, "
                            "'trusted_import', NULL, 'coverage', ?)",
                            (project_id, track_id, label, annotation_state, now),
                        )

                partial_cursor = connection.execute(
                    "INSERT INTO tracks("
                    "project_id, path, exact_sha256, duration_seconds, group_id, "
                    "split, created_at) VALUES (?, ?, ?, 100, ?, 'train', ?)",
                    (
                        project_id,
                        str(root / "private" / "partial.wav"),
                        "a" * 64,
                        "b" * 64,
                        now,
                    ),
                )
                partial_track_id = int(partial_cursor.lastrowid)
                connection.execute(
                    "INSERT INTO track_sources(track_id, source_id, suggested_label) "
                    "VALUES (?, ?, ?)",
                    (partial_track_id, source_cursor.lastrowid, NEW_LABELS[0]),
                )
                for label, annotation_state in (
                    (NEW_LABELS[0], "negative"),
                    (NEW_LABELS[0], "positive"),
                    (NEW_LABELS[1], "negative"),
                ):
                    connection.execute(
                        "INSERT INTO confirmed_label_events("
                        "project_id, track_id, label, state, event_kind, batch_id, "
                        "note, created_at) VALUES (?, ?, ?, ?, 'correction', "
                        "NULL, 'latest wins', ?)",
                        (
                            project_id,
                            partial_track_id,
                            label,
                            annotation_state,
                            now,
                        ),
                    )

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
            partial = next(row for row in rows if row["track_id"] == "sha256:" + "a" * 64)
            self.assertEqual(report.rows_written, 7)
            self.assertTrue(all("path" not in row for row in rows))
            self.assertEqual(partial["group_id"], "group:" + "b" * 64)
            self.assertEqual(partial["window_offsets_seconds"], [5.0, 35.0, 65.0])
            self.assertEqual(
                partial["labels"],
                {
                    NEW_LABELS[0]: "positive",
                    NEW_LABELS[1]: "negative",
                    NEW_LABELS[2]: "unreviewed",
                },
            )
            self.assertEqual(
                partial["label_mask"],
                {
                    NEW_LABELS[0]: 1,
                    NEW_LABELS[1]: 1,
                    NEW_LABELS[2]: 0,
                },
            )
            self.assertEqual(
                partial["candidate_roles"],
                {
                    NEW_LABELS[0]: "positive_candidate",
                    NEW_LABELS[1]: "hard_negative_candidate",
                },
            )
            self.assertEqual(
                partial["label_provenance"][NEW_LABELS[0]],
                {"event_kind": "correction", "batch_id": None},
            )
            self.assertIsNone(partial["label_provenance"][NEW_LABELS[2]])
            summary = json.loads(
                output_path.with_name("dataset_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                summary["rows_by_split"],
                {"train": 3, "val": 2, "test": 2},
            )
            self.assertEqual(len(summary["split_audit_sha256"]), 64)

            with store.connection() as connection:
                connection.execute(
                    "DELETE FROM confirmed_label_events WHERE track_id = ("
                    "SELECT id FROM tracks WHERE split = 'test' "
                    "AND path LIKE '%coverage-5.wav') AND label = ?",
                    (NEW_LABELS[2],),
                )
            with self.assertRaisesRegex(RuntimeError, "coverage"):
                export_training_manifest(
                    store,
                    project_id,
                    root / "export" / "invalid.jsonl",
                    portable=True,
                )
