from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.constants import NEW_LABELS
from tools.maest522.queues import create_round


class QueueConstructionTests(TestCase):
    def _create_frozen_project(self, root: Path) -> tuple[AnnotationStore, int]:
        store = AnnotationStore(root / "annotations.db")
        store.initialize()
        project_id = store.create_project("queue-test")
        now = datetime.now(timezone.utc).isoformat()
        with store.connection() as connection:
            connection.execute(
                "UPDATE projects SET split_frozen_at = ? WHERE id = ?",
                (now, project_id),
            )
        return store, project_id

    def test_enforces_label_quotas_hard_negatives_and_overlap_deduplication(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store, project_id = self._create_frozen_project(Path(temp_dir))
            now = datetime.now(timezone.utc).isoformat()
            with store.connection() as connection:
                sources: dict[tuple[str, str], int] = {}
                for label in NEW_LABELS:
                    for role in ("positive_candidate", "hard_negative_candidate"):
                        cursor = connection.execute(
                            "INSERT INTO sources("
                            "project_id, kind, source_path, candidate_role, imported_at"
                            ") VALUES (?, 'folder', ?, ?, ?)",
                            (project_id, f"{label}-{role}", role, now),
                        )
                        sources[(label, role)] = int(cursor.lastrowid)

                shared_track_id = self._insert_track(
                    connection, project_id, "shared", now
                )
                for label in NEW_LABELS:
                    connection.execute(
                        "INSERT INTO track_sources(track_id, source_id, suggested_label) "
                        "VALUES (?, ?, ?)",
                        (
                            shared_track_id,
                            sources[(label, "positive_candidate")],
                            label,
                        ),
                    )

                for label_index, label in enumerate(NEW_LABELS):
                    for candidate_index in range(99):
                        role = (
                            "hard_negative_candidate"
                            if candidate_index < 25
                            else "positive_candidate"
                        )
                        track_id = self._insert_track(
                            connection,
                            project_id,
                            f"{label_index}-{candidate_index:03d}",
                            now,
                        )
                        connection.execute(
                            "INSERT INTO track_sources("
                            "track_id, source_id, suggested_label"
                            ") VALUES (?, ?, ?)",
                            (track_id, sources[(label, role)], label),
                        )

            summary = create_round(store, project_id, round_number=1)

            self.assertEqual(summary.source_counts, {label: 100 for label in NEW_LABELS})
            self.assertEqual(
                summary.hard_negative_counts,
                {label: 25 for label in NEW_LABELS},
            )
            self.assertEqual(summary.unique_tracks, 298)
            with store.connection() as connection:
                queue_count = connection.execute(
                    "SELECT COUNT(*) FROM queue_items"
                ).fetchone()[0]
                credit_count = connection.execute(
                    "SELECT COUNT(*) FROM queue_credits"
                ).fetchone()[0]
            self.assertEqual(queue_count, 298)
            self.assertEqual(credit_count, 300)

    def test_rejects_student_scores_for_blind_holdouts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store, project_id = self._create_frozen_project(Path(temp_dir))

            with self.assertRaisesRegex(ValueError, "blind"):
                create_round(
                    store,
                    project_id,
                    round_number=1,
                    split="val",
                    student_scores={1: 0.9},
                )

    def test_rejects_active_learning_before_blind_holdouts_exist(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store, project_id = self._create_frozen_project(Path(temp_dir))

            with self.assertRaisesRegex(RuntimeError, "blind val and test"):
                create_round(
                    store,
                    project_id,
                    round_number=2,
                    split="train",
                    student_scores={},
                )

    def _insert_track(
        self,
        connection,
        project_id: int,
        suffix: str,
        created_at: str,
    ) -> int:
        cursor = connection.execute(
            "INSERT INTO tracks("
            "project_id, path, exact_sha256, duration_seconds, group_id, split, "
            "created_at"
            ") VALUES (?, ?, ?, 180, ?, 'train', ?)",
            (
                project_id,
                f"track-{suffix}.wav",
                f"hash-{suffix}",
                f"group-{suffix}",
                created_at,
            ),
        )
        return int(cursor.lastrowid)
