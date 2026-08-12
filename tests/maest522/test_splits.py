from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.constants import NEW_LABELS
from tools.maest522.splits import (
    TrackIdentity,
    audit_split_leakage,
    build_duplicate_groups,
    freeze_group_splits,
)


class DuplicateGroupingTests(TestCase):
    def test_groups_exact_fingerprint_release_and_artist_matches(self) -> None:
        tracks = [
            TrackIdentity("original", "hash-a", "fp-a", "artist-a", "release-a"),
            TrackIdentity("copied", "hash-a", "fp-b", "artist-b", "release-b"),
            TrackIdentity("radio-edit", "hash-b", "fp-c", "artist-c", "release-c"),
            TrackIdentity("full-mix", "hash-c", "fp-c", "artist-d", "release-d"),
            TrackIdentity("release-track-1", "hash-d", "fp-d", "artist-e", "release-e"),
            TrackIdentity("release-track-2", "hash-e", "fp-e", "artist-f", "release-e"),
            TrackIdentity("artist-track-1", "hash-f", "fp-f", "same artist", "release-f"),
            TrackIdentity("artist-track-2", "hash-g", "fp-g", "Same  Artist", "release-g"),
        ]

        assignments = build_duplicate_groups(tracks)

        self.assertEqual(assignments["original"], assignments["copied"])
        self.assertEqual(assignments["radio-edit"], assignments["full-mix"])
        self.assertEqual(
            assignments["release-track-1"], assignments["release-track-2"]
        )
        self.assertEqual(assignments["artist-track-1"], assignments["artist-track-2"])


class FrozenSplitTests(TestCase):
    def _build_store(self, root: Path) -> tuple[AnnotationStore, int]:
        store = AnnotationStore(root / "annotations.db")
        store.initialize()
        project_id = store.create_project("split-test")
        now = datetime.now(timezone.utc).isoformat()
        with store.connection() as connection:
            source_ids = []
            for label_index, label in enumerate(NEW_LABELS):
                cursor = connection.execute(
                    "INSERT INTO sources("
                    "project_id, kind, source_path, candidate_role, imported_at"
                    ") VALUES (?, 'folder', ?, 'positive_candidate', ?)",
                    (project_id, f"source-{label_index}", now),
                )
                source_ids.append(int(cursor.lastrowid))

            for track_index in range(20):
                cursor = connection.execute(
                    "INSERT INTO tracks("
                    "project_id, path, exact_sha256, duration_seconds, artist, "
                    "release_id, created_at"
                    ") VALUES (?, ?, ?, 180, ?, ?, ?)",
                    (
                        project_id,
                        f"track-{track_index}.wav",
                        f"hash-{track_index:02d}",
                        f"artist-{track_index:02d}",
                        f"release-{track_index:02d}",
                        now,
                    ),
                )
                track_id = int(cursor.lastrowid)
                label_index = track_index % len(NEW_LABELS)
                connection.execute(
                    "INSERT INTO track_sources(track_id, source_id, suggested_label) "
                    "VALUES (?, ?, ?)",
                    (track_id, source_ids[label_index], NEW_LABELS[label_index]),
                )
                connection.execute(
                    "INSERT INTO fingerprint_audit(track_id, status, detail, created_at) "
                    "VALUES (?, 'unavailable', 'fixture', ?)",
                    (track_id, now),
                )
        return store, project_id

    def test_freezes_deterministic_group_splits_without_leakage(self) -> None:
        with TemporaryDirectory() as first_temp, TemporaryDirectory() as second_temp:
            first_store, first_project = self._build_store(Path(first_temp))
            second_store, second_project = self._build_store(Path(second_temp))

            first = freeze_group_splits(first_store, first_project, seed=522)
            second = freeze_group_splits(second_store, second_project, seed=522)

            self.assertEqual(first.assignments, second.assignments)
            self.assertEqual(sum(first.split_counts.values()), 20)
            self.assertGreater(first.split_counts["train"], 0)
            self.assertGreater(first.split_counts["val"], 0)
            self.assertGreater(first.split_counts["test"], 0)
            self.assertTrue(first_store.is_split_frozen(first_project))
            audit = audit_split_leakage(first_store, first_project)
            self.assertTrue(audit.clean, audit.issues)
            self.assertEqual(len(audit.digest_sha256), 64)

    def test_audit_detects_a_group_forced_across_splits(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store, project_id = self._build_store(Path(temp_dir))
            freeze_group_splits(store, project_id, seed=522)
            with store.connection() as connection:
                rows = connection.execute(
                    "SELECT id, group_id, split FROM tracks ORDER BY id LIMIT 2"
                ).fetchall()
                forced_split = "test" if rows[0]["split"] != "test" else "val"
                connection.execute(
                    "UPDATE tracks SET group_id = ?, split = ? WHERE id = ?",
                    (rows[0]["group_id"], forced_split, rows[1]["id"]),
                )

            audit = audit_split_leakage(store, project_id)

            self.assertFalse(audit.clean)
            self.assertTrue(
                any("group=" in issue and "crosses splits" in issue for issue in audit.issues)
            )
