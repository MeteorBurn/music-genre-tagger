from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.confirmed_labels import (
    append_correction,
    current_confirmed_states,
    get_label_goals,
    get_label_progress,
    update_label_goal,
)
from tools.maest522.constants import NEW_LABELS


class ConfirmedLabelTests(TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.store = AnnotationStore(Path(self.temp_dir.name) / "annotations.db")
        self.store.initialize()
        self.project_id = self.store.create_project("ledger")
        now = datetime.now(timezone.utc).isoformat()
        with self.store.connection() as connection:
            cursor = connection.execute(
                "INSERT INTO tracks(project_id, path, exact_sha256, "
                "duration_seconds, created_at) VALUES (?, 'track.wav', ?, 60, ?)",
                (self.project_id, "a" * 64, now),
            )
            self.track_id = int(cursor.lastrowid)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _append_event(
        self,
        label: str,
        state: str,
        event_kind: str = "manual_review",
        note: str = "",
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.store.connection() as connection:
            cursor = connection.execute(
                "INSERT INTO confirmed_label_events("
                "project_id, track_id, label, state, event_kind, note, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self.project_id,
                    self.track_id,
                    label,
                    state,
                    event_kind,
                    note,
                    now,
                ),
            )
        return int(cursor.lastrowid)

    def test_tracks_soft_goals_and_latest_event_progress(self) -> None:
        goals = get_label_goals(self.store, self.project_id)

        self.assertEqual([goal.label for goal in goals], list(NEW_LABELS))
        self.assertTrue(
            all(
                goal.positive_target == 1000 and goal.negative_target == 1000
                for goal in goals
            )
        )

        updated = update_label_goal(
            self.store,
            self.project_id,
            NEW_LABELS[0],
            positive_target=1,
            negative_target=2,
        )
        self.assertEqual((updated.positive_target, updated.negative_target), (1, 2))
        self._append_event(NEW_LABELS[0], "positive")
        progress = get_label_progress(self.store, self.project_id)[0]
        self.assertEqual(progress.positive_count, 1)
        self.assertEqual(progress.negative_count, 0)
        self.assertEqual(progress.uncertain_count, 0)
        self.assertFalse(progress.complete)

        self._append_event(NEW_LABELS[0], "uncertain")
        progress = get_label_progress(self.store, self.project_id)[0]
        self.assertEqual(progress.positive_count, 0)
        self.assertEqual(progress.uncertain_count, 1)

        update_label_goal(
            self.store,
            self.project_id,
            NEW_LABELS[0],
            positive_target=1,
            negative_target=1,
        )
        self._append_event(NEW_LABELS[0], "positive")
        second_track_time = datetime.now(timezone.utc).isoformat()
        with self.store.connection() as connection:
            second = connection.execute(
                "INSERT INTO tracks(project_id, path, exact_sha256, "
                "duration_seconds, created_at) VALUES (?, 'second.wav', ?, 60, ?)",
                (self.project_id, "b" * 64, second_track_time),
            )
            connection.execute(
                "INSERT INTO confirmed_label_events("
                "project_id, track_id, label, state, event_kind, note, created_at"
                ") VALUES (?, ?, ?, 'negative', 'manual_review', '', ?)",
                (self.project_id, second.lastrowid, NEW_LABELS[0], second_track_time),
            )
        progress = get_label_progress(self.store, self.project_id)[0]
        self.assertEqual((progress.positive_count, progress.negative_count), (1, 1))
        self.assertTrue(progress.complete)

    def test_goal_updates_reject_invalid_or_frozen_projects(self) -> None:
        for target in (0, -1):
            with self.assertRaisesRegex(ValueError, "positive"):
                update_label_goal(
                    self.store,
                    self.project_id,
                    NEW_LABELS[0],
                    positive_target=target,
                    negative_target=1000,
                )
        with self.assertRaisesRegex(ValueError, "Unknown.*label"):
            update_label_goal(
                self.store,
                self.project_id,
                "Electronic---Unknown",
                positive_target=1000,
                negative_target=1000,
            )
        with self.store.connection() as connection:
            connection.execute(
                "UPDATE projects SET split_frozen_at = 'now' WHERE id = ?",
                (self.project_id,),
            )
        with self.assertRaisesRegex(RuntimeError, "split"):
            update_label_goal(
                self.store,
                self.project_id,
                NEW_LABELS[0],
                positive_target=2000,
                negative_target=2000,
            )

    def test_correction_is_append_only_and_requires_a_reason(self) -> None:
        original_id = self._append_event(NEW_LABELS[0], "positive")

        with self.assertRaisesRegex(ValueError, "reason"):
            append_correction(
                self.store,
                self.project_id,
                self.track_id,
                NEW_LABELS[0],
                "negative",
                "  ",
            )

        result = append_correction(
            self.store,
            self.project_id,
            self.track_id,
            NEW_LABELS[0],
            "negative",
            "seed playlist mistake",
        )

        self.assertGreater(result.event_id, original_id)
        self.assertEqual(result.previous_state, "positive")
        self.assertEqual(result.state, "negative")
        with self.store.connection() as connection:
            events = connection.execute(
                "SELECT id, state, event_kind, note FROM confirmed_label_events "
                "WHERE project_id = ? AND track_id = ? AND label = ? ORDER BY id",
                (self.project_id, self.track_id, NEW_LABELS[0]),
            ).fetchall()
            current = current_confirmed_states(connection, self.project_id)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["state"], "positive")
        self.assertEqual(events[1]["event_kind"], "correction")
        self.assertEqual(events[1]["note"], "seed playlist mistake")
        self.assertEqual(current[(self.track_id, NEW_LABELS[0])], "negative")

    def test_correction_rejects_missing_state_and_cross_project_track(self) -> None:
        with self.assertRaisesRegex(ValueError, "current confirmed"):
            append_correction(
                self.store,
                self.project_id,
                self.track_id,
                NEW_LABELS[0],
                "negative",
                "no prior label",
            )
        other_project = self.store.create_project("other")
        with self.assertRaisesRegex(ValueError, "track"):
            append_correction(
                self.store,
                other_project,
                self.track_id,
                NEW_LABELS[0],
                "negative",
                "wrong project",
            )
