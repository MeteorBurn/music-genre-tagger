import shutil
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.constants import NEW_LABELS
from tools.maest522.trusted_import import (
    commit_trusted_playlist,
    preflight_trusted_playlist,
)


def write_wav(path: Path, sample_value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = int(sample_value).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(sample * 1600)


def write_playlist(path: Path, entries: list[Path]) -> None:
    path.write_text(
        "#EXTM3U\n" + "\n".join(str(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )


class TrustedImportTests(TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = AnnotationStore(self.root / "annotations.db")
        self.store.initialize()
        self.project_id = self.store.create_project("trusted-import")
        self.first = self.root / "first.wav"
        self.second = self.root / "second.wav"
        write_wav(self.first, 1)
        write_wav(self.second, 2)
        self.playlist = self.root / "positive.m3u"
        write_playlist(self.playlist, [self.first, self.second])

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _counts(self) -> tuple[int, int, int]:
        with self.store.connection() as connection:
            return tuple(
                int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "tracks",
                    "confirmed_label_batches",
                    "confirmed_label_events",
                )
            )

    def test_clean_preflight_is_read_only_and_commit_is_idempotent(self) -> None:
        before = self._counts()

        preflight = preflight_trusted_playlist(
            self.store,
            self.project_id,
            self.playlist,
            NEW_LABELS[0],
            "positive",
        )

        self.assertEqual(self._counts(), before)
        self.assertEqual(preflight.discovered, 2)
        self.assertEqual(preflight.new_count, 2)
        self.assertEqual(preflight.existing_count, 0)
        self.assertEqual(preflight.missing_paths, ())
        self.assertEqual(preflight.duplicate_paths, ())
        self.assertEqual(preflight.conflict_paths, ())

        first_batch = commit_trusted_playlist(
            self.store,
            self.project_id,
            self.playlist,
            NEW_LABELS[0],
            "positive",
            preflight.playlist_sha256,
        )
        self.assertEqual((first_batch.new_count, first_batch.existing_count), (2, 0))
        self.assertEqual(self._counts(), (2, 1, 2))

        repeated = preflight_trusted_playlist(
            self.store,
            self.project_id,
            self.playlist,
            NEW_LABELS[0],
            "positive",
        )
        self.assertEqual((repeated.new_count, repeated.existing_count), (0, 2))
        second_batch = commit_trusted_playlist(
            self.store,
            self.project_id,
            self.playlist,
            NEW_LABELS[0],
            "positive",
            repeated.playlist_sha256,
        )
        self.assertEqual((second_batch.new_count, second_batch.existing_count), (0, 2))
        self.assertEqual(self._counts(), (2, 2, 2))

    def test_conflicting_batch_is_rejected_without_partial_writes(self) -> None:
        positive = preflight_trusted_playlist(
            self.store,
            self.project_id,
            self.playlist,
            NEW_LABELS[0],
            "positive",
        )
        commit_trusted_playlist(
            self.store,
            self.project_id,
            self.playlist,
            NEW_LABELS[0],
            "positive",
            positive.playlist_sha256,
        )
        before = self._counts()

        negative = preflight_trusted_playlist(
            self.store,
            self.project_id,
            self.playlist,
            NEW_LABELS[0],
            "negative",
        )

        self.assertEqual(
            set(negative.conflict_paths),
            {str(self.first.resolve()), str(self.second.resolve())},
        )
        with self.assertRaisesRegex(ValueError, "conflict"):
            commit_trusted_playlist(
                self.store,
                self.project_id,
                self.playlist,
                NEW_LABELS[0],
                "negative",
                negative.playlist_sha256,
            )
        self.assertEqual(self._counts(), before)

    def test_reports_missing_duplicate_and_physical_copy_entries(self) -> None:
        missing = self.root / "missing.wav"
        copied = self.root / "copy.wav"
        shutil.copyfile(self.first, copied)
        write_playlist(
            self.playlist,
            [self.first, self.first, copied, missing],
        )

        preflight = preflight_trusted_playlist(
            self.store,
            self.project_id,
            self.playlist,
            NEW_LABELS[0],
            "positive",
        )

        self.assertEqual(preflight.discovered, 4)
        self.assertEqual(preflight.missing_paths, (str(missing.resolve()),))
        self.assertEqual(
            set(preflight.duplicate_paths),
            {str(self.first.resolve()), str(copied.resolve())},
        )
        with self.assertRaisesRegex(ValueError, "missing|duplicate"):
            commit_trusted_playlist(
                self.store,
                self.project_id,
                self.playlist,
                NEW_LABELS[0],
                "positive",
                preflight.playlist_sha256,
            )
        self.assertEqual(self._counts(), (0, 0, 0))

    def test_changed_digest_and_cross_label_overlap(self) -> None:
        initial = preflight_trusted_playlist(
            self.store,
            self.project_id,
            self.playlist,
            NEW_LABELS[0],
            "positive",
        )
        self.playlist.write_text(
            self.playlist.read_text(encoding="utf-8") + "# changed\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            commit_trusted_playlist(
                self.store,
                self.project_id,
                self.playlist,
                NEW_LABELS[0],
                "positive",
                initial.playlist_sha256,
            )
        self.assertEqual(self._counts(), (0, 0, 0))

        updated = preflight_trusted_playlist(
            self.store,
            self.project_id,
            self.playlist,
            NEW_LABELS[0],
            "positive",
        )
        commit_trusted_playlist(
            self.store,
            self.project_id,
            self.playlist,
            NEW_LABELS[0],
            "positive",
            updated.playlist_sha256,
        )
        overlap = preflight_trusted_playlist(
            self.store,
            self.project_id,
            self.playlist,
            NEW_LABELS[1],
            "negative",
        )
        self.assertEqual(overlap.conflict_paths, ())
        commit_trusted_playlist(
            self.store,
            self.project_id,
            self.playlist,
            NEW_LABELS[1],
            "negative",
            overlap.playlist_sha256,
        )
        self.assertEqual(self._counts(), (2, 2, 4))
