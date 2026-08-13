import math
import shutil
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.constants import NEW_LABELS
from tools.maest522.library import import_source


def write_silent_wav(path: Path, duration_seconds: float = 0.1) -> None:
    frame_count = math.ceil(16_000 * duration_seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * frame_count)


class LibraryImportTests(TestCase):
    def test_deduplicates_overlapping_folder_and_playlist_sources(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_folder = root / "audio"
            audio_path = audio_folder / "track.wav"
            copied_audio_path = root / "copied-track.wav"
            missing_path = root / "missing.wav"
            playlist_path = root / "seed.m3u"
            write_silent_wav(audio_path)
            shutil.copyfile(audio_path, copied_audio_path)
            playlist_path.write_text(
                f"{copied_audio_path}\n{missing_path}\n",
                encoding="utf-8",
            )
            store = AnnotationStore(root / "annotations.db")
            store.initialize()
            project_id = store.create_project("community-extension")

            folder_summary = import_source(
                store=store,
                project_id=project_id,
                source_path=audio_folder,
                suggested_label=NEW_LABELS[0],
                candidate_role="positive_candidate",
            )
            playlist_summary = import_source(
                store=store,
                project_id=project_id,
                source_path=playlist_path,
                suggested_label=NEW_LABELS[0],
                candidate_role="hard_negative_candidate",
            )

            self.assertEqual(folder_summary.discovered, 1)
            self.assertEqual(folder_summary.imported_new, 1)
            self.assertEqual(playlist_summary.discovered, 2)
            self.assertEqual(playlist_summary.imported_new, 0)
            self.assertEqual(playlist_summary.linked_existing, 1)
            self.assertEqual(
                [error.path for error in playlist_summary.errors],
                [missing_path.resolve()],
            )

            with store.connection() as connection:
                track_count = connection.execute(
                    "SELECT COUNT(*) FROM tracks"
                ).fetchone()[0]
                source_count = connection.execute(
                    "SELECT COUNT(*) FROM sources"
                ).fetchone()[0]
                track_source_count = connection.execute(
                    "SELECT COUNT(*) FROM track_sources"
                ).fetchone()[0]
                duration = connection.execute(
                    "SELECT duration_seconds FROM tracks"
                ).fetchone()[0]

            self.assertEqual(track_count, 1)
            self.assertEqual(source_count, 2)
            self.assertEqual(track_source_count, 2)
            self.assertAlmostEqual(duration, 0.1, places=2)
