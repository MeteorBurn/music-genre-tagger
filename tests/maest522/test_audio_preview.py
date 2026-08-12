import asyncio
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.audio_preview import (
    PreviewConversionError,
    PreviewNotFoundError,
    build_preview_response,
    resolve_preview_path,
)


class AudioPreviewTests(TestCase):
    def _insert_track(
        self,
        store: AnnotationStore,
        project_id: int,
        audio_path: Path,
        exact_sha256: str,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with store.connection() as connection:
            cursor = connection.execute(
                "INSERT INTO tracks("
                "project_id, path, exact_sha256, duration_seconds, created_at"
                ") VALUES (?, ?, ?, 60, ?)",
                (project_id, str(audio_path), exact_sha256, now),
            )
        return int(cursor.lastrowid)

    def test_serves_project_bound_byte_ranges(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "track.mp3"
            audio_path.write_bytes(b"0123456789")
            store = AnnotationStore(root / "annotations.db")
            store.initialize()
            project_id = store.create_project("preview-test")
            other_project_id = store.create_project("other-project")
            track_id = self._insert_track(store, project_id, audio_path, "a" * 64)
            response = build_preview_response(
                store,
                project_id,
                track_id,
                "bytes=2-5",
            )

            async def collect_body() -> bytes:
                chunks = [chunk async for chunk in response.body_iterator]
                return b"".join(chunks)

            self.assertEqual(response.status_code, 206)
            self.assertEqual(asyncio.run(collect_body()), b"2345")
            self.assertEqual(response.headers["content-range"], "bytes 2-5/10")
            self.assertEqual(response.headers["content-length"], "4")
            self.assertEqual(response.headers["accept-ranges"], "bytes")
            with self.assertRaises(PreviewNotFoundError):
                build_preview_response(
                    store,
                    other_project_id,
                    track_id,
                    None,
                )

    @patch("tools.maest522.audio_preview.subprocess.run")
    def test_converts_unsupported_audio_into_content_addressed_cache(
        self,
        run_mock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "track.dsf"
            audio_path.write_bytes(b"not-real-dsf")
            store = AnnotationStore(root / "annotations.db")
            store.initialize()
            project_id = store.create_project("conversion-test")
            exact_sha256 = "b" * 64
            track_id = self._insert_track(
                store,
                project_id,
                audio_path,
                exact_sha256,
            )

            def create_output(arguments, **_kwargs):
                Path(arguments[-1]).write_bytes(b"converted")
                return subprocess.CompletedProcess(arguments, 0, "", "")

            run_mock.side_effect = create_output
            preview_path, media_type = resolve_preview_path(
                store,
                project_id,
                track_id,
                ffmpeg_path=Path("C:/Utils/tools/ffmpeg.exe"),
            )

            self.assertEqual(preview_path, root / "preview-cache" / f"{exact_sha256}.mp3")
            self.assertEqual(preview_path.read_bytes(), b"converted")
            self.assertEqual(media_type, "audio/mpeg")
            arguments = run_mock.call_args.args[0]
            self.assertEqual(arguments[0], "C:\\Utils\\tools\\ffmpeg.exe")
            self.assertIn("-nostdin", arguments)
            self.assertIn("libmp3lame", arguments)
            self.assertEqual(arguments[arguments.index("-i") + 1], str(audio_path))

    @patch(
        "tools.maest522.audio_preview.subprocess.run",
        side_effect=FileNotFoundError("ffmpeg missing"),
    )
    def test_reports_missing_ffmpeg_as_preview_conversion_error(
        self,
        _run_mock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "track.ape"
            audio_path.write_bytes(b"not-real-ape")
            store = AnnotationStore(root / "annotations.db")
            store.initialize()
            project_id = store.create_project("missing-ffmpeg-test")
            track_id = self._insert_track(store, project_id, audio_path, "c" * 64)

            with self.assertRaisesRegex(PreviewConversionError, "ffmpeg missing"):
                resolve_preview_path(store, project_id, track_id)
