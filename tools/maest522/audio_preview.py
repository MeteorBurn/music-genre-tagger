"""Project-bound, range-capable audio previews for the annotation UI."""

import subprocess
from pathlib import Path
from typing import Iterator

from starlette.responses import StreamingResponse

from .annotation_db import AnnotationStore


DIRECT_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}
STREAM_CHUNK_SIZE = 64 * 1024


class PreviewNotFoundError(ValueError):
    """Raised when a requested track does not belong to the project."""


class PreviewConversionError(RuntimeError):
    """Raised when ffmpeg cannot create a browser-compatible preview."""


class PreviewRangeError(ValueError):
    """Raised when an HTTP byte range cannot be satisfied."""


def _lookup_track(
    store: AnnotationStore,
    project_id: int,
    track_id: int,
) -> tuple[Path, str]:
    with store.connection() as connection:
        row = connection.execute(
            "SELECT path, exact_sha256 FROM tracks WHERE project_id = ? AND id = ?",
            (project_id, track_id),
        ).fetchone()
    if row is None:
        raise PreviewNotFoundError(
            f"Track {track_id} does not belong to annotation project {project_id}."
        )
    audio_path = Path(str(row["path"]))
    if not audio_path.is_file():
        raise PreviewNotFoundError(f"Audio file is unavailable: {audio_path}")
    return audio_path, str(row["exact_sha256"])


def resolve_preview_path(
    store: AnnotationStore,
    project_id: int,
    track_id: int,
    cache_dir: Path | None = None,
    ffmpeg_path: Path = Path("ffmpeg"),
) -> tuple[Path, str]:
    """Resolve a direct audio file or atomically create an MP3 preview."""
    audio_path, exact_sha256 = _lookup_track(store, project_id, track_id)
    direct_media_type = DIRECT_MEDIA_TYPES.get(audio_path.suffix.lower())
    if direct_media_type is not None:
        return audio_path, direct_media_type

    resolved_cache = (
        Path(cache_dir)
        if cache_dir is not None
        else store.database_path.parent / "preview-cache"
    )
    resolved_cache.mkdir(parents=True, exist_ok=True)
    preview_path = resolved_cache / f"{exact_sha256}.mp3"
    if preview_path.is_file():
        return preview_path, "audio/mpeg"

    temporary_path = resolved_cache / f"{exact_sha256}.tmp.mp3"
    try:
        completed = subprocess.run(
            [
                str(ffmpeg_path),
                "-nostdin",
                "-y",
                "-i",
                str(audio_path),
                "-vn",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-c:a",
                "libmp3lame",
                str(temporary_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        temporary_path.unlink(missing_ok=True)
        raise PreviewConversionError(
            f"Could not start ffmpeg for {audio_path}: {error}"
        ) from error
    if completed.returncode != 0:
        temporary_path.unlink(missing_ok=True)
        detail = completed.stderr.strip() or (
            f"ffmpeg exited with status {completed.returncode}."
        )
        raise PreviewConversionError(
            f"Could not create preview for {audio_path}: {detail}"
        )
    if not temporary_path.is_file():
        raise PreviewConversionError(
            f"ffmpeg reported success but created no preview for {audio_path}."
        )
    temporary_path.replace(preview_path)
    return preview_path, "audio/mpeg"


def _parse_range(range_header: str | None, file_size: int) -> tuple[int, int]:
    if file_size < 1:
        raise PreviewRangeError("Cannot stream an empty preview file.")
    if range_header is None:
        return 0, file_size - 1
    if not range_header.startswith("bytes=") or "," in range_header:
        raise PreviewRangeError(f"Unsupported byte range: {range_header}")
    raw_start, separator, raw_end = range_header[6:].partition("-")
    if not separator:
        raise PreviewRangeError(f"Invalid byte range: {range_header}")
    try:
        if not raw_start:
            suffix_length = int(raw_end)
            if suffix_length <= 0:
                raise ValueError
            start = max(0, file_size - suffix_length)
            end = file_size - 1
        else:
            start = int(raw_start)
            end = file_size - 1 if not raw_end else int(raw_end)
    except ValueError as error:
        raise PreviewRangeError(f"Invalid byte range: {range_header}") from error
    if start < 0 or start >= file_size or end < start:
        raise PreviewRangeError(f"Unsatisfiable byte range: {range_header}")
    return start, min(end, file_size - 1)


def _iterate_file(path: Path, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as input_file:
        input_file.seek(start)
        remaining = length
        while remaining > 0:
            chunk = input_file.read(min(STREAM_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def build_preview_response(
    store: AnnotationStore,
    project_id: int,
    track_id: int,
    range_header: str | None,
    cache_dir: Path | None = None,
    ffmpeg_path: Path = Path("ffmpeg"),
) -> StreamingResponse:
    """Build a streaming response for one authorized project track."""
    preview_path, media_type = resolve_preview_path(
        store,
        project_id,
        track_id,
        cache_dir=cache_dir,
        ffmpeg_path=ffmpeg_path,
    )
    file_size = preview_path.stat().st_size
    start, end = _parse_range(range_header, file_size)
    content_length = end - start + 1
    is_partial = range_header is not None
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Cache-Control": "private, max-age=3600",
    }
    if is_partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    return StreamingResponse(
        _iterate_file(preview_path, start, content_length),
        status_code=206 if is_partial else 200,
        media_type=media_type,
        headers=headers,
    )
