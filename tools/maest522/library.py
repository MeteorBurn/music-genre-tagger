"""Audio-library import and content deduplication for MAEST 522."""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import mutagen
import soundfile

from .annotation_db import AnnotationStore
from .constants import CANDIDATE_ROLES, NEW_LABELS
from .playlists import parse_playlist


SUPPORTED_AUDIO_EXTENSIONS = {
    ".flac",
    ".wav",
    ".aif",
    ".aiff",
    ".mp3",
    ".m4a",
    ".dsf",
    ".ape",
    ".wv",
}


@dataclass(frozen=True)
class ImportErrorDetail:
    path: Path
    message: str


@dataclass(frozen=True)
class ImportSummary:
    source_id: int
    discovered: int
    imported_new: int
    linked_existing: int
    errors: tuple[ImportErrorDetail, ...]


@dataclass(frozen=True)
class AudioMetadata:
    duration_seconds: float
    artist: str | None
    release_id: str | None


@dataclass(frozen=True)
class AudioIdentity:
    path: Path
    exact_sha256: str
    duration_seconds: float
    artist: str | None
    release_id: str | None


def discover_audio_files(folder: Path) -> list[Path]:
    """Return supported audio files below a folder in deterministic order."""
    resolved_folder = Path(folder).resolve()
    if not resolved_folder.is_dir():
        raise ValueError(f"Audio source is not a directory: {resolved_folder}")
    return sorted(
        (
            path.resolve()
            for path in resolved_folder.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )


def stream_sha256(audio_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash file content without loading the full track into memory."""
    digest = hashlib.sha256()
    with Path(audio_path).open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _first_tag_value(audio: Any, keys: Iterable[str]) -> str | None:
    if audio is None or not hasattr(audio, "get"):
        return None
    for key in keys:
        values = audio.get(key)
        if not values:
            continue
        value = values[0] if isinstance(values, (list, tuple)) else values
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def read_audio_metadata(audio_path: Path) -> AudioMetadata:
    """Read duration plus optional artist/release grouping metadata."""
    audio = mutagen.File(str(audio_path), easy=True)
    duration = None
    if audio is not None and getattr(audio, "info", None) is not None:
        duration = getattr(audio.info, "length", None)
    if duration is None:
        duration = float(soundfile.info(str(audio_path)).duration)
    duration_seconds = float(duration)
    if duration_seconds < 0:
        raise ValueError(f"Audio duration is negative: {audio_path}")
    artist = _first_tag_value(audio, ("albumartist", "artist"))
    release_id = _first_tag_value(
        audio,
        ("musicbrainz_albumid", "barcode", "album"),
    )
    return AudioMetadata(duration_seconds, artist, release_id)


def inspect_audio_identity(audio_path: Path) -> AudioIdentity:
    """Inspect one supported audio file without writing annotation state."""
    resolved_path = Path(audio_path).resolve()
    if not resolved_path.is_file():
        raise ValueError(f"Audio file does not exist: {resolved_path}")
    if resolved_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio extension: {resolved_path.suffix}")
    metadata = read_audio_metadata(resolved_path)
    return AudioIdentity(
        path=resolved_path,
        exact_sha256=stream_sha256(resolved_path),
        duration_seconds=metadata.duration_seconds,
        artist=metadata.artist,
        release_id=metadata.release_id,
    )


def _resolve_source(source_path: Path) -> tuple[str, list[Path]]:
    resolved_source = Path(source_path).resolve()
    if resolved_source.is_dir():
        return "folder", discover_audio_files(resolved_source)
    if not resolved_source.is_file():
        raise ValueError(f"Annotation source does not exist: {resolved_source}")
    suffix = resolved_source.suffix.lower()
    if suffix not in {".m3u", ".m3u8"}:
        raise ValueError(f"Annotation source is not a folder or playlist: {resolved_source}")
    return suffix[1:], parse_playlist(resolved_source)


def import_source(
    store: AnnotationStore,
    project_id: int,
    source_path: Path,
    suggested_label: str | None,
    candidate_role: str,
) -> ImportSummary:
    """Import one folder or playlist and link duplicates to the new source."""
    if candidate_role not in CANDIDATE_ROLES:
        raise ValueError(f"Unknown candidate role: {candidate_role}")
    if suggested_label is not None and suggested_label not in NEW_LABELS:
        raise ValueError(f"Unknown MAEST 522 extension label: {suggested_label}")
    if store.is_split_frozen(project_id):
        raise RuntimeError("Cannot import sources after project splits are frozen.")

    resolved_source = Path(source_path).resolve()
    source_kind, audio_paths = _resolve_source(resolved_source)
    imported_at = datetime.now(timezone.utc).isoformat()
    imported_new = 0
    linked_existing = 0
    errors: list[ImportErrorDetail] = []

    with store.connection() as connection:
        cursor = connection.execute(
            "INSERT INTO sources("
            "project_id, kind, source_path, candidate_role, imported_at"
            ") VALUES (?, ?, ?, ?, ?)",
            (
                project_id,
                source_kind,
                str(resolved_source),
                candidate_role,
                imported_at,
            ),
        )
        source_id = int(cursor.lastrowid)

        for audio_path in audio_paths:
            if not audio_path.is_file():
                errors.append(ImportErrorDetail(audio_path, "Audio file does not exist."))
                continue
            if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                errors.append(
                    ImportErrorDetail(
                        audio_path,
                        f"Unsupported audio extension: {audio_path.suffix}",
                    )
                )
                continue
            try:
                identity = inspect_audio_identity(audio_path)
                track_cursor = connection.execute(
                    "INSERT OR IGNORE INTO tracks("
                    "project_id, path, exact_sha256, duration_seconds, artist, "
                    "release_id, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        project_id,
                        str(identity.path),
                        identity.exact_sha256,
                        identity.duration_seconds,
                        identity.artist,
                        identity.release_id,
                        imported_at,
                    ),
                )
                is_new = track_cursor.rowcount == 1
                track_row = connection.execute(
                    "SELECT id FROM tracks "
                    "WHERE project_id = ? AND exact_sha256 = ?",
                    (project_id, identity.exact_sha256),
                ).fetchone()
                if track_row is None:
                    raise RuntimeError(f"Imported track was not persisted: {audio_path}")
                connection.execute(
                    "INSERT OR IGNORE INTO track_sources("
                    "track_id, source_id, suggested_label"
                    ") VALUES (?, ?, ?)",
                    (
                        int(track_row["id"]),
                        source_id,
                        suggested_label or "",
                    ),
                )
                if is_new:
                    imported_new += 1
                else:
                    linked_existing += 1
            except Exception as error:
                errors.append(ImportErrorDetail(audio_path, str(error)))

    return ImportSummary(
        source_id=source_id,
        discovered=len(audio_paths),
        imported_new=imported_new,
        linked_existing=linked_existing,
        errors=tuple(errors),
    )
