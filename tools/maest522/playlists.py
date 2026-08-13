"""M3U and M3U8 playlist parsing for local annotation sources."""

import re
from pathlib import Path
from urllib.parse import unquote, urlparse


WINDOWS_URI_PATH = re.compile(r"^/[A-Za-z]:/")


def _path_from_file_uri(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.scheme.lower() != "file":
        raise ValueError(f"Playlist entry is not a file URI: {value}")
    if parsed.netloc not in ("", "localhost"):
        raise ValueError(f"Unsupported file URI host: {parsed.netloc}")
    decoded_path = unquote(parsed.path)
    if WINDOWS_URI_PATH.match(decoded_path):
        decoded_path = decoded_path[1:]
    return Path(decoded_path).resolve()


def parse_playlist(playlist_path: Path) -> list[Path]:
    """Parse local paths from an M3U/M3U8 file in stable source order."""
    resolved_playlist = Path(playlist_path).resolve()
    if resolved_playlist.suffix.lower() not in {".m3u", ".m3u8"}:
        raise ValueError(f"Unsupported playlist extension: {resolved_playlist.suffix}")

    return parse_playlist_text(
        resolved_playlist.read_text(encoding="utf-8-sig"),
        resolved_playlist.parent,
    )


def parse_playlist_text(playlist_text: str, base_directory: Path) -> list[Path]:
    """Parse uploaded playlist text relative to an explicit local base directory."""
    resolved_base = Path(base_directory).resolve()
    if not resolved_base.is_dir():
        raise ValueError(f"Playlist base directory does not exist: {resolved_base}")

    entries: list[Path] = []
    for raw_line in playlist_text.lstrip("\ufeff").splitlines():
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        parsed = urlparse(value)
        if parsed.scheme.lower() in {"http", "https"}:
            raise ValueError(f"Playlist contains a remote URL: {value}")
        if parsed.scheme.lower() == "file":
            entries.append(_path_from_file_uri(value))
            continue

        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = resolved_base / candidate
        entries.append(candidate.resolve())
    return entries
