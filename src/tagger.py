#!/usr/bin/env python3

import logging
import platform
from pathlib import Path
from typing import Any
from typing import Dict

import pandas as pd
from mutagen import File
from mutagen.aiff import AIFF
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.wave import WAVE

from extractor import auto_adjust_excel_columns
from report import summarize_excel_dataframe


def _is_wsl_environment() -> bool:
    return "microsoft" in platform.uname().release.lower()


def convert_path_for_current_env(file_path: str) -> str:
    if not isinstance(file_path, str):
        return str(file_path) if file_path else ""

    if _is_wsl_environment():
        if len(file_path) >= 3 and file_path[1] == ":" and file_path[2] in ["\\", "/"]:
            drive = file_path[0].lower()
            rest = file_path[3:].replace("\\", "/")
            return f"/mnt/{drive}/{rest}"
        return file_path

    if platform.system().lower().startswith("win") and file_path.startswith("/mnt/"):
        parts = file_path.split("/")
        if len(parts) > 2:
            drive = parts[2].upper()
            rest = "\\".join(parts[3:])
            return f"{drive}:\\{rest}"

    return file_path


def _extract_text_tag_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _extract_text_tag_value(item)
            if text:
                return text
        return ""
    # Handle mutagen ID3 frame objects (TCON, TPE1, etc.) — they stringify to their text content
    text = str(value).strip()
    return text


def _get_id3_tag_text(audio_file: Any, key: str) -> str:
    if not hasattr(audio_file, "tags") or not audio_file.tags:
        return ""
    value = audio_file.tags.get(key)
    if value is None:
        return ""
    # ID3 frame objects have a .text attribute with a list of strings
    if hasattr(value, "text") and isinstance(value.text, list):
        for item in value.text:
            text = str(item).strip()
            if text:
                return text
    return str(value).strip()


def _is_mp4_like(audio_file: Any, file_extension: str) -> bool:
    return isinstance(audio_file, MP4) or file_extension in {".m4a", ".alac"}


def _read_first_existing_genre(audio_file: Any, keys: list[str]) -> str:
    for key in keys:
        value = None
        if hasattr(audio_file, "tags") and audio_file.tags:
            value = audio_file.tags.get(key)
        if value is None and hasattr(audio_file, "get"):
            try:
                value = audio_file.get(key)
            except Exception:
                value = None
        text = _extract_text_tag_value(value)
        if text:
            return text
    return ""


def _set_id3_genre(audio_file: Any, genre_value: str) -> None:
    from mutagen.id3 import TCON

    # tags may be None OR an empty dict-like object — check by identity not truthiness
    if not hasattr(audio_file, "tags") or audio_file.tags is None:
        try:
            audio_file.add_tags()
        except Exception:
            pass

    if audio_file.tags is None:
        raise RuntimeError("Unable to create or access ID3 tags")

    if "TCON" in audio_file.tags:
        del audio_file.tags["TCON"]
    audio_file.tags.add(TCON(encoding=3, text=[genre_value]))


def get_existing_genre(audio_file: Any, file_extension: str):
    try:
        if _is_mp4_like(audio_file, file_extension):
            genre = _read_first_existing_genre(
                audio_file, ["\xa9gen", "GENRE", "Genre"]
            )
            return genre or None
        elif isinstance(audio_file, FLAC):
            genre = _read_first_existing_genre(audio_file, ["GENRE", "Genre"])
            return genre or None

        # For ID3-based formats use dedicated ID3 frame reader first
        for key in ["TCON", "IGNR"]:
            text = _get_id3_tag_text(audio_file, key)
            if text:
                return text

        genre = _read_first_existing_genre(
            audio_file,
            ["GENRE", "Genre", "\xa9gen"],
        )
        return genre or None
    except Exception:
        return None


def set_genre_tag(audio_file: Any, genre_value: str, file_extension: str) -> bool:
    try:
        if _is_mp4_like(audio_file, file_extension):
            audio_file["\xa9gen"] = [genre_value]
        elif isinstance(audio_file, (MP3, WAVE, AIFF)) or file_extension in {
            ".mp3",
            ".wav",
            ".aif",
            ".aiff",
            ".dsf",
            ".dff",
        }:
            _set_id3_genre(audio_file, genre_value)
        elif isinstance(audio_file, FLAC):
            audio_file["GENRE"] = genre_value
        elif file_extension in {".ape", ".wv"}:
            if not hasattr(audio_file, "tags") or not audio_file.tags:
                audio_file.add_tags()
            audio_file["Genre"] = genre_value
        else:
            if not hasattr(audio_file, "tags") or not audio_file.tags:
                audio_file.add_tags()
            if hasattr(audio_file.tags, "add"):
                _set_id3_genre(audio_file, genre_value)
            else:
                audio_file["Genre"] = genre_value
        return True
    except Exception as exc:
        logging.error("Error setting genre tag: %s", exc)
        return False


def prepare_genre_string(genre_data: Any, max_genres: int, separator: str):
    if pd.isna(genre_data) or not str(genre_data).strip():
        return None
    genres = [genre.strip() for genre in str(genre_data).split(",") if genre.strip()]
    if max_genres and len(genres) > max_genres:
        genres = genres[:max_genres]
    return separator.join(genres)


def _map_status(result: str) -> tuple[str, str]:
    if result == "success":
        return "tag_success", "success"
    if result == "skipped_existing":
        return "tag_skipped_existing", "skipped"
    return "tag_error", "error"


def _process_audio_file(file_path: Any, genre_data: Any, config: Dict[str, Any]) -> str:
    normalized_path = convert_path_for_current_env(str(file_path))
    path_obj = Path(normalized_path)
    file_extension = path_obj.suffix.lower()
    if not path_obj.exists():
        return "file_not_found"

    try:
        audio_file = File(str(path_obj))
        if audio_file is None:
            return "unsupported_format"
    except Exception as exc:
        logging.error("Error loading %s: %s", normalized_path, exc)
        return "load_error"

    existing_genre = get_existing_genre(audio_file, file_extension)
    if existing_genre and not config["overwrite_existing"]:
        return "skipped_existing"

    genre_string = prepare_genre_string(
        genre_data,
        config["max_genres"],
        config["genre_separator"],
    )
    if not genre_string:
        return "empty_genre"

    if set_genre_tag(audio_file, genre_string, file_extension):
        try:
            audio_file.save()
            return "success"
        except Exception as exc:
            logging.error("Error saving %s: %s", normalized_path, exc)
            return "save_error"
    return "tag_error"


def run_genre_tagging(excel_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    if not excel_path.is_file():
        raise RuntimeError(f"Excel file not found: {excel_path}")

    try:
        dataframe = pd.read_excel(excel_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to read Excel file {excel_path}: {exc}") from exc

    required_columns = [config["file_path_field"], config["genre_source_field"]]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise RuntimeError(f"Missing required columns in Excel: {joined}")

    status_field = config["status_field"]
    if status_field not in dataframe.columns:
        dataframe[status_field] = ""

    if config["max_rows"]:
        dataframe = dataframe.head(config["max_rows"]).copy()

    results: Dict[str, Any] = {
        "success": 0,
        "skipped": 0,
        "error": 0,
        "already_processed": 0,
        "status_updates": [],
    }
    total = len(dataframe)

    for index, row in dataframe.iterrows():
        file_path_raw = str(row[config["file_path_field"]])
        track_name = Path(file_path_raw).name

        current_status = row.get(status_field, "")
        if current_status == "tag_success":
            results["already_processed"] += 1
            logging.info("[%d/%d] Already tagged: %s", index + 1, total, track_name)
            continue

        process_result = _process_audio_file(
            file_path_raw,
            row[config["genre_source_field"]],
            config,
        )
        mapped_status, counter_key = _map_status(process_result)
        dataframe.at[index, status_field] = mapped_status
        results[counter_key] += 1
        results["status_updates"].append((file_path_raw, mapped_status))

        if process_result == "success":
            logging.info("[%d/%d] Tagged: %s", index + 1, total, track_name)
        elif process_result == "skipped_existing":
            logging.info(
                "[%d/%d] Skipped (tag exists): %s", index + 1, total, track_name
            )
        elif process_result == "empty_genre":
            logging.warning(
                "[%d/%d] Skipped (no genre data): %s", index + 1, total, track_name
            )
        elif process_result == "file_not_found":
            logging.error("[%d/%d] File not found: %s", index + 1, total, file_path_raw)
        else:
            logging.error(
                "[%d/%d] Failed (%s): %s", index + 1, total, process_result, track_name
            )

    dataframe.to_excel(excel_path, index=False)
    auto_adjust_excel_columns(excel_path)
    library_summary = summarize_excel_dataframe(dataframe)
    results["status_breakdown"] = library_summary.get("status_breakdown", [])
    results["library_summary"] = library_summary
    return results
