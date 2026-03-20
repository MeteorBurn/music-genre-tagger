#!/usr/bin/env python3

import logging
import platform
from pathlib import Path
from typing import Any
from typing import Dict

import pandas as pd
from openpyxl import load_workbook
from mutagen import File
from mutagen.aiff import AIFF
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.wave import WAVE


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


def get_existing_genre(audio_file: Any):
    try:
        if isinstance(audio_file, WAVE):
            if hasattr(audio_file, "tags") and audio_file.tags:
                if "TCON" in audio_file.tags:
                    return str(audio_file.tags["TCON"])
                ignr = audio_file.tags.get("IGNR")
                return ignr[0] if ignr else None
        elif isinstance(audio_file, MP3):
            if (
                hasattr(audio_file, "tags")
                and audio_file.tags
                and "TCON" in audio_file.tags
            ):
                return str(audio_file.tags["TCON"])
        elif isinstance(audio_file, AIFF):
            if hasattr(audio_file, "tags") and audio_file.tags:
                if "TCON" in audio_file.tags:
                    return str(audio_file.tags["TCON"])
                return (
                    audio_file.tags.get("GENRE", [None])[0]
                    if "GENRE" in audio_file.tags
                    else None
                )
        elif isinstance(audio_file, FLAC):
            return (
                audio_file.get("GENRE", [None])[0] if audio_file.get("GENRE") else None
            )
        else:
            if hasattr(audio_file, "tags") and audio_file.tags:
                if "TCON" in audio_file.tags:
                    return str(audio_file.tags["TCON"])
                return (
                    audio_file.get("GENRE", [None])[0]
                    if audio_file.get("GENRE")
                    else None
                )
    except Exception:
        return None
    return None


def set_genre_tag(audio_file: Any, genre_value: str) -> bool:
    try:
        if isinstance(audio_file, WAVE):
            if not hasattr(audio_file, "tags") or not audio_file.tags:
                audio_file.add_tags()
            if hasattr(audio_file.tags, "add"):
                from mutagen.id3 import TCON

                if "TCON" in audio_file.tags:
                    del audio_file.tags["TCON"]
                audio_file.tags.add(TCON(encoding=3, text=[genre_value]))
            else:
                audio_file.tags["IGNR"] = genre_value
        elif isinstance(audio_file, MP3):
            if not hasattr(audio_file, "tags") or not audio_file.tags:
                audio_file.add_tags()
            from mutagen.id3 import TCON

            if "TCON" in audio_file.tags:
                del audio_file.tags["TCON"]
            audio_file.tags.add(TCON(encoding=3, text=[genre_value]))
        elif isinstance(audio_file, FLAC):
            audio_file["GENRE"] = genre_value
        else:
            if not hasattr(audio_file, "tags") or not audio_file.tags:
                audio_file.add_tags()
            if hasattr(audio_file.tags, "add"):
                from mutagen.id3 import TCON

                if "TCON" in audio_file.tags:
                    del audio_file.tags["TCON"]
                audio_file.tags.add(TCON(encoding=3, text=[genre_value]))
            else:
                audio_file["GENRE"] = genre_value
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


def _map_status(result: str) -> str:
    if result == "success":
        return "success"
    if result == "skipped_existing":
        return "skipped"
    return "error"


def _process_audio_file(file_path: Any, genre_data: Any, config: Dict[str, Any]) -> str:
    normalized_path = convert_path_for_current_env(str(file_path))
    path_obj = Path(normalized_path)
    if not path_obj.exists():
        return "file_not_found"

    try:
        audio_file = File(str(path_obj))
        if audio_file is None:
            return "unsupported_format"
    except Exception as exc:
        logging.error("Error loading %s: %s", normalized_path, exc)
        return "load_error"

    existing_genre = get_existing_genre(audio_file)
    if existing_genre and not config["overwrite_existing"]:
        return "skipped_existing"

    genre_string = prepare_genre_string(
        genre_data,
        config["max_genres"],
        config["genre_separator"],
    )
    if not genre_string:
        return "empty_genre"

    if set_genre_tag(audio_file, genre_string):
        try:
            audio_file.save()
            return "success"
        except Exception as exc:
            logging.error("Error saving %s: %s", normalized_path, exc)
            return "save_error"
    return "tag_error"


def auto_adjust_excel_columns(excel_path: Path) -> None:
    workbook = load_workbook(excel_path)
    worksheet = workbook.active
    for column_cells in worksheet.columns:
        max_len = 0
        column_letter = column_cells[0].column_letter
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            if len(value) > max_len:
                max_len = len(value)
        worksheet.column_dimensions[column_letter].width = min(max_len + 2, 80)
    workbook.save(excel_path)


def run_genre_tagging(excel_path: Path, config: Dict[str, Any]) -> Dict[str, int]:
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

    results = {"success": 0, "skipped": 0, "error": 0, "already_processed": 0}

    for index, row in dataframe.iterrows():
        current_status = row.get(status_field, "")
        if current_status == "success":
            results["already_processed"] += 1
            continue

        process_result = _process_audio_file(
            row[config["file_path_field"]],
            row[config["genre_source_field"]],
            config,
        )
        mapped_status = _map_status(process_result)
        dataframe.at[index, status_field] = mapped_status
        results[mapped_status] += 1

    dataframe.to_excel(excel_path, index=False)
    auto_adjust_excel_columns(excel_path)
    return results
