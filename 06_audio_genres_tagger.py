#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from mutagen import File
from mutagen.aiff import AIFF
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.wave import WAVE


# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
CONFIG = {
    "excel_file": "analysis.xlsx",
    "genre_source_field": "genres_maest",
    "file_path_field": "file_path",
    "status_field": "status",
    "genre_separator": "; ",
    "max_genres": 3,
    "overwrite_existing": False,
    "max_rows": None,
    "loglevel": "INFO",
}


SCRIPT_DIR = Path(__file__).resolve().parent


def setup_logging(level: str):
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def resolve_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return SCRIPT_DIR / p


def is_wsl_environment() -> bool:
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def convert_path_for_current_env(file_path: str) -> str:
    if not isinstance(file_path, str):
        return str(file_path) if file_path else ""

    if is_wsl_environment():
        if len(file_path) >= 3 and file_path[1] == ":" and file_path[2] in ["\\", "/"]:
            drive = file_path[0].lower()
            rest = file_path[3:].replace("\\", "/")
            return f"/mnt/{drive}/{rest}"
        return file_path

    if file_path.startswith("/mnt/"):
        parts = file_path.split("/")
        if len(parts) > 2:
            drive = parts[2].upper()
            rest = "\\".join(parts[3:])
            return f"{drive}:\\{rest}"
    return file_path


def get_existing_genre(audio_file):
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
        pass
    return None


def set_genre_tag(audio_file, genre_value: str) -> bool:
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
    except Exception as e:
        logging.error("Error setting genre: %s", e)
        return False


def prepare_genre_string(genre_data, max_genres, separator):
    if pd.isna(genre_data) or not str(genre_data).strip():
        return None

    genres = [g.strip() for g in str(genre_data).split(",") if g.strip()]
    if max_genres and len(genres) > max_genres:
        genres = genres[:max_genres]
    return separator.join(genres)


def map_status(result: str) -> str:
    if result == "success":
        return "success"
    if result == "skipped_existing":
        return "skipped"
    return "error"


def process_audio_file(file_path, genre_data, config) -> str:
    normalized_path = convert_path_for_current_env(file_path)
    path_obj = Path(normalized_path)
    if not path_obj.exists():
        logging.debug("File not found: %s", normalized_path)
        return "file_not_found"

    try:
        audio_file = File(str(path_obj))
        if audio_file is None:
            logging.warning("Unsupported format: %s", normalized_path)
            return "unsupported_format"
    except Exception as e:
        logging.error("Error loading %s: %s", normalized_path, e)
        return "load_error"

    existing_genre = get_existing_genre(audio_file)
    if existing_genre and not config["overwrite_existing"]:
        logging.debug("Genre exists in %s: %s", path_obj.name, existing_genre)
        return "skipped_existing"

    genre_string = prepare_genre_string(
        genre_data, config["max_genres"], config["genre_separator"]
    )
    if not genre_string:
        logging.warning("Empty genre data for %s", path_obj.name)
        return "empty_genre"

    if set_genre_tag(audio_file, genre_string):
        try:
            audio_file.save()
            action = "Updated" if existing_genre else "Added"
            logging.info("%s genre '%s' in %s", action, genre_string, path_obj.name)
            return "success"
        except Exception as e:
            logging.error("Error saving %s: %s", normalized_path, e)
            return "save_error"

    return "tag_error"


def run_genre_tagging(config):
    logging.info("Starting genre tagging process")
    excel_path = resolve_path(config["excel_file"])

    if not excel_path.is_file():
        logging.critical("Excel file not found: %s", excel_path)
        return

    try:
        df = pd.read_excel(excel_path)
        logging.info("Loaded Excel file with %d rows", len(df))
    except Exception as e:
        logging.critical("Error reading Excel file: %s", e)
        return

    required_columns = [config["file_path_field"], config["genre_source_field"]]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.critical("Missing columns: %s", missing_columns)
        return

    if config["status_field"] not in df.columns:
        df[config["status_field"]] = ""
        logging.info("Created '%s' column", config["status_field"])

    if config["max_rows"]:
        df = df.head(config["max_rows"])
        logging.info("Processing %d rows (limited)", len(df))

    results = {"success": 0, "skipped": 0, "error": 0, "already_processed": 0}

    for index, row in df.iterrows():
        current_status = row.get(config["status_field"], "")
        if current_status == "success":
            results["already_processed"] += 1
            continue

        file_path = row[config["file_path_field"]]
        genre_data = row[config["genre_source_field"]]

        result = process_audio_file(file_path, genre_data, config)
        status = map_status(result)
        df.at[index, config["status_field"]] = status
        results[status] += 1

    try:
        df.to_excel(excel_path, index=False)
        auto_adjust_excel_columns(excel_path)
        logging.info("Updated Excel file: %s", excel_path)
    except Exception as e:
        logging.error("Error saving Excel file: %s", e)

    logging.info("Genre tagging completed")
    logging.info("Success: %d", results["success"])
    logging.info("Skipped: %d", results["skipped"])
    logging.info("Error: %d", results["error"])
    logging.info("Already processed: %d", results["already_processed"])


def auto_adjust_excel_columns(excel_path: Path):
    wb = load_workbook(excel_path)
    ws = wb.active

    for column_cells in ws.columns:
        max_len = 0
        column_letter = column_cells[0].column_letter
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            if len(value) > max_len:
                max_len = len(value)
        ws.column_dimensions[column_letter].width = min(max_len + 2, 80)

    wb.save(excel_path)


if __name__ == "__main__":
    setup_logging(CONFIG.get("loglevel", "INFO"))
    try:
        run_genre_tagging(CONFIG)
    except Exception as e:
        logging.critical("Critical error: %s", e)
        raise SystemExit(1)
