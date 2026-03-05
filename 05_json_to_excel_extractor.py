#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from openpyxl import load_workbook


# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
CONFIG = {
    "json_directory": "json",
    "output_excel_path": "analysis.xlsx",
    "maest_result_key": "maest_519l_pytorch",
    "loglevel": "INFO",
}


# =============================================================================
# CONSTANTS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BROKEN_BEAT_GENRES = [
    "Breakbeat",
    "Breakcore",
    "Breaks",
    "Progressive Breaks",
    "Broken Beat",
    "Drum n Bass",
    "Jungle",
    "Halftime",
    "Juke",
    "UK Garage",
    "Speed Garage",
    "Bassline",
    "Electro",
]


# =============================================================================
# HELPERS
# =============================================================================
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


def find_json_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        logging.error("JSON directory not found: %s", directory)
        return []
    files = [item for item in directory.rglob("*.json") if item.is_file()]
    files.sort()
    logging.info("Found %d JSON files", len(files))
    return files


def format_list_with_comma(items: List[str]) -> str:
    if not items:
        return ""
    unique_items: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    return ", ".join(unique_items)


def format_confidences(confidences: List[float]) -> str:
    if not confidences:
        return ""
    return ", ".join(str(c) for c in confidences)


def check_broken_beat(genres: List[str]) -> bool:
    return any(g in BROKEN_BEAT_GENRES for g in genres)


def process_json_file(
    json_path: Path, maest_result_key: str
) -> Optional[Dict[str, str]]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        analysis_results = data.get("analysis_results", {})
        maest_result = analysis_results.get(maest_result_key, {})
        labels = maest_result.get("labels", [])
        confidences = maest_result.get("confidences", [])

        file_path_data = data.get("file_path", {})
        win_path = file_path_data.get("win", "")
        if not win_path:
            win_path = str(json_path)

        return {
            "file_path": win_path,
            "file_name": data.get("file_name", ""),
            "genres_maest": format_list_with_comma(labels),
            "confidences": format_confidences(confidences),
            "is_broken_beat": check_broken_beat(labels),
            "model_key": maest_result_key,
        }
    except Exception as e:
        logging.error("Error processing %s: %s", json_path, e)
        return None


def load_existing_excel(excel_path: Path) -> pd.DataFrame:
    try:
        if excel_path.exists():
            df = pd.read_excel(excel_path, engine="openpyxl")
            logging.info("Loaded existing Excel file with %d rows", len(df))
            return df
        logging.info("Excel file does not exist, will create a new one")
        return pd.DataFrame()
    except Exception as e:
        logging.error("Error loading existing Excel file: %s", e)
        return pd.DataFrame()


def filter_new_records(
    new_data: List[Dict[str, str]], existing_df: pd.DataFrame
) -> List[Dict[str, str]]:
    if existing_df.empty:
        return new_data

    existing_paths = (
        set(existing_df["file_path"].tolist())
        if "file_path" in existing_df.columns
        else set()
    )
    new_records = []
    duplicates_count = 0
    for record in new_data:
        if record["file_path"] not in existing_paths:
            new_records.append(record)
        else:
            duplicates_count += 1

    logging.info(
        "Found %d new records, %d duplicates skipped",
        len(new_records),
        duplicates_count,
    )
    return new_records


def create_excel_report(config: Dict[str, str]):
    json_dir = resolve_path(config["json_directory"])
    output_excel = resolve_path(config["output_excel_path"])
    maest_key = config["maest_result_key"]

    json_files = find_json_files(json_dir)
    if not json_files:
        logging.error("No JSON files found")
        return

    new_rows = []
    for json_file in json_files:
        row_data = process_json_file(json_file, maest_key)
        if row_data:
            new_rows.append(row_data)

    if not new_rows:
        logging.error("No valid data extracted from JSON files")
        return

    existing_df = load_existing_excel(output_excel)
    filtered_new_rows = filter_new_records(new_rows, existing_df)
    if not filtered_new_rows:
        logging.info("No new records to add")
        return

    new_df = pd.DataFrame(filtered_new_rows)
    if not existing_df.empty:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        logging.info(
            "Appending %d new rows to existing %d rows", len(new_df), len(existing_df)
        )
    else:
        final_df = new_df
        logging.info("Creating new Excel file with %d rows", len(new_df))

    output_excel.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_excel(output_excel, index=False, engine="openpyxl")
    auto_adjust_excel_columns(output_excel)
    logging.info("Excel file saved: %s", output_excel)
    logging.info("Total rows in file: %d", len(final_df))


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


def validate_config(config: Dict[str, str]):
    if not config.get("json_directory"):
        raise ValueError("json_directory not specified in config")
    if not config.get("output_excel_path"):
        raise ValueError("output_excel_path not specified in config")
    if not config.get("maest_result_key"):
        raise ValueError("maest_result_key not specified in config")


if __name__ == "__main__":
    setup_logging(CONFIG.get("loglevel", "INFO"))
    try:
        validate_config(CONFIG)
        create_excel_report(CONFIG)
        logging.info("Script completed successfully")
    except Exception as e:
        logging.critical("Critical error: %s", e)
        raise SystemExit(1)
