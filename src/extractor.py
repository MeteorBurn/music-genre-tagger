#!/usr/bin/env python3

import json
import logging
import platform
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from openpyxl import load_workbook


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


def find_json_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    files = [item for item in directory.rglob("*.json") if item.is_file()]
    files.sort()
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
    return ", ".join(str(confidence) for confidence in confidences)


def check_broken_beat(genres: List[str]) -> bool:
    return any(genre in BROKEN_BEAT_GENRES for genre in genres)


def _is_wsl_environment() -> bool:
    return "microsoft" in platform.uname().release.lower()


def _preferred_path(path_data: Dict[str, str]) -> str:
    win_path = str(path_data.get("win", "")).strip()
    wsl_path = str(path_data.get("wsl", "")).strip()
    linux_path = str(path_data.get("linux", "")).strip()

    if _is_wsl_environment():
        return wsl_path or linux_path or win_path
    if platform.system().lower().startswith("win"):
        return win_path or wsl_path or linux_path
    return linux_path or wsl_path or win_path


def process_json_file(
    json_path: Path,
    maest_result_key: str,
) -> Optional[Dict[str, object]]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        analysis_results = data.get("analysis_results", {})
        maest_result = analysis_results.get(maest_result_key, {})
        labels = maest_result.get("labels", [])
        confidences = maest_result.get("confidences", [])

        file_path_data = data.get("file_path", {})
        file_path_value = _preferred_path(file_path_data)
        if not file_path_value:
            file_path_value = str(json_path)

        return {
            "file_path": file_path_value,
            "file_name": data.get("file_name", ""),
            "genres_maest": format_list_with_comma(labels),
            "confidences": format_confidences(confidences),
            "is_broken_beat": check_broken_beat(labels),
            "model_key": maest_result_key,
        }
    except Exception as exc:
        logging.error("Error processing %s: %s", json_path, exc)
        return None


def _load_existing_excel(excel_path: Path) -> pd.DataFrame:
    if not excel_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(excel_path, engine="openpyxl")
    except Exception as exc:
        logging.error("Error reading existing Excel file %s: %s", excel_path, exc)
        return pd.DataFrame()


def _filter_new_records(
    new_data: List[Dict[str, object]],
    existing_df: pd.DataFrame,
) -> List[Dict[str, object]]:
    if existing_df.empty:
        return new_data

    existing_paths = (
        set(existing_df["file_path"].tolist())
        if "file_path" in existing_df.columns
        else set()
    )
    return [record for record in new_data if record["file_path"] not in existing_paths]


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


def create_excel_report(
    json_dir: Path,
    output_excel: Path,
    maest_result_key: str,
) -> Dict[str, int]:
    json_files = find_json_files(json_dir)
    if not json_files:
        raise RuntimeError(f"No JSON files found in {json_dir}")

    rows: List[Dict[str, object]] = []
    for json_file in json_files:
        row_data = process_json_file(json_file, maest_result_key)
        if row_data:
            rows.append(row_data)

    if not rows:
        raise RuntimeError("No valid records extracted from JSON files")

    existing_df = _load_existing_excel(output_excel)
    new_rows = _filter_new_records(rows, existing_df)

    if not new_rows:
        logging.info("Excel is up to date, no new records")
        return {
            "json_files": len(json_files),
            "rows_added": 0,
            "rows_total": len(existing_df),
        }

    new_df = pd.DataFrame(new_rows)
    final_df = (
        pd.concat([existing_df, new_df], ignore_index=True)
        if not existing_df.empty
        else new_df
    )

    output_excel.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_excel(output_excel, index=False, engine="openpyxl")
    auto_adjust_excel_columns(output_excel)

    return {
        "json_files": len(json_files),
        "rows_added": len(new_rows),
        "rows_total": len(final_df),
    }
