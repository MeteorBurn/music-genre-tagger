#!/usr/bin/env python3

import logging
import re
from collections import Counter
from datetime import datetime
from datetime import time as datetime_time
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd

from storage import build_excel_dataframe


BROKEN_BEAT_GENRES = {
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
}

TOP_ITEMS_LIMIT = 10
TOP_GENRES_LIMIT = 3
TIMING_KEYS = (
    "total_runtime_seconds",
    "analyze_seconds",
    "excel_seconds",
    "tag_seconds",
)


def load_cumulative_timings(report_path: Path) -> Dict[str, float]:
    timings = {key: 0.0 for key in TIMING_KEYS}
    if not report_path.exists():
        return timings

    try:
        content = report_path.read_text(encoding="utf-8")
    except Exception as exc:
        logging.warning("Unable to read existing report %s: %s", report_path, exc)
        return timings

    for key in TIMING_KEYS:
        match = re.search(
            rf"^- {re.escape(key)}:\s*([0-9]+(?:\.[0-9]+)?)\s*$", content, re.MULTILINE
        )
        if not match:
            continue
        try:
            timings[key] = float(match.group(1))
        except ValueError:
            timings[key] = 0.0
    return timings


def summarize_excel_dataframe(dataframe: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "total_tracks": 0,
        "total_duration_seconds": 0.0,
        "average_duration_seconds": 0.0,
        "broken_beat_tracks": 0,
        "tracks_with_empty_genres": 0,
        "top_genres": [],
        "extension_counts": [],
        "model_key_counts": [],
        "status_breakdown": [],
    }

    if dataframe.empty:
        return summary

    genre_counter: Counter[str] = Counter()
    extension_counter: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    total_duration_seconds = 0.0
    broken_beat_tracks = 0
    tracks_with_empty_genres = 0

    for _, row in dataframe.iterrows():
        genres = _normalize_genre_list(row.get("genres", ""))
        if genres:
            genre_counter.update(genres)
        else:
            tracks_with_empty_genres += 1

        path_value = str(row.get("path", "")).strip()
        extension = Path(path_value).suffix.lower() if path_value else ""
        extension_counter[extension or "[no_ext]"] += 1

        model_key = str(row.get("model_key", "")).strip()
        if model_key:
            model_counter[model_key] += 1

        status_value = str(row.get("status", "")).strip()
        if status_value:
            status_counter[status_value] += 1

        total_duration_seconds += _coerce_duration_seconds(row.get("duration"))

        broken_value = row.get("is_broken_beat", False)
        if broken_value is None or pd.isna(broken_value):
            continue
        if isinstance(broken_value, str):
            if broken_value.strip().lower() in {"true", "1", "yes"}:
                broken_beat_tracks += 1
        elif bool(broken_value):
            broken_beat_tracks += 1

    total_tracks = int(len(dataframe))
    summary["total_tracks"] = total_tracks
    summary["total_duration_seconds"] = total_duration_seconds
    summary["average_duration_seconds"] = (
        total_duration_seconds / total_tracks if total_tracks else 0.0
    )
    summary["broken_beat_tracks"] = broken_beat_tracks
    summary["tracks_with_empty_genres"] = tracks_with_empty_genres
    summary["top_genres"] = _format_counter(genre_counter, limit=TOP_GENRES_LIMIT)
    summary["extension_counts"] = _format_counter(extension_counter)
    summary["model_key_counts"] = _format_counter(model_counter)
    summary["status_breakdown"] = _format_counter(status_counter)
    return summary


def summarize_excel_library(excel_path: Path) -> Dict[str, object]:
    if not excel_path.exists():
        return summarize_excel_dataframe(pd.DataFrame())
    try:
        dataframe = pd.read_excel(excel_path, engine="openpyxl")
    except Exception as exc:
        logging.error("Error reading existing Excel file %s: %s", excel_path, exc)
        return summarize_excel_dataframe(pd.DataFrame())
    return summarize_excel_dataframe(dataframe)


def summarize_database_library(db_path: Path) -> Dict[str, object]:
    if not db_path.exists():
        return summarize_excel_dataframe(pd.DataFrame())
    dataframe = build_excel_dataframe(db_path)
    return summarize_excel_dataframe(dataframe)


def load_best_available_library_summary(
    db_path: Path,
    excel_path: Path,
) -> Optional[Dict[str, object]]:
    if db_path.exists():
        return summarize_database_library(db_path)
    if excel_path.exists():
        return summarize_excel_library(excel_path)
    return None


def write_markdown_report(
    report_path: Path,
    stage: str,
    input_dir: Optional[Path],
    db_path: Path,
    tracks_json_path: Optional[Path],
    excel_path: Path,
    analysis_stats: Optional[Dict[str, Any]],
    excel_stats: Optional[Dict[str, Any]],
    tag_stats: Optional[Dict[str, Any]],
    library_summary: Optional[Dict[str, Any]],
    report_status: str,
    cumulative_timings: Dict[str, float],
    success: bool,
    error_text: str,
) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# MusicTagger Report",
        "",
        "## Run Overview",
        "",
        f"- timestamp: {timestamp}",
        f"- status: {report_status}",
        f"- stage: {stage}",
        f"- success: {str(success).lower()}",
        f"- input_dir: {input_dir if input_dir else ''}",
        f"- db_path: {db_path}",
        f"- tracks_json_path: {tracks_json_path if tracks_json_path else 'disabled'}",
        f"- excel_path: {excel_path}",
        f"- report_path: {report_path}",
        f"- total_runtime_seconds: {_format_seconds(cumulative_timings.get('total_runtime_seconds'))}",
        f"- analyze_seconds: {_format_seconds(cumulative_timings.get('analyze_seconds'))}",
        f"- excel_seconds: {_format_seconds(cumulative_timings.get('excel_seconds'))}",
        f"- tag_seconds: {_format_seconds(cumulative_timings.get('tag_seconds'))}",
    ]

    if error_text:
        lines.extend(["", "## Error", "", error_text])

    if analysis_stats is not None:
        lines.extend(
            [
                "",
                "## Analyze",
                "",
                f"- audio_files_found: {analysis_stats.get('audio_files', 0)}",
                f"- processed_now: {analysis_stats.get('processed', 0)}",
                f"- skipped_existing: {analysis_stats.get('skipped_existing', 0)}",
                f"- analysis_errors_now: {analysis_stats.get('errors', 0)}",
                f"- files_attempted_now: {analysis_stats.get('files_attempted', 0)}",
                f"- elapsed_seconds_now: {_format_seconds(analysis_stats.get('elapsed_seconds'))}",
                f"- average_seconds_per_file_now: {_format_seconds(analysis_stats.get('average_seconds'))}",
            ]
        )

    if excel_stats is not None:
        lines.extend(
            [
                "",
                "## Excel",
                "",
                f"- tracks_seen: {excel_stats.get('tracks_seen', 0)}",
                f"- rows_written_now: {excel_stats.get('rows_written', 0)}",
                f"- rows_total: {excel_stats.get('rows_total', 0)}",
            ]
        )

    tag_status = "completed" if tag_stats is not None else "not_run"
    tag_success = tag_stats.get("success", 0) if tag_stats is not None else 0
    tag_skipped = tag_stats.get("skipped", 0) if tag_stats is not None else 0
    tag_error = tag_stats.get("error", 0) if tag_stats is not None else 0
    tag_already_processed = (
        tag_stats.get("already_processed", 0) if tag_stats is not None else 0
    )
    lines.extend(
        [
            "",
            "## Tag",
            "",
            f"- status: {tag_status}",
            f"- tag_success_now: {tag_success}",
            f"- tag_skipped_existing_now: {tag_skipped}",
            f"- tag_error_now: {tag_error}",
            f"- already_processed_now: {tag_already_processed}",
        ]
    )

    if library_summary is not None:
        status_breakdown = library_summary.get("status_breakdown", [])
        total_tracks = int(library_summary.get("total_tracks", 0) or 0)
        analyzed_tracks = sum(
            count
            for name, count in status_breakdown
            if name
            in {"analysis_success", "tag_success", "tag_skipped_existing", "tag_error"}
        )
        analysis_error_total = sum(
            count for name, count in status_breakdown if name == "analysis_error"
        )
        broken_beat_tracks = int(library_summary.get("broken_beat_tracks", 0) or 0)
        non_broken_beat_tracks = max(total_tracks - broken_beat_tracks, 0)
        remaining_unanalyzed = 0
        if analysis_stats is not None:
            remaining_unanalyzed = max(
                int(analysis_stats.get("audio_files", 0) or 0) - total_tracks,
                0,
            )

        lines.extend(
            [
                "",
                "## Library Summary",
                "",
                f"- total_tracks: {total_tracks}",
                f"- analyzed_tracks: {analyzed_tracks}",
                f"- analysis_error_total: {analysis_error_total}",
                f"- remaining_unanalyzed: {remaining_unanalyzed}",
                f"- total_duration: {_format_duration_seconds(library_summary.get('total_duration_seconds'))}",
                f"- average_duration: {_format_duration_seconds(library_summary.get('average_duration_seconds'))}",
                f"- tracks_with_empty_genres: {library_summary.get('tracks_with_empty_genres', 0)}",
                f"- top_genres: {_format_counter_items(library_summary.get('top_genres'))}",
                f"- extensions: {_format_counter_items(library_summary.get('extension_counts'))}",
                f"- model_keys: {_format_counter_items(library_summary.get('model_key_counts'))}",
            ]
        )

        lines.extend(
            [
                "",
                "## Broken Beat Summary",
                "",
                f"- broken_beat_tracks: {broken_beat_tracks}",
                f"- non_broken_beat_tracks: {non_broken_beat_tracks}",
                f"- broken_beat_share: {_format_percentage(broken_beat_tracks, total_tracks)}",
            ]
        )

        if status_breakdown:
            lines.extend(
                [
                    "",
                    "## Status Breakdown",
                    "",
                    f"- statuses: {_format_counter_items(status_breakdown)}",
                ]
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_broken_beat(genres: List[str]) -> bool:
    return any(genre in BROKEN_BEAT_GENRES for genre in genres)


def _normalize_genre_list(value: object) -> List[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        text = str(value or "").strip()
        if not text:
            return []
        items = [item.strip() for item in text.split(",") if item.strip()]

    unique_items: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items


def _coerce_duration_seconds(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())
    if isinstance(value, timedelta):
        return float(value.total_seconds())
    if isinstance(value, datetime_time):
        return float(value.hour * 3600 + value.minute * 60 + value.second)
    if isinstance(value, str):
        return float(_duration_text_to_timedelta(value).total_seconds())
    if isinstance(value, (int, float)):
        return float(value) * 86400.0

    try:
        coerced = pd.to_timedelta(value)
        if pd.isna(coerced):
            return 0.0
        return float(coerced.total_seconds())
    except Exception:
        return 0.0


def _duration_text_to_timedelta(value: str) -> timedelta:
    text = str(value or "").strip()
    if not text:
        return timedelta(0)
    try:
        parts = text.split(":")
        if len(parts) != 3:
            return timedelta(0)
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except Exception:
        return timedelta(0)


def _format_counter(
    counter: Counter[str], limit: int = TOP_ITEMS_LIMIT
) -> List[Tuple[str, int]]:
    return [(key, count) for key, count in counter.most_common(limit)]


def _format_counter_items(items: Any) -> str:
    if not items:
        return "n/a"

    formatted_items: List[str] = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        name, count = item
        formatted_items.append(f"{name} ({count})")
    return ", ".join(formatted_items) if formatted_items else "n/a"


def _format_seconds(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _format_percentage(part: int, whole: int) -> str:
    if whole <= 0:
        return "0.00%"
    return f"{(float(part) / float(whole)) * 100.0:.2f}%"


def _format_duration_seconds(value: Any) -> str:
    try:
        total_seconds = int(round(float(value)))
    except (TypeError, ValueError):
        total_seconds = 0
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
