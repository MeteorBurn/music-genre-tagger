#!/usr/bin/env python3

import json
import logging
import sqlite3
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple

import pandas as pd


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

TRACK_COLUMNS = [
    "timestamp",
    "hash",
    "path",
    "name",
    "extension",
    "megabytes",
    "bytes",
    "time",
    "seconds",
    "labels",
    "confidences",
    "model",
    "audio_segment_offset",
    "audio_segment_duration",
    "error",
    "status",
    "updated_at",
]

EXCEL_COLUMNS = [
    "status",
    "path",
    "name",
    "duration",
    "genres",
    "is_broken_beat",
    "model_key",
    "confidences",
]


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tracks (
                timestamp TEXT NOT NULL,
                hash TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                name TEXT NOT NULL,
                extension TEXT NOT NULL,
                megabytes REAL NOT NULL,
                bytes INTEGER NOT NULL,
                time TEXT NOT NULL,
                seconds REAL NOT NULL,
                labels TEXT NOT NULL,
                confidences TEXT NOT NULL,
                model TEXT NOT NULL,
                audio_segment_offset REAL NOT NULL,
                audio_segment_duration REAL NOT NULL,
                error TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_tracks_path ON tracks(path)")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(status)"
        )


def get_existing_hashes(db_path: Path) -> set[str]:
    if not db_path.is_file():
        return set()
    with _connect(db_path) as connection:
        rows = connection.execute("SELECT hash FROM tracks").fetchall()
    return {str(row[0]) for row in rows}


def upsert_track(db_path: Path, track_data: Dict[str, Any]) -> None:
    record = _flatten_track(track_data)
    placeholders = ", ".join("?" for _ in TRACK_COLUMNS)
    assignments = ", ".join(
        f"{column}=excluded.{column}" for column in TRACK_COLUMNS if column != "hash"
    )
    values = [record[column] for column in TRACK_COLUMNS]

    with _connect(db_path) as connection:
        connection.execute(
            f"""
            INSERT INTO tracks ({", ".join(TRACK_COLUMNS)})
            VALUES ({placeholders})
            ON CONFLICT(hash) DO UPDATE SET
                {assignments}
            """,
            values,
        )


def build_excel_dataframe(db_path: Path) -> pd.DataFrame:
    records = load_track_records(db_path)
    if not records:
        return pd.DataFrame(columns=EXCEL_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for record in records:
        labels = _load_json_list(record["labels"])
        confidences = _load_json_list(record["confidences"])
        rows.append(
            {
                "status": record["status"],
                "path": record["path"],
                "name": record["name"],
                "duration": _duration_text_to_timedelta(record["time"]),
                "genres": _format_list_with_comma(labels),
                "is_broken_beat": _check_broken_beat(labels),
                "model_key": record["model"],
                "confidences": _format_confidences(confidences),
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.reindex(columns=EXCEL_COLUMNS)
    dataframe["duration"] = pd.to_timedelta(
        dataframe["duration"], errors="coerce"
    ).fillna(pd.Timedelta(0))
    dataframe["is_broken_beat"] = dataframe["is_broken_beat"].fillna(False).astype(bool)
    return dataframe


def load_track_records(db_path: Path) -> List[Dict[str, Any]]:
    if not db_path.is_file():
        return []
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM tracks ORDER BY path COLLATE NOCASE"
        ).fetchall()
    return [dict(row) for row in rows]


def update_track_statuses(
    db_path: Path, status_updates: Sequence[Tuple[str, str]]
) -> None:
    if not status_updates:
        return

    updated_at = datetime.now().isoformat()
    with _connect(db_path) as connection:
        connection.executemany(
            "UPDATE tracks SET status = ?, updated_at = ? WHERE path = ?",
            [(status, updated_at, path) for path, status in status_updates],
        )


def export_tracks_json(db_path: Path, output_path: Path) -> None:
    tracks = [_build_track_payload(record) for record in load_track_records(db_path)]
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tracks": tracks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def _flatten_track(track_data: Dict[str, Any]) -> Dict[str, Any]:
    file_data = track_data.get("file", {})
    size_data = file_data.get("size", {})
    duration_data = file_data.get("duration", {})
    genres_data = track_data.get("genres", {})
    analysis_config = track_data.get("analysis_config", {})
    error_text = str(track_data.get("error", "") or "").strip()
    updated_at = datetime.now().isoformat()

    return {
        "timestamp": str(track_data.get("timestamp", "") or updated_at),
        "hash": str(track_data.get("hash", "") or "").strip(),
        "path": str(file_data.get("path", "") or "").strip(),
        "name": str(file_data.get("name", "") or "").strip(),
        "extension": str(file_data.get("extension", "") or "").strip(),
        "megabytes": float(size_data.get("megabytes", 0.0) or 0.0),
        "bytes": int(size_data.get("bytes", 0) or 0),
        "time": str(duration_data.get("time", "") or "00:00:00").strip(),
        "seconds": float(duration_data.get("seconds", 0.0) or 0.0),
        "labels": json.dumps(list(genres_data.get("labels", [])), ensure_ascii=False),
        "confidences": json.dumps(
            list(genres_data.get("confidences", [])), ensure_ascii=False
        ),
        "model": str(genres_data.get("model", "") or "").strip(),
        "audio_segment_offset": float(
            analysis_config.get("audio_segment_offset", 0.0) or 0.0
        ),
        "audio_segment_duration": float(
            analysis_config.get("audio_segment_duration", 0.0) or 0.0
        ),
        "error": error_text,
        "status": "analysis_error" if error_text else "analysis_success",
        "updated_at": updated_at,
    }


def _build_track_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "timestamp": record["timestamp"],
        "hash": record["hash"],
        "file": {
            "path": record["path"],
            "name": record["name"],
            "extension": record["extension"],
            "size": {
                "megabytes": record["megabytes"],
                "bytes": record["bytes"],
            },
            "duration": {
                "time": record["time"],
                "seconds": record["seconds"],
            },
        },
        "genres": {
            "labels": _load_json_list(record["labels"]),
            "confidences": _load_json_list(record["confidences"]),
            "model": record["model"],
        },
        "analysis_config": {
            "audio_segment_offset": record["audio_segment_offset"],
            "audio_segment_duration": record["audio_segment_duration"],
        },
    }
    if str(record["error"]).strip():
        payload["error"] = record["error"]
    return payload


def _load_json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except Exception as exc:
        logging.warning("Unable to parse JSON list '%s': %s", text, exc)
        return []
    return loaded if isinstance(loaded, list) else []


def _format_list_with_comma(items: Iterable[Any]) -> str:
    unique_items: List[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        unique_items.append(text)
        seen.add(text)
    return ", ".join(unique_items)


def _format_confidences(confidences: Iterable[Any]) -> str:
    values: List[str] = []
    for confidence in confidences:
        text = str(confidence).strip()
        if text:
            values.append(text)
    return ", ".join(values)


def _check_broken_beat(genres: Iterable[Any]) -> bool:
    return any(str(genre).strip() in BROKEN_BEAT_GENRES for genre in genres)


def _duration_text_to_timedelta(value: str) -> timedelta:
    text = str(value or "").strip()
    if not text:
        return timedelta(0)
    try:
        hours, minutes, seconds = [int(part) for part in text.split(":", 2)]
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except Exception:
        return timedelta(0)
