"""Chromaprint fingerprint collection for annotation split grouping."""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .annotation_db import AnnotationStore


@dataclass(frozen=True)
class FingerprintResult:
    status: str
    fingerprint: str | None
    duration_seconds: float | None
    detail: str


@dataclass(frozen=True)
class FingerprintSummary:
    available: int
    unavailable: int
    errors: int


def calculate_fingerprint(
    audio_path: Path,
    fpcalc_path: Path,
) -> FingerprintResult:
    """Run `fpcalc -json` for one local audio file without shell expansion."""
    try:
        completed = subprocess.run(
            [str(fpcalc_path), "-json", str(audio_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError as error:
        return FingerprintResult("unavailable", None, None, str(error))
    except subprocess.TimeoutExpired as error:
        return FingerprintResult("error", None, None, str(error))
    except OSError as error:
        return FingerprintResult("error", None, None, str(error))

    if completed.returncode != 0:
        detail = completed.stderr.strip() or (
            f"fpcalc exited with status {completed.returncode}."
        )
        return FingerprintResult("error", None, None, detail)

    try:
        payload = json.loads(completed.stdout)
        fingerprint = str(payload["fingerprint"]).strip()
        duration_seconds = float(payload["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return FingerprintResult(
            "error",
            None,
            None,
            f"Invalid fpcalc JSON output: {error}",
        )
    if not fingerprint:
        return FingerprintResult("error", None, None, "fpcalc returned an empty fingerprint.")
    return FingerprintResult("available", fingerprint, duration_seconds, "")


def fingerprint_project(
    store: AnnotationStore,
    project_id: int,
    fpcalc_path: Path,
) -> FingerprintSummary:
    """Fingerprint unaudited project tracks and persist explicit outcomes."""
    with store.connection() as connection:
        rows = connection.execute(
            "SELECT tracks.id, tracks.path "
            "FROM tracks "
            "WHERE tracks.project_id = ? "
            "AND NOT EXISTS ("
            "    SELECT 1 FROM fingerprint_audit "
            "    WHERE fingerprint_audit.track_id = tracks.id"
            ") "
            "ORDER BY tracks.id",
            (project_id,),
        ).fetchall()

    counts = {"available": 0, "unavailable": 0, "error": 0}
    for row in rows:
        track_id = int(row["id"])
        result = calculate_fingerprint(Path(row["path"]), fpcalc_path)
        created_at = datetime.now(timezone.utc).isoformat()
        with store.connection() as connection:
            if result.status == "available":
                connection.execute(
                    "UPDATE tracks SET acoustic_fingerprint = ? WHERE id = ?",
                    (result.fingerprint, track_id),
                )
            connection.execute(
                "INSERT INTO fingerprint_audit(track_id, status, detail, created_at) "
                "VALUES (?, ?, ?, ?)",
                (track_id, result.status, result.detail, created_at),
            )
        counts[result.status] += 1

    return FingerprintSummary(
        available=counts["available"],
        unavailable=counts["unavailable"],
        errors=counts["error"],
    )
