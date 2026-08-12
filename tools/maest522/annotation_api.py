"""FastAPI application for local MAEST 522 annotation."""

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .annotation_db import AnnotationStore
from .audio_preview import (
    PreviewConversionError,
    PreviewNotFoundError,
    PreviewRangeError,
    build_preview_response,
)
from .constants import CANDIDATE_ROLES, NEW_LABELS, REVIEW_STATES, SPLITS
from .fingerprints import fingerprint_project
from .library import import_source
from .manifests import export_training_manifest
from .playlists import parse_playlist_text
from .queues import create_round
from .splits import audit_split_leakage, freeze_group_splits


class ProjectCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class SourceImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_path: str | None = None
    playlist_name: str | None = None
    playlist_text: str | None = None
    base_directory: str | None = None
    suggested_label: str | None = None
    candidate_role: str

    @field_validator("source_path", "playlist_name", "base_directory")
    @classmethod
    def reject_blank_paths(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Source paths and playlist names must not be blank.")
        return normalized

    @model_validator(mode="after")
    def validate_source_choice(self) -> "SourceImportRequest":
        has_server_path = self.source_path is not None
        has_playlist = self.playlist_text is not None
        if has_server_path == has_playlist:
            raise ValueError("Provide exactly one source_path or playlist_text.")
        if has_playlist and (not self.playlist_name or not self.base_directory):
            raise ValueError(
                "Uploaded playlist text requires playlist_name and base_directory."
            )
        if self.candidate_role not in CANDIDATE_ROLES:
            raise ValueError(f"Unknown candidate role: {self.candidate_role}")
        if self.suggested_label is not None and self.suggested_label not in NEW_LABELS:
            raise ValueError(f"Unknown extension label: {self.suggested_label}")
        return self


class SplitFreezeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 522


class RoundCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round_number: int
    split: Literal["train", "val", "test"] = "train"
    student_scores: dict[int, float] | None = None


class AnnotationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    states: dict[str, str]
    note: str = ""

    @field_validator("states")
    @classmethod
    def validate_states(cls, states: dict[str, str]) -> dict[str, str]:
        if set(states) != set(NEW_LABELS):
            raise ValueError("Annotation request must contain exactly all three labels.")
        invalid = {state for state in states.values() if state not in REVIEW_STATES}
        if invalid:
            raise ValueError("Unknown annotation states: " + ", ".join(sorted(invalid)))
        return states


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, ValueError):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=409, detail=str(error))


def _store_uploaded_playlist(
    store: AnnotationStore,
    project_id: int,
    request: SourceImportRequest,
) -> Path:
    playlist_name = Path(request.playlist_name or "seed.m3u").name
    suffix = Path(playlist_name).suffix.lower()
    if suffix not in {".m3u", ".m3u8"}:
        raise ValueError("Uploaded playlist must use .m3u or .m3u8.")
    entries = parse_playlist_text(
        request.playlist_text or "",
        Path(request.base_directory or ""),
    )
    digest = hashlib.sha256(
        (
            (request.playlist_text or "")
            + "\n"
            + str(Path(request.base_directory or "").resolve())
        ).encode("utf-8")
    ).hexdigest()
    upload_dir = store.database_path.parent / "playlist-uploads" / str(project_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_path = upload_dir / f"{digest}{suffix}"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        "#EXTM3U\n" + "\n".join(str(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def _queue_item_payload(
    store: AnnotationStore,
    project_id: int,
    queue_item_id: int,
) -> dict[str, object]:
    with store.connection() as connection:
        row = connection.execute(
            "SELECT queue_items.id AS queue_item_id, queue_items.round_number, "
            "tracks.id AS track_id, tracks.path, tracks.duration_seconds, "
            "tracks.split FROM queue_items "
            "JOIN tracks ON tracks.id = queue_items.track_id "
            "WHERE queue_items.project_id = ? AND queue_items.id = ?",
            (project_id, queue_item_id),
        ).fetchone()
        source_rows = connection.execute(
            "SELECT DISTINCT track_sources.suggested_label FROM track_sources "
            "JOIN queue_items ON queue_items.track_id = track_sources.track_id "
            "WHERE queue_items.id = ? AND track_sources.suggested_label <> ''",
            (queue_item_id,),
        ).fetchall()
        credit_rows = connection.execute(
            "SELECT label, candidate_role FROM queue_credits "
            "WHERE queue_item_id = ? ORDER BY label",
            (queue_item_id,),
        ).fetchall()
    if row is None:
        raise ValueError(f"Unknown queue item ID for project {project_id}: {queue_item_id}")
    current = store.current_annotations(queue_item_id)
    notes = [value for value in current.values() if value["note"]]
    latest_note = max(notes, key=lambda value: value["created_at"])["note"] if notes else ""
    return {
        "queue_item_id": int(row["queue_item_id"]),
        "track_id": int(row["track_id"]),
        "filename": Path(str(row["path"])).name,
        "duration_seconds": float(row["duration_seconds"]),
        "split": str(row["split"]),
        "round_number": int(row["round_number"]),
        "source_labels": sorted(str(source["suggested_label"]) for source in source_rows),
        "candidate_roles": {
            str(credit["label"]): str(credit["candidate_role"])
            for credit in credit_rows
        },
        "states": {
            label: current.get(label, {}).get("state", "unreviewed")
            for label in NEW_LABELS
        },
        "note": latest_note,
        "audio_url": f"/api/projects/{project_id}/audio/{int(row['track_id'])}",
    }


def _next_incomplete_queue_item(
    store: AnnotationStore,
    project_id: int,
    after_id: int | None,
) -> dict[str, object] | None:
    with store.connection() as connection:
        row = connection.execute(
            "WITH latest AS ("
            "  SELECT events.queue_item_id, events.label, events.state "
            "  FROM annotation_events AS events "
            "  JOIN ("
            "    SELECT queue_item_id, label, MAX(id) AS latest_id "
            "    FROM annotation_events GROUP BY queue_item_id, label"
            "  ) AS selected ON selected.latest_id = events.id"
            ") "
            "SELECT queue_items.id FROM queue_items "
            "LEFT JOIN latest ON latest.queue_item_id = queue_items.id "
            "WHERE queue_items.project_id = ? "
            "AND (? IS NULL OR queue_items.id > ?) "
            "GROUP BY queue_items.id "
            "HAVING SUM(CASE WHEN latest.state IN "
            "('positive', 'negative', 'uncertain') THEN 1 ELSE 0 END) < 3 "
            "ORDER BY queue_items.id LIMIT 1",
            (project_id, after_id, after_id),
        ).fetchone()
    return None if row is None else _queue_item_payload(store, project_id, int(row["id"]))


def create_app(
    database_path: Path,
    fpcalc_path: Path = Path("fpcalc"),
    ffmpeg_path: Path = Path("ffmpeg"),
    static_dir: Path | None = None,
) -> FastAPI:
    """Create a localhost-oriented annotation application."""
    store = AnnotationStore(Path(database_path))
    store.initialize()
    resolved_static = static_dir or Path(__file__).with_name("static")
    app = FastAPI(title="MAEST 522 Annotation", docs_url="/api/docs")
    app.state.store = store
    app.mount("/static", StaticFiles(directory=resolved_static), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(resolved_static / "index.html")

    @app.post("/api/projects", status_code=status.HTTP_201_CREATED)
    def create_project(request: ProjectCreateRequest) -> dict[str, int]:
        try:
            return {"project_id": store.create_project(request.name)}
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.post("/api/projects/{project_id}/sources")
    def add_source(project_id: int, request: SourceImportRequest) -> dict[str, object]:
        try:
            source_path = (
                Path(request.source_path)
                if request.source_path is not None
                else _store_uploaded_playlist(store, project_id, request)
            )
            summary = import_source(
                store,
                project_id,
                source_path,
                request.suggested_label,
                request.candidate_role,
            )
            return {
                "source_id": summary.source_id,
                "discovered": summary.discovered,
                "imported_new": summary.imported_new,
                "linked_existing": summary.linked_existing,
                "errors": [
                    {"path": str(error.path), "message": error.message}
                    for error in summary.errors
                ],
            }
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.post("/api/projects/{project_id}/fingerprints")
    def fingerprint(project_id: int) -> dict[str, int]:
        try:
            return asdict(fingerprint_project(store, project_id, fpcalc_path))
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.post("/api/projects/{project_id}/splits/freeze")
    def freeze_splits(
        project_id: int,
        request: SplitFreezeRequest,
    ) -> dict[str, object]:
        try:
            summary = freeze_group_splits(store, project_id, seed=request.seed)
            audit = audit_split_leakage(store, project_id)
            return {
                "split_counts": summary.split_counts,
                "group_counts": summary.group_counts,
                "audit_clean": audit.clean,
                "audit_issues": audit.issues,
                "audit_sha256": audit.digest_sha256,
            }
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.post(
        "/api/projects/{project_id}/rounds",
        status_code=status.HTTP_201_CREATED,
    )
    def add_round(project_id: int, request: RoundCreateRequest) -> dict[str, object]:
        try:
            summary = create_round(
                store,
                project_id,
                request.round_number,
                split=request.split,
                student_scores=request.student_scores,
            )
            return asdict(summary)
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.get("/api/projects/{project_id}/queue/next", response_model=None)
    def next_queue_item(
        project_id: int,
        after_id: int | None = None,
    ) -> Response | dict[str, object]:
        try:
            payload = _next_incomplete_queue_item(store, project_id, after_id)
            return Response(status_code=204) if payload is None else payload
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.get("/api/projects/{project_id}/queue/{queue_item_id}")
    def get_queue_item(project_id: int, queue_item_id: int) -> dict[str, object]:
        try:
            return _queue_item_payload(store, project_id, queue_item_id)
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.post("/api/projects/{project_id}/queue/{queue_item_id}/annotations")
    def annotate(
        project_id: int,
        queue_item_id: int,
        request: AnnotationRequest,
    ) -> dict[str, object]:
        try:
            _queue_item_payload(store, project_id, queue_item_id)
            event_ids = store.append_review(
                queue_item_id,
                request.states,
                request.note,
            )
            return {"event_ids": event_ids, "states": request.states}
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.get("/api/projects/{project_id}/progress")
    def progress(project_id: int) -> dict[str, int]:
        with store.connection() as connection:
            rows = connection.execute(
                "WITH latest AS ("
                "  SELECT events.queue_item_id, events.label, events.state "
                "  FROM annotation_events AS events "
                "  JOIN ("
                "    SELECT queue_item_id, label, MAX(id) AS latest_id "
                "    FROM annotation_events GROUP BY queue_item_id, label"
                "  ) AS selected ON selected.latest_id = events.id"
                ") "
                "SELECT queue_items.id, latest.label, latest.state "
                "FROM queue_items LEFT JOIN latest "
                "ON latest.queue_item_id = queue_items.id "
                "WHERE queue_items.project_id = ? ORDER BY queue_items.id",
                (project_id,),
            ).fetchall()
        reviewed_states: dict[int, dict[str, str]] = {}
        for row in rows:
            queue_item_id = int(row["id"])
            reviewed_states.setdefault(queue_item_id, {})
            if row["label"] is not None:
                reviewed_states[queue_item_id][str(row["label"])] = str(row["state"])
        total = len(reviewed_states)
        completed = sum(
            all(
                states.get(label) in {"positive", "negative", "uncertain"}
                for label in NEW_LABELS
            )
            for states in reviewed_states.values()
        )
        return {"total": total, "completed": completed, "remaining": total - completed}

    @app.get("/api/projects/{project_id}/export")
    def export(project_id: int) -> dict[str, object]:
        output_path = (
            store.database_path.parent
            / "exports"
            / str(project_id)
            / "training.jsonl"
        )
        try:
            report = export_training_manifest(
                store,
                project_id,
                output_path,
                portable=True,
            )
            return {
                "rows_written": report.rows_written,
                "output_path": str(report.output_path),
                "summary_path": str(report.summary_path),
                "split_audit_sha256": report.split_audit_sha256,
            }
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.get("/api/projects/{project_id}/audio/{track_id}")
    def audio(project_id: int, track_id: int, request: Request):
        try:
            return build_preview_response(
                store,
                project_id,
                track_id,
                request.headers.get("range"),
                ffmpeg_path=ffmpeg_path,
            )
        except PreviewNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except PreviewRangeError as error:
            raise HTTPException(status_code=416, detail=str(error)) from error
        except PreviewConversionError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    return app
