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
from .confirmed_labels import (
    append_correction,
    append_manual_review,
    get_label_goals,
    get_label_progress,
    list_confirmed_batches,
    update_label_goal,
)
from .fingerprints import fingerprint_project
from .library import import_source
from .manifests import export_training_manifest
from .playlists import parse_playlist_text
from .queues import create_round
from .splits import audit_split_leakage, freeze_group_splits
from .trusted_import import commit_trusted_playlist, preflight_trusted_playlist


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
    label: str
    round_number: int
    split: Literal["train", "val", "test"] = "train"
    student_scores: dict[int, float] | None = None


    @field_validator("label")
    @classmethod
    def validate_label(cls, label: str) -> str:
        if label not in NEW_LABELS:
            raise ValueError(f"Unknown extension label: {label}")
        return label


class AnnotationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    state: Literal["positive", "negative", "uncertain"]
    note: str = ""

    @field_validator("label")
    @classmethod
    def validate_label(cls, label: str) -> str:
        if label not in NEW_LABELS:
            raise ValueError(f"Unknown extension label: {label}")
        return label


class GoalUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    positive_target: int
    negative_target: int


class TrustedPlaylistRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    playlist_path: str
    label: str
    state: Literal["positive", "negative"]

    @field_validator("playlist_path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        normalized = path.strip()
        if not normalized:
            raise ValueError("playlist_path must not be blank")
        return normalized

    @field_validator("label")
    @classmethod
    def validate_label(cls, label: str) -> str:
        if label not in NEW_LABELS:
            raise ValueError(f"Unknown extension label: {label}")
        return label


class TrustedPlaylistCommitRequest(TrustedPlaylistRequest):
    expected_playlist_sha256: str


class CorrectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    track_id: int
    label: str
    state: Literal["positive", "negative", "uncertain"]
    reason: str

    @field_validator("label")
    @classmethod
    def validate_label(cls, label: str) -> str:
        if label not in NEW_LABELS:
            raise ValueError(f"Unknown extension label: {label}")
        return label


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
            "queue_items.label AS active_label, "
            "tracks.id AS track_id, tracks.path, tracks.duration_seconds, "
            "tracks.split FROM queue_items "
            "JOIN tracks ON tracks.id = queue_items.track_id "
            "WHERE queue_items.project_id = ? AND queue_items.id = ?",
            (project_id, queue_item_id),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Unknown queue item ID for project {project_id}: {queue_item_id}"
            )
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
        current = connection.execute(
            "SELECT state, note FROM confirmed_label_events "
            "WHERE project_id = ? AND track_id = ? AND label = ? "
            "ORDER BY id DESC LIMIT 1",
            (project_id, int(row["track_id"]), str(row["active_label"])),
        ).fetchone()
    return {
        "queue_item_id": int(row["queue_item_id"]),
        "track_id": int(row["track_id"]),
        "filename": Path(str(row["path"])).name,
        "duration_seconds": float(row["duration_seconds"]),
        "split": str(row["split"]),
        "round_number": int(row["round_number"]),
        "active_label": str(row["active_label"]),
        "source_labels": sorted(str(source["suggested_label"]) for source in source_rows),
        "candidate_roles": {
            str(credit["label"]): str(credit["candidate_role"])
            for credit in credit_rows
        },
        "state": "unreviewed" if current is None else str(current["state"]),
        "note": "" if current is None else str(current["note"]),
        "audio_url": f"/api/projects/{project_id}/audio/{int(row['track_id'])}",
    }


def _next_incomplete_queue_item(
    store: AnnotationStore,
    project_id: int,
    after_id: int | None,
) -> dict[str, object] | None:
    with store.connection() as connection:
        row = connection.execute(
            "SELECT queue_items.id FROM queue_items "
            "WHERE queue_items.project_id = ? "
            "AND (? IS NULL OR queue_items.id > ?) "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM confirmed_label_events AS events "
            "  WHERE events.project_id = queue_items.project_id "
            "  AND events.track_id = queue_items.track_id "
            "  AND events.label = queue_items.label"
            ") "
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

    @app.get("/api/projects/{project_id}/goals")
    def goals(project_id: int) -> list[dict[str, object]]:
        try:
            return [asdict(goal) for goal in get_label_goals(store, project_id)]
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.patch("/api/projects/{project_id}/goals/{label}")
    def change_goal(
        project_id: int,
        label: str,
        request: GoalUpdateRequest,
    ) -> dict[str, object]:
        try:
            return asdict(
                update_label_goal(
                    store,
                    project_id,
                    label,
                    request.positive_target,
                    request.negative_target,
                )
            )
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.get("/api/projects/{project_id}/confirmed-progress")
    def confirmed_progress(project_id: int) -> list[dict[str, object]]:
        try:
            return [asdict(item) for item in get_label_progress(store, project_id)]
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.get("/api/projects/{project_id}/confirmed-batches/{label}")
    def confirmed_batches(project_id: int, label: str) -> list[dict[str, object]]:
        try:
            return [
                asdict(item)
                for item in list_confirmed_batches(store, project_id, label)
            ]
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.post("/api/projects/{project_id}/trusted-playlists/preflight")
    def trusted_preflight(
        project_id: int,
        request: TrustedPlaylistRequest,
    ) -> dict[str, object]:
        try:
            result = preflight_trusted_playlist(
                store,
                project_id,
                Path(request.playlist_path),
                request.label,
                request.state,
            )
            return {
                "playlist_path": result.playlist_path,
                "playlist_sha256": result.playlist_sha256,
                "label": result.label,
                "state": result.state,
                "discovered": result.discovered,
                "new_count": result.new_count,
                "existing_count": result.existing_count,
                "missing_paths": result.missing_paths,
                "duplicate_paths": result.duplicate_paths,
                "invalid_paths": result.invalid_paths,
                "conflict_paths": result.conflict_paths,
                "clean": result.clean,
            }
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.post("/api/projects/{project_id}/trusted-playlists/commit")
    def trusted_commit(
        project_id: int,
        request: TrustedPlaylistCommitRequest,
    ) -> dict[str, object]:
        try:
            return asdict(
                commit_trusted_playlist(
                    store,
                    project_id,
                    Path(request.playlist_path),
                    request.label,
                    request.state,
                    request.expected_playlist_sha256,
                )
            )
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.post("/api/projects/{project_id}/confirmed-labels/correct")
    def correct_label(
        project_id: int,
        request: CorrectionRequest,
    ) -> dict[str, object]:
        try:
            return asdict(
                append_correction(
                    store,
                    project_id,
                    request.track_id,
                    request.label,
                    request.state,
                    request.reason,
                )
            )
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
                request.label,
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
            payload = _queue_item_payload(store, project_id, queue_item_id)
            if payload["active_label"] != request.label:
                raise ValueError(
                    f"Queue item label is {payload['active_label']}, "
                    f"not requested label {request.label}"
                )
            event_id = append_manual_review(
                store,
                project_id,
                queue_item_id,
                request.label,
                request.state,
                request.note,
            )
            return {
                "event_id": event_id,
                "label": request.label,
                "state": request.state,
            }
        except (ValueError, RuntimeError) as error:
            raise _http_error(error) from error

    @app.get("/api/projects/{project_id}/progress")
    def progress(project_id: int) -> dict[str, int]:
        with store.connection() as connection:
            rows = connection.execute(
                "SELECT queue_items.id, EXISTS("
                "  SELECT 1 FROM confirmed_label_events AS events "
                "  WHERE events.project_id = queue_items.project_id "
                "  AND events.track_id = queue_items.track_id "
                "  AND events.label = queue_items.label"
                ") AS completed "
                "FROM queue_items WHERE queue_items.project_id = ? "
                "ORDER BY queue_items.id",
                (project_id,),
            ).fetchall()
        total = len(rows)
        completed = sum(bool(row["completed"]) for row in rows)
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
