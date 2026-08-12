"""SQLite persistence for the MAEST 522 annotation workflow."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .constants import SCHEMA_VERSION


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    split_frozen_at TEXT
);

CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    exact_sha256 TEXT NOT NULL,
    acoustic_fingerprint TEXT,
    duration_seconds REAL NOT NULL CHECK(duration_seconds >= 0),
    artist TEXT,
    release_id TEXT,
    group_id TEXT,
    split TEXT CHECK(split IN ('train', 'val', 'test')),
    created_at TEXT NOT NULL,
    UNIQUE(project_id, exact_sha256)
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK(kind IN ('folder', 'm3u', 'm3u8')),
    source_path TEXT NOT NULL,
    candidate_role TEXT NOT NULL CHECK(
        candidate_role IN (
            'positive_candidate',
            'hard_negative_candidate',
            'unlabeled_pool'
        )
    ),
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS track_sources (
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    suggested_label TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(track_id, source_id, suggested_label)
);

CREATE TABLE IF NOT EXISTS queue_items (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    round_number INTEGER NOT NULL CHECK(round_number > 0),
    acquisition_kind TEXT NOT NULL,
    acquisition_score REAL,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, track_id)
);

CREATE TABLE IF NOT EXISTS annotation_events (
    id INTEGER PRIMARY KEY,
    queue_item_id INTEGER NOT NULL REFERENCES queue_items(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    state TEXT NOT NULL CHECK(
        state IN ('positive', 'negative', 'uncertain', 'unreviewed')
    ),
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fingerprint_audit (
    id INTEGER PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('available', 'unavailable', 'error')),
    detail TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tracks_project ON tracks(project_id);
CREATE INDEX IF NOT EXISTS idx_tracks_group ON tracks(project_id, group_id);
CREATE INDEX IF NOT EXISTS idx_tracks_split ON tracks(project_id, split);
CREATE INDEX IF NOT EXISTS idx_sources_project ON sources(project_id);
CREATE INDEX IF NOT EXISTS idx_queue_project_round
    ON queue_items(project_id, round_number);
CREATE INDEX IF NOT EXISTS idx_annotation_queue_label
    ON annotation_events(queue_item_id, label, id);
"""


class AnnotationStore:
    """Own the versioned annotation database and its connection settings."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)

    def connect(self) -> sqlite3.Connection:
        """Open a configured connection with row access by column name."""
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection and always close its file handle."""
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create schema version 1, or validate an existing annotation database."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            existing = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name = 'schema_meta'"
            ).fetchone()
            if existing is not None:
                row = connection.execute(
                    "SELECT version FROM schema_meta LIMIT 1"
                ).fetchone()
                if row is None or int(row["version"]) != SCHEMA_VERSION:
                    found = "missing" if row is None else str(row["version"])
                    raise RuntimeError(
                        "Annotation database schema is incompatible: "
                        f"expected {SCHEMA_VERSION}, found {found}."
                    )

            connection.executescript(SCHEMA_SQL)
            row = connection.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_meta(version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )

    def schema_version(self) -> int:
        """Return the stored annotation schema version."""
        with self.connection() as connection:
            row = connection.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError("Annotation database has no schema version.")
        return int(row["version"])

    def create_project(self, name: str) -> int:
        """Create a project or return the existing project with the same name."""
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Project name must not be empty.")
        created_at = datetime.now(timezone.utc).isoformat()
        with self.connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO projects(name, created_at) VALUES (?, ?)",
                (normalized_name, created_at),
            )
            row = connection.execute(
                "SELECT id FROM projects WHERE name = ?",
                (normalized_name,),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Could not create annotation project: {normalized_name}")
        return int(row["id"])

    def is_split_frozen(self, project_id: int) -> bool:
        """Return whether a project has immutable split assignments."""
        with self.connection() as connection:
            row = connection.execute(
                "SELECT split_frozen_at FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown annotation project ID: {project_id}")
        return row["split_frozen_at"] is not None
