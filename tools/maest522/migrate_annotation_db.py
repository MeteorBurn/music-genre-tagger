"""Explicit backup-first migration for the MAEST 522 annotation database."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .constants import (
    DEFAULT_NEGATIVE_TARGET,
    DEFAULT_POSITIVE_TARGET,
    NEW_LABELS,
    SCHEMA_VERSION,
)


LEGACY_MINIMAL_DEEP_TECH_LABEL = "Electronic---DeepTech-Minimal"
MINIMAL_DEEP_TECH_LABEL = "Electronic---Minimal-Deep-Tech"
EXPECTED_SEED_COUNT = 100
LABEL_SQL = ", ".join(f"'{label}'" for label in NEW_LABELS)


@dataclass(frozen=True)
class MigrationReport:
    database_path: str
    backup_path: str
    source_sha256: str
    backup_sha256: str
    converted_positive: int
    converted_negative: int
    schema_version: int
    integrity_check: str
    foreign_key_violations: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _validate_database(connection: sqlite3.Connection) -> None:
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        raise ValueError(f"annotation database integrity check failed: {integrity}")
    foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_keys:
        raise ValueError(
            f"annotation database has {len(foreign_keys)} foreign-key violations"
        )
    version_row = connection.execute(
        "SELECT version FROM schema_meta LIMIT 1"
    ).fetchone()
    if version_row is None or int(version_row["version"]) != 1:
        found = "missing" if version_row is None else str(version_row["version"])
        raise ValueError(f"migration requires schema version 1; found {found}")
    for table in ("queue_rounds", "queue_items", "annotation_events"):
        count = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        if count:
            raise ValueError(
                "legacy queue tables must be empty before migration; "
                f"{table} contains {count} rows"
            )


def _validated_source_tracks(
    connection: sqlite3.Connection,
    project_id: int,
    source_id: int,
    expected_role: str,
    description: str,
) -> tuple[sqlite3.Row, tuple[sqlite3.Row, ...]]:
    source = connection.execute(
        "SELECT id, project_id, kind, source_path, candidate_role "
        "FROM sources WHERE id = ?",
        (source_id,),
    ).fetchone()
    if source is None:
        raise ValueError(f"{description} source ID does not exist: {source_id}")
    if int(source["project_id"]) != project_id:
        raise ValueError(f"{description} source belongs to another project")
    if str(source["candidate_role"]) != expected_role:
        raise ValueError(
            f"{description} source must use role {expected_role}; "
            f"found {source['candidate_role']}"
        )
    if str(source["kind"]) not in {"m3u", "m3u8"}:
        raise ValueError(f"{description} source must be an M3U/M3U8 playlist")
    playlist_path = Path(str(source["source_path"])).resolve()
    if not playlist_path.is_file():
        raise ValueError(f"{description} source playlist is unavailable: {playlist_path}")
    rows = tuple(
        connection.execute(
            "SELECT tracks.id, tracks.path, tracks.exact_sha256, "
            "track_sources.suggested_label FROM track_sources "
            "JOIN tracks ON tracks.id = track_sources.track_id "
            "WHERE track_sources.source_id = ? AND tracks.project_id = ? "
            "ORDER BY tracks.id",
            (source_id, project_id),
        ).fetchall()
    )
    if len(rows) != EXPECTED_SEED_COUNT:
        raise ValueError(
            f"{description} source must link exactly {EXPECTED_SEED_COUNT} tracks; "
            f"found {len(rows)}"
        )
    identities = {str(row["exact_sha256"]) for row in rows}
    if len(identities) != EXPECTED_SEED_COUNT:
        raise ValueError(f"{description} source contains duplicate track identities")
    invalid_labels = {
        str(row["suggested_label"])
        for row in rows
        if str(row["suggested_label"]) != LEGACY_MINIMAL_DEEP_TECH_LABEL
    }
    if invalid_labels:
        raise ValueError(
            f"{description} source contains unexpected legacy labels: "
            + ", ".join(sorted(invalid_labels))
        )
    missing = [str(row["path"]) for row in rows if not Path(str(row["path"])).is_file()]
    if missing:
        raise ValueError(
            f"{description} source has unavailable audio files: " + "; ".join(missing[:5])
        )
    return source, rows


def _create_v2_tables(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE annotation_events")
    connection.execute("DROP TABLE queue_credits")
    connection.execute("DROP TABLE queue_items")
    connection.execute("DROP TABLE queue_rounds")
    connection.execute(
        f"""CREATE TABLE queue_items (
            id INTEGER PRIMARY KEY,
            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
            label TEXT NOT NULL CHECK(label IN ({LABEL_SQL})),
            round_number INTEGER NOT NULL CHECK(round_number > 0),
            acquisition_kind TEXT NOT NULL,
            acquisition_score REAL,
            created_at TEXT NOT NULL,
            UNIQUE(project_id, track_id, label)
        )"""
    )
    connection.execute(
        f"""CREATE TABLE queue_rounds (
            id INTEGER PRIMARY KEY,
            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            label TEXT NOT NULL CHECK(label IN ({LABEL_SQL})),
            round_number INTEGER NOT NULL CHECK(round_number > 0),
            split TEXT NOT NULL CHECK(split IN ('train', 'val', 'test')),
            acquisition_kind TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(project_id, label, round_number, split)
        )"""
    )
    connection.execute(
        """CREATE TABLE queue_credits (
            queue_item_id INTEGER NOT NULL REFERENCES queue_items(id) ON DELETE CASCADE,
            label TEXT NOT NULL,
            candidate_role TEXT NOT NULL CHECK(candidate_role IN (
                'positive_candidate', 'hard_negative_candidate', 'unlabeled_pool'
            )),
            PRIMARY KEY(queue_item_id, label)
        )"""
    )
    connection.execute(
        """CREATE TABLE annotation_events (
            id INTEGER PRIMARY KEY,
            queue_item_id INTEGER NOT NULL REFERENCES queue_items(id) ON DELETE CASCADE,
            label TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN (
                'positive', 'negative', 'uncertain', 'unreviewed'
            )),
            note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )"""
    )
    connection.execute(
        f"""CREATE TABLE label_goals (
            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            label TEXT NOT NULL CHECK(label IN ({LABEL_SQL})),
            positive_target INTEGER NOT NULL CHECK(positive_target > 0),
            negative_target INTEGER NOT NULL CHECK(negative_target > 0),
            updated_at TEXT NOT NULL,
            PRIMARY KEY(project_id, label)
        )"""
    )
    connection.execute(
        f"""CREATE TABLE confirmed_label_batches (
            id INTEGER PRIMARY KEY,
            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            label TEXT NOT NULL CHECK(label IN ({LABEL_SQL})),
            state TEXT NOT NULL CHECK(state IN ('positive', 'negative')),
            source_kind TEXT NOT NULL CHECK(source_kind IN ('m3u', 'm3u8')),
            source_path TEXT NOT NULL,
            playlist_sha256 TEXT NOT NULL,
            discovered_count INTEGER NOT NULL CHECK(discovered_count >= 0),
            new_count INTEGER NOT NULL CHECK(new_count >= 0),
            existing_count INTEGER NOT NULL CHECK(existing_count >= 0),
            created_at TEXT NOT NULL
        )"""
    )
    connection.execute(
        f"""CREATE TABLE confirmed_label_events (
            id INTEGER PRIMARY KEY,
            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
            label TEXT NOT NULL CHECK(label IN ({LABEL_SQL})),
            state TEXT NOT NULL CHECK(state IN ('positive', 'negative', 'uncertain')),
            event_kind TEXT NOT NULL CHECK(event_kind IN (
                'trusted_import', 'manual_review', 'correction'
            )),
            batch_id INTEGER REFERENCES confirmed_label_batches(id) ON DELETE RESTRICT,
            note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )"""
    )
    connection.execute(
        "CREATE INDEX idx_queue_project_round "
        "ON queue_items(project_id, label, round_number)"
    )
    connection.execute(
        "CREATE INDEX idx_queue_rounds_project_split "
        "ON queue_rounds(project_id, label, split, round_number)"
    )
    connection.execute(
        "CREATE INDEX idx_queue_credits_label "
        "ON queue_credits(label, candidate_role)"
    )
    connection.execute(
        "CREATE INDEX idx_annotation_queue_label "
        "ON annotation_events(queue_item_id, label, id)"
    )
    connection.execute(
        "CREATE INDEX idx_confirmed_event_current "
        "ON confirmed_label_events(project_id, track_id, label, id)"
    )
    connection.execute(
        "CREATE INDEX idx_confirmed_event_progress "
        "ON confirmed_label_events(project_id, label, state, id)"
    )
    connection.execute(
        "CREATE INDEX idx_confirmed_event_batch ON confirmed_label_events(batch_id)"
    )


def migrate_v1_to_v2(
    database_path: Path,
    backup_path: Path,
    project_id: int,
    positive_source_id: int,
    negative_source_id: int,
) -> MigrationReport:
    """Migrate the exact trusted v1 seed project after creating a verified backup."""
    resolved_database = Path(database_path).resolve()
    resolved_backup = Path(backup_path).resolve()
    if not resolved_database.is_file():
        raise FileNotFoundError(f"annotation database is unavailable: {resolved_database}")
    if resolved_database == resolved_backup:
        raise ValueError("backup path must differ from the live database path")
    if resolved_backup.exists():
        raise FileExistsError(f"backup path already exists: {resolved_backup}")

    with closing(_connect(resolved_database)) as connection:
        _validate_database(connection)
        positive_source, positive_rows = _validated_source_tracks(
            connection,
            project_id,
            positive_source_id,
            "positive_candidate",
            "positive source",
        )
        negative_source, negative_rows = _validated_source_tracks(
            connection,
            project_id,
            negative_source_id,
            "hard_negative_candidate",
            "negative source",
        )
        overlap = {str(row["exact_sha256"]) for row in positive_rows}.intersection(
            str(row["exact_sha256"]) for row in negative_rows
        )
        if overlap:
            raise ValueError("positive and negative sources contain overlapping tracks")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    resolved_backup.parent.mkdir(parents=True, exist_ok=True)
    pre_migration_sha256 = _sha256_file(resolved_database)
    shutil.copy2(resolved_database, resolved_backup)
    backup_sha256 = _sha256_file(resolved_backup)
    if backup_sha256 != pre_migration_sha256:
        resolved_backup.unlink(missing_ok=True)
        raise RuntimeError("verified database backup SHA-256 does not match the source")

    created_at = datetime.now(timezone.utc).isoformat()
    connection = _connect(resolved_database)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        _validate_database(connection)
        positive_source, positive_rows = _validated_source_tracks(
            connection,
            project_id,
            positive_source_id,
            "positive_candidate",
            "positive source",
        )
        negative_source, negative_rows = _validated_source_tracks(
            connection,
            project_id,
            negative_source_id,
            "hard_negative_candidate",
            "negative source",
        )
        _create_v2_tables(connection)
        for label in NEW_LABELS:
            connection.execute(
                "INSERT INTO label_goals(project_id, label, positive_target, "
                "negative_target, updated_at) VALUES (?, ?, ?, ?, ?)",
                (
                    project_id,
                    label,
                    DEFAULT_POSITIVE_TARGET,
                    DEFAULT_NEGATIVE_TARGET,
                    created_at,
                ),
            )
        batch_ids: dict[str, int] = {}
        for state, source, rows in (
            ("positive", positive_source, positive_rows),
            ("negative", negative_source, negative_rows),
        ):
            source_path = Path(str(source["source_path"])).resolve()
            cursor = connection.execute(
                "INSERT INTO confirmed_label_batches("
                "project_id, label, state, source_kind, source_path, "
                "playlist_sha256, discovered_count, new_count, existing_count, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (
                    project_id,
                    MINIMAL_DEEP_TECH_LABEL,
                    state,
                    str(source["kind"]),
                    str(source_path),
                    _sha256_file(source_path),
                    len(rows),
                    len(rows),
                    created_at,
                ),
            )
            batch_ids[state] = int(cursor.lastrowid)
            for row in rows:
                connection.execute(
                    "INSERT INTO confirmed_label_events("
                    "project_id, track_id, label, state, event_kind, batch_id, "
                    "note, created_at) VALUES (?, ?, ?, ?, 'trusted_import', ?, '', ?)",
                    (
                        project_id,
                        int(row["id"]),
                        MINIMAL_DEEP_TECH_LABEL,
                        state,
                        batch_ids[state],
                        created_at,
                    ),
                )
        connection.execute(
            "UPDATE track_sources SET suggested_label = ? "
            "WHERE source_id IN (?, ?) AND suggested_label = ?",
            (
                MINIMAL_DEEP_TECH_LABEL,
                positive_source_id,
                negative_source_id,
                LEGACY_MINIMAL_DEEP_TECH_LABEL,
            ),
        )
        connection.execute("UPDATE schema_meta SET version = ?", (SCHEMA_VERSION,))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    with closing(_connect(resolved_database)) as verification:
        integrity_check = str(
            verification.execute("PRAGMA integrity_check").fetchone()[0]
        )
        foreign_key_violations = len(
            verification.execute("PRAGMA foreign_key_check").fetchall()
        )
        version = int(
            verification.execute("SELECT version FROM schema_meta").fetchone()[0]
        )
        counts = {
            str(row["state"]): int(row["count"])
            for row in verification.execute(
                "SELECT state, COUNT(*) AS count FROM confirmed_label_events "
                "GROUP BY state"
            ).fetchall()
        }
    if integrity_check != "ok" or foreign_key_violations or version != SCHEMA_VERSION:
        raise RuntimeError("post-migration database verification failed")
    if counts != {"negative": EXPECTED_SEED_COUNT, "positive": EXPECTED_SEED_COUNT}:
        raise RuntimeError(f"post-migration seed counts are invalid: {counts}")
    return MigrationReport(
        database_path=str(resolved_database),
        backup_path=str(resolved_backup),
        source_sha256=_sha256_file(resolved_database),
        backup_sha256=backup_sha256,
        converted_positive=EXPECTED_SEED_COUNT,
        converted_negative=EXPECTED_SEED_COUNT,
        schema_version=version,
        integrity_check=integrity_check,
        foreign_key_violations=foreign_key_violations,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly migrate a trusted MAEST 522 annotation DB v1 to v2."
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--backup", required=True, type=Path)
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--positive-source-id", required=True, type=int)
    parser.add_argument("--negative-source-id", required=True, type=int)
    args = parser.parse_args()
    report = migrate_v1_to_v2(
        args.db,
        args.backup,
        args.project_id,
        args.positive_source_id,
        args.negative_source_id,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
