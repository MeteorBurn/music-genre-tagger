import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.constants import NEW_LABELS
from tools.maest522.migrate_annotation_db import migrate_v1_to_v2


LEGACY_LABEL = "Electronic---DeepTech-Minimal"


V1_SCHEMA = """
CREATE TABLE schema_meta(version INTEGER NOT NULL);
INSERT INTO schema_meta(version) VALUES (1);
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    split_frozen_at TEXT
);
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    exact_sha256 TEXT NOT NULL,
    acoustic_fingerprint TEXT,
    duration_seconds REAL NOT NULL,
    artist TEXT,
    release_id TEXT,
    group_id TEXT,
    split TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, exact_sha256)
);
CREATE TABLE sources (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    source_path TEXT NOT NULL,
    candidate_role TEXT NOT NULL,
    imported_at TEXT NOT NULL
);
CREATE TABLE track_sources (
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    suggested_label TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(track_id, source_id, suggested_label)
);
CREATE TABLE queue_items (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    round_number INTEGER NOT NULL,
    acquisition_kind TEXT NOT NULL,
    acquisition_score REAL,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, track_id)
);
CREATE TABLE queue_rounds (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    round_number INTEGER NOT NULL,
    split TEXT NOT NULL,
    acquisition_kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, round_number, split)
);
CREATE TABLE queue_credits (
    queue_item_id INTEGER NOT NULL REFERENCES queue_items(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    candidate_role TEXT NOT NULL,
    PRIMARY KEY(queue_item_id, label)
);
CREATE TABLE annotation_events (
    id INTEGER PRIMARY KEY,
    queue_item_id INTEGER NOT NULL REFERENCES queue_items(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    state TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE fingerprint_audit (
    id INTEGER PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_v1_database(root: Path) -> Path:
    database_path = root / "annotations.db"
    positive_playlist = root / "positive.m3u"
    negative_playlist = root / "negative.m3u"
    positive_lines = ["#EXTM3U"]
    negative_lines = ["#EXTM3U"]
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            connection.executescript(V1_SCHEMA)
            connection.execute(
                "INSERT INTO projects(id, name, created_at) VALUES (1, 'seed', 'now')"
            )
            connection.execute(
                "INSERT INTO sources(id, project_id, kind, source_path, "
                "candidate_role, imported_at) VALUES (1, 1, 'm3u', ?, "
                "'positive_candidate', 'now')",
                (str(positive_playlist),),
            )
            connection.execute(
                "INSERT INTO sources(id, project_id, kind, source_path, "
                "candidate_role, imported_at) VALUES (2, 1, 'm3u', ?, "
                "'hard_negative_candidate', 'now')",
                (str(negative_playlist),),
            )
            for index in range(200):
                audio_path = root / f"track-{index:03d}.wav"
                audio_path.write_bytes(f"audio-{index}".encode("ascii"))
                sha256 = _digest(audio_path)
                cursor = connection.execute(
                    "INSERT INTO tracks(project_id, path, exact_sha256, "
                    "duration_seconds, created_at) VALUES (1, ?, ?, 60, 'now')",
                    (str(audio_path), sha256),
                )
                source_id = 1 if index < 100 else 2
                connection.execute(
                    "INSERT INTO track_sources(track_id, source_id, suggested_label) "
                    "VALUES (?, ?, ?)",
                    (cursor.lastrowid, source_id, LEGACY_LABEL),
                )
                if source_id == 1:
                    positive_lines.append(str(audio_path))
                else:
                    negative_lines.append(str(audio_path))
    positive_playlist.write_text("\n".join(positive_lines) + "\n", encoding="utf-8")
    negative_playlist.write_text("\n".join(negative_lines) + "\n", encoding="utf-8")
    return database_path


class AnnotationMigrationTests(TestCase):
    def test_ordinary_startup_rejects_v1_without_mutating_it(self) -> None:
        with TemporaryDirectory() as temp_dir:
            database_path = build_v1_database(Path(temp_dir))

            with self.assertRaisesRegex(
                RuntimeError,
                "expected 2, found 1",
            ):
                AnnotationStore(database_path).initialize()

            with closing(sqlite3.connect(database_path)) as connection:
                version = connection.execute(
                    "SELECT version FROM schema_meta"
                ).fetchone()[0]
                confirmed_table = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='confirmed_label_events'"
                ).fetchone()
            self.assertEqual(version, 1)
            self.assertIsNone(confirmed_table)

    def test_migrates_exact_trusted_seed_sources_after_verified_backup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = build_v1_database(root)
            backup_path = root / "backups" / "annotations-v1.sqlite"

            report = migrate_v1_to_v2(
                database_path,
                backup_path,
                project_id=1,
                positive_source_id=1,
                negative_source_id=2,
            )

            self.assertEqual(report.converted_positive, 100)
            self.assertEqual(report.converted_negative, 100)
            self.assertEqual(report.schema_version, 2)
            self.assertEqual(report.integrity_check, "ok")
            self.assertEqual(report.foreign_key_violations, 0)
            self.assertTrue(backup_path.is_file())
            self.assertEqual(report.backup_sha256, _digest(backup_path))
            self.assertNotEqual(report.source_sha256, report.backup_sha256)
            store = AnnotationStore(database_path)
            store.initialize()
            with store.connection() as connection:
                counts = connection.execute(
                    "SELECT label, state, COUNT(*) FROM confirmed_label_events "
                    "GROUP BY label, state ORDER BY label, state"
                ).fetchall()
                goals = connection.execute(
                    "SELECT label, positive_target, negative_target "
                    "FROM label_goals WHERE project_id = 1 ORDER BY label"
                ).fetchall()
                links = connection.execute(
                    "SELECT DISTINCT suggested_label FROM track_sources"
                ).fetchall()
            self.assertEqual(
                [(row[0], row[1], row[2]) for row in counts],
                [
                    ("Electronic---Minimal-Deep-Tech", "negative", 100),
                    ("Electronic---Minimal-Deep-Tech", "positive", 100),
                ],
            )
            self.assertEqual(
                [(row[0], row[1], row[2]) for row in goals],
                sorted((label, 1000, 1000) for label in NEW_LABELS),
            )
            self.assertEqual(
                [row[0] for row in links],
                ["Electronic---Minimal-Deep-Tech"],
            )

    def test_rejects_invalid_source_contract_without_mutating_v1(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = build_v1_database(root)
            before = _digest(database_path)
            with closing(sqlite3.connect(database_path)) as connection:
                with connection:
                    connection.execute(
                        "UPDATE sources SET candidate_role = 'unlabeled_pool' WHERE id = 2"
                    )
            before = _digest(database_path)

            with self.assertRaisesRegex(ValueError, "negative source"):
                migrate_v1_to_v2(
                    database_path,
                    root / "backup.sqlite",
                    project_id=1,
                    positive_source_id=1,
                    negative_source_id=2,
                )

            self.assertEqual(_digest(database_path), before)
            with closing(sqlite3.connect(database_path)) as connection:
                version = connection.execute(
                    "SELECT version FROM schema_meta"
                ).fetchone()[0]
                confirmed_table = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='confirmed_label_events'"
                ).fetchone()
            self.assertEqual(version, 1)
            self.assertIsNone(confirmed_table)

    def test_rejects_nonempty_legacy_queues_and_existing_backup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = build_v1_database(root)
            backup_path = root / "backup.sqlite"
            backup_path.write_bytes(b"keep")
            with self.assertRaisesRegex(FileExistsError, "backup"):
                migrate_v1_to_v2(
                    database_path,
                    backup_path,
                    project_id=1,
                    positive_source_id=1,
                    negative_source_id=2,
                )
            backup_path.unlink()
            with closing(sqlite3.connect(database_path)) as connection:
                with connection:
                    connection.execute(
                        "INSERT INTO queue_rounds(project_id, round_number, split, "
                        "acquisition_kind, created_at) VALUES (1, 1, 'train', "
                        "'fixture', 'now')"
                    )
            with self.assertRaisesRegex(ValueError, "queue"):
                migrate_v1_to_v2(
                    database_path,
                    backup_path,
                    project_id=1,
                    positive_source_id=1,
                    negative_source_id=2,
                )
