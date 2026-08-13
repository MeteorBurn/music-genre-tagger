"""Content-addressed safetensors cache for immutable MAEST teacher outputs."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor


@dataclass(frozen=True)
class CacheKey:
    track_id: str
    offset_milliseconds: int
    teacher_sha256: str
    preprocessing_version: str

    def digest(self) -> str:
        payload = json.dumps(
            {
                "offset_milliseconds": self.offset_milliseconds,
                "preprocessing_version": self.preprocessing_version,
                "teacher_sha256": self.teacher_sha256,
                "track_id": self.track_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TeacherRecord:
    legacy_logits: Tensor
    pooled_embedding: Tensor


class CacheValidationError(RuntimeError):
    """Raised when a present teacher cache record fails integrity checks."""


class TeacherCache:
    """SQLite-indexed, checksum-verified teacher cache."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.records_dir = self.root / "records"
        self.database_path = self.root / "cache-index.db"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS teacher_records ("
                "cache_id TEXT PRIMARY KEY, "
                "track_id TEXT NOT NULL, "
                "offset_milliseconds INTEGER NOT NULL, "
                "teacher_sha256 TEXT NOT NULL, "
                "preprocessing_version TEXT NOT NULL, "
                "filename TEXT NOT NULL, "
                "logits_shape TEXT NOT NULL, "
                "embedding_shape TEXT NOT NULL, "
                "logits_dtype TEXT NOT NULL, "
                "embedding_dtype TEXT NOT NULL, "
                "checksum_sha256 TEXT NOT NULL, "
                "created_at TEXT NOT NULL"
                ")"
            )

    @staticmethod
    def _validate_record(record: TeacherRecord) -> None:
        if tuple(record.legacy_logits.shape) != (519,):
            raise ValueError(
                "teacher legacy logits must have shape (519,), got "
                f"{tuple(record.legacy_logits.shape)}"
            )
        if tuple(record.pooled_embedding.shape) != (768,):
            raise ValueError(
                "teacher pooled embedding must have shape (768,), got "
                f"{tuple(record.pooled_embedding.shape)}"
            )
        if not record.legacy_logits.is_floating_point():
            raise ValueError("teacher legacy logits must be floating point")
        if not record.pooled_embedding.is_floating_point():
            raise ValueError("teacher pooled embedding must be floating point")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def put_atomic(self, key: CacheKey, record: TeacherRecord) -> None:
        """Write a complete safetensors record before publishing its index row."""
        self._validate_record(record)
        cache_id = key.digest()
        record_path = self.records_dir / f"{cache_id}.safetensors"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.records_dir,
            prefix=f".{cache_id}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        tensors = {
            "legacy_logits": record.legacy_logits.detach().cpu().contiguous(),
            "pooled_embedding": record.pooled_embedding.detach().cpu().contiguous(),
        }
        try:
            save_file(tensors, temporary_path)
            with temporary_path.open("r+b") as temporary_file:
                os.fsync(temporary_file.fileno())
            checksum = self._sha256_file(temporary_path)
            os.replace(temporary_path, record_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO teacher_records("
                "cache_id, track_id, offset_milliseconds, teacher_sha256, "
                "preprocessing_version, filename, logits_shape, embedding_shape, "
                "logits_dtype, embedding_dtype, checksum_sha256, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cache_id,
                    key.track_id,
                    key.offset_milliseconds,
                    key.teacher_sha256,
                    key.preprocessing_version,
                    record_path.name,
                    json.dumps(list(tensors["legacy_logits"].shape)),
                    json.dumps(list(tensors["pooled_embedding"].shape)),
                    str(tensors["legacy_logits"].dtype),
                    str(tensors["pooled_embedding"].dtype),
                    checksum,
                    created_at,
                ),
            )

    def get(self, key: CacheKey) -> TeacherRecord | None:
        """Return a verified cache record, or ``None`` for an exact-key miss."""
        cache_id = key.digest()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM teacher_records WHERE cache_id = ?",
                (cache_id,),
            ).fetchone()
        if row is None:
            return None
        record_path = self.records_dir / str(row["filename"])
        if not record_path.is_file():
            raise CacheValidationError(f"cache file is missing: {record_path}")
        actual_checksum = self._sha256_file(record_path)
        if actual_checksum != str(row["checksum_sha256"]):
            raise CacheValidationError(
                f"cache checksum mismatch for {record_path.name}"
            )
        try:
            tensors = load_file(record_path, device="cpu")
        except Exception as error:
            raise CacheValidationError(
                f"could not read cache record {record_path.name}: {error}"
            ) from error
        if set(tensors) != {"legacy_logits", "pooled_embedding"}:
            raise CacheValidationError(
                f"cache record {record_path.name} has unexpected tensor names"
            )
        record = TeacherRecord(
            legacy_logits=tensors["legacy_logits"],
            pooled_embedding=tensors["pooled_embedding"],
        )
        try:
            self._validate_record(record)
        except ValueError as error:
            raise CacheValidationError(str(error)) from error
        expected_metadata = {
            "logits_shape": list(record.legacy_logits.shape),
            "embedding_shape": list(record.pooled_embedding.shape),
            "logits_dtype": str(record.legacy_logits.dtype),
            "embedding_dtype": str(record.pooled_embedding.dtype),
        }
        for field, actual_value in expected_metadata.items():
            stored_value = row[field]
            if field.endswith("_shape"):
                stored_value = json.loads(str(stored_value))
            if stored_value != actual_value:
                raise CacheValidationError(
                    f"cache metadata mismatch for {record_path.name}: {field}"
                )
        return record
