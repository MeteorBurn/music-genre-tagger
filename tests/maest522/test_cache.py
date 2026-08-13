from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from tools.maest522.cache import (
    CacheKey,
    CacheValidationError,
    TeacherCache,
    TeacherRecord,
)


class TeacherCacheTests(TestCase):
    def test_round_trip_and_key_components_invalidate_cache(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache = TeacherCache(Path(temp_dir))
            key = CacheKey(
                track_id="sha256:track-a",
                offset_milliseconds=5000,
                teacher_sha256="a" * 64,
                preprocessing_version="maest-mel-v1",
            )
            record = TeacherRecord(
                legacy_logits=torch.arange(519, dtype=torch.float32),
                pooled_embedding=torch.arange(768, dtype=torch.float32),
            )

            cache.put_atomic(key, record)
            restored = cache.get(key)

            self.assertIsNotNone(restored)
            torch.testing.assert_close(restored.legacy_logits, record.legacy_logits)
            torch.testing.assert_close(
                restored.pooled_embedding,
                record.pooled_embedding,
            )
            self.assertIsNone(cache.get(replace(key, teacher_sha256="b" * 64)))
            self.assertIsNone(
                cache.get(replace(key, preprocessing_version="maest-mel-v2"))
            )

    def test_rejects_invalid_shapes_before_write(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache = TeacherCache(Path(temp_dir))
            key = CacheKey("track", 0, "a" * 64, "v1")
            record = TeacherRecord(torch.zeros(518), torch.zeros(768))

            with self.assertRaisesRegex(ValueError, "519"):
                cache.put_atomic(key, record)

    def test_detects_cache_file_corruption(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache = TeacherCache(root)
            key = CacheKey("track", 0, "a" * 64, "v1")
            record = TeacherRecord(torch.zeros(519), torch.zeros(768))
            cache.put_atomic(key, record)
            cache_file = next((root / "records").glob("*.safetensors"))
            cache_file.write_bytes(cache_file.read_bytes() + b"corrupt")

            with self.assertRaisesRegex(CacheValidationError, "checksum"):
                cache.get(key)
