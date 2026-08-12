import json
import math
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.constants import NEW_LABELS
from tools.maest522.fingerprints import FingerprintResult, fingerprint_project
from tools.maest522.library import import_source
from tools.maest522.manifests import export_training_manifest
from tools.maest522.queues import create_round
from tools.maest522.splits import audit_split_leakage, freeze_group_splits


def write_unique_wav(path: Path, sample_value: int) -> None:
    frame_count = math.ceil(16_000 * 0.1)
    sample = int(sample_value).to_bytes(2, "little", signed=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(sample * frame_count)


class AnnotationWorkflowTests(TestCase):
    @patch("tools.maest522.fingerprints.calculate_fingerprint")
    def test_imports_splits_reviews_reopens_and_exports(self, calculate_mock) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_dir = root / "audio"
            audio_paths = []
            for track_index in range(12):
                audio_path = audio_dir / f"track-{track_index:02d}.wav"
                write_unique_wav(audio_path, sample_value=track_index + 1)
                audio_paths.append(audio_path)
            micro_playlist = root / "microhouse.m3u"
            minimal_playlist = root / "minimal-hard-negatives.m3u8"
            micro_playlist.write_text(
                "\n".join(str(path) for path in audio_paths[:8]) + "\n",
                encoding="utf-8",
            )
            minimal_playlist.write_text(
                "\n".join(str(path) for path in audio_paths[4:]) + "\n",
                encoding="utf-8",
            )
            calculate_mock.side_effect = lambda audio_path, _fpcalc_path: FingerprintResult(
                "available",
                f"fingerprint-{Path(audio_path).stem}",
                0.1,
                "",
            )

            database_path = root / "annotations.db"
            store = AnnotationStore(database_path)
            store.initialize()
            project_id = store.create_project("workflow")
            import_source(
                store,
                project_id,
                audio_dir,
                NEW_LABELS[0],
                "positive_candidate",
            )
            import_source(
                store,
                project_id,
                micro_playlist,
                NEW_LABELS[1],
                "positive_candidate",
            )
            import_source(
                store,
                project_id,
                minimal_playlist,
                NEW_LABELS[2],
                "hard_negative_candidate",
            )

            fingerprint_summary = fingerprint_project(
                store,
                project_id,
                Path("fpcalc"),
            )
            self.assertEqual(fingerprint_summary.available, 12)
            split_summary = freeze_group_splits(store, project_id, seed=522)
            self.assertEqual(sum(split_summary.split_counts.values()), 12)
            self.assertTrue(audit_split_leakage(store, project_id).clean)

            val_round = create_round(
                store, project_id, NEW_LABELS[0], 1, split="val"
            )
            test_round = create_round(
                store, project_id, NEW_LABELS[0], 1, split="test"
            )
            train_round = create_round(
                store, project_id, NEW_LABELS[0], 1, split="train"
            )
            queue_item_ids = (
                val_round.queue_item_ids
                + test_round.queue_item_ids
                + train_round.queue_item_ids
            )
            self.assertEqual(len(queue_item_ids), 12)
            for queue_item_id in queue_item_ids:
                store.append_review(
                    queue_item_id,
                    {
                        NEW_LABELS[0]: "positive",
                        NEW_LABELS[1]: "negative",
                        NEW_LABELS[2]: "uncertain",
                    },
                    "fixture review",
                )

            reopened_store = AnnotationStore(database_path)
            reopened_store.initialize()
            output_path = root / "export" / "training.jsonl"
            report = export_training_manifest(
                reopened_store,
                project_id,
                output_path,
                portable=True,
            )
            exported_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(report.rows_written, 12)
            self.assertEqual(len(exported_rows), 12)
            self.assertTrue(all("path" not in row for row in exported_rows))
            self.assertEqual({row["split"] for row in exported_rows}, {"train", "val", "test"})
            with reopened_store.connection() as connection:
                foreign_key_issues = connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall()
            self.assertEqual(foreign_key_issues, [])
