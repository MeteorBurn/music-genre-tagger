import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
import soundfile

from tools.maest522.constants import NEW_LABELS
from tools.maest522.training_data import (
    decode_manifest_row,
    build_maest_mel,
    load_audio_window,
    load_manifest,
    load_replay_manifest,
    select_window_offsets,
)


class TrainingDataTests(TestCase):
    def test_window_offsets_match_production_clamping_and_deduplication(self) -> None:
        self.assertEqual(select_window_offsets(100.0), (5.0, 35.0, 65.0))
        self.assertEqual(select_window_offsets(30.0), (0.0,))
        self.assertEqual(select_window_offsets(12.5), (0.0,))
        self.assertEqual(select_window_offsets(31.0), (0.0, 0.5, 1.0))
        self.assertEqual(select_window_offsets(30.1), (0.0, 0.05, 0.1))

    def test_decodes_positive_negative_and_uncertain_masks(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "track.wav"
            audio_path.write_bytes(b"audio-placeholder")
            row = self._row("track-a", "group-a", "train", audio_path.name)
            row["labels"] = {
                NEW_LABELS[0]: "positive",
                NEW_LABELS[1]: "negative",
                NEW_LABELS[2]: "uncertain",
            }
            row["label_mask"][NEW_LABELS[2]] = 0

            sample = decode_manifest_row(row, root, require_labels=True)

            torch.testing.assert_close(sample.targets, torch.tensor([1.0, 0.0, 0.0]))
            torch.testing.assert_close(sample.target_mask, torch.tensor([1.0, 1.0, 0.0]))
            self.assertEqual(sample.audio_path, audio_path.resolve())
            self.assertEqual(sample.window_offsets_seconds, (5.0, 35.0, 65.0))

    def test_manifest_rejects_split_leakage_missing_audio_and_digest_drift(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "first.wav").write_bytes(b"first")
            (root / "second.wav").write_bytes(b"second")
            audit_digest = "a" * 64
            rows = [
                self._row("track-a", "shared-group", "train", "first.wav"),
                self._row("track-b", "shared-group", "val", "second.wav"),
            ]
            manifest_path = self._write_manifest(root, rows, audit_digest)

            with self.assertRaisesRegex(ValueError, "multiple splits"):
                load_manifest(manifest_path, audit_digest)

            rows[1]["group_id"] = "group-b"
            rows[1]["audio_ref"] = "missing.wav"
            manifest_path = self._write_manifest(root, rows, audit_digest)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                load_manifest(manifest_path, audit_digest)

            rows[1]["audio_ref"] = "second.wav"
            manifest_path = self._write_manifest(root, rows, audit_digest)
            with self.assertRaisesRegex(ValueError, "split audit"):
                load_manifest(manifest_path, "b" * 64)
            wrong_manifest_digest = "0" * 64
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                load_manifest(
                    manifest_path,
                    audit_digest,
                    expected_manifest_sha256=wrong_manifest_digest,
                )

    def test_loads_valid_manifest_and_rejects_incomplete_labels(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("train.wav", "val.wav", "test.wav"):
                (root / name).write_bytes(name.encode("ascii"))
            audit_digest = "c" * 64
            rows = [
                self._row("track-a", "group-a", "train", "train.wav"),
                self._row("track-b", "group-b", "val", "val.wav"),
                self._row("track-c", "group-c", "test", "test.wav"),
            ]
            manifest_path = self._write_manifest(root, rows, audit_digest)
            manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

            manifest = load_manifest(
                manifest_path,
                audit_digest,
                expected_manifest_sha256=manifest_digest,
            )

            self.assertEqual(len(manifest.samples), 3)
            self.assertEqual(manifest.manifest_sha256, manifest_digest)
            self.assertEqual(manifest.split_audit_sha256, audit_digest)
            self.assertEqual(manifest.by_split("val")[0].track_id, "track-b")

            del rows[0]["labels"][NEW_LABELS[2]]
            manifest_path = self._write_manifest(root, rows, audit_digest)
            with self.assertRaisesRegex(ValueError, NEW_LABELS[2]):
                load_manifest(manifest_path, audit_digest)

            rows[0]["labels"][NEW_LABELS[2]] = "unreviewed"
            rows[0]["label_mask"][NEW_LABELS[2]] = 1
            manifest_path = self._write_manifest(root, rows, audit_digest)
            with self.assertRaisesRegex(ValueError, "label_mask conflicts"):
                load_manifest(manifest_path, audit_digest)

    def test_replay_manifest_has_no_new_label_supervision(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "replay.wav").write_bytes(b"replay")
            (root / "holdout.wav").write_bytes(b"holdout")
            rows = [
                self._row("replay-a", "group-a", "replay_train", "replay.wav"),
                self._row(
                    "holdout-a",
                    "group-b",
                    "regression_holdout",
                    "holdout.wav",
                ),
            ]
            for row in rows:
                row.pop("labels")
            manifest_path = root / "replay.jsonl"
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

            manifest = load_replay_manifest(manifest_path, digest)

            self.assertEqual(len(manifest.by_split("replay_train")), 1)
            torch.testing.assert_close(manifest.samples[0].target_mask, torch.zeros(3))
            torch.testing.assert_close(manifest.samples[0].targets, torch.zeros(3))

    def test_audio_decode_is_mono_16khz_and_uses_installed_maest_mel(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "short.wav"
            stereo = torch.stack(
                (
                    torch.linspace(-0.5, 0.5, 12_000),
                    torch.linspace(0.5, -0.5, 12_000),
                ),
                dim=1,
            ).numpy()
            soundfile.write(audio_path, stereo, 8_000)
            row = self._row("short", "group-short", "train", audio_path.name)
            row["duration_seconds"] = 1.5
            row["window_offsets_seconds"] = [0.0]
            sample = decode_manifest_row(row, root)

            waveform = load_audio_window(sample, 0.0)
            mel = build_maest_mel(waveform)

            self.assertEqual(tuple(waveform.shape), (24_000,))
            self.assertEqual(mel.shape[0], 96)
            with self.assertRaisesRegex(ValueError, "frozen window"):
                load_audio_window(sample, 1.0)

    @staticmethod
    def _row(
        track_id: str,
        group_id: str,
        split: str,
        audio_ref: str,
    ) -> dict[str, object]:
        return {
            "track_id": track_id,
            "group_id": group_id,
            "audio_ref": audio_ref,
            "split": split,
            "duration_seconds": 100.0,
            "window_offsets_seconds": [5.0, 35.0, 65.0],
            "candidate_roles": {},
            "labels": {label: "negative" for label in NEW_LABELS},
            "label_mask": {label: 1 for label in NEW_LABELS},
        }

    @staticmethod
    def _write_manifest(
        root: Path,
        rows: list[dict[str, object]],
        audit_digest: str,
    ) -> Path:
        manifest_path = root / "training.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        (root / "dataset_summary.json").write_text(
            json.dumps({"split_audit_sha256": audit_digest}),
            encoding="utf-8",
        )
        return manifest_path
