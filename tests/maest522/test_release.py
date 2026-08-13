import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from safetensors.torch import save_file

from tools.maest522.model_labels import build_522_labels, load_official_519_labels
from tools.maest522.release import ReleaseInputs, stage_release


class ReleaseStagingTests(TestCase):
    def _fixture(self, root: Path) -> ReleaseInputs:
        source_dir = root / "source"
        source_dir.mkdir()
        labels = build_522_labels(load_official_519_labels())
        (source_dir / "labels-522.txt").write_text(
            "\n".join(labels) + "\n",
            encoding="utf-8",
        )
        config = {
            "num_labels": 522,
            "id2label": {str(index): label for index, label in enumerate(labels)},
            "label2id": {label: index for index, label in enumerate(labels)},
        }
        (source_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        save_file(
            {
                "classifier.dense.weight": torch.zeros(522, 768),
                "classifier.dense.bias": torch.zeros(522),
            },
            source_dir / "model.safetensors",
        )
        native_state = {
            "head.1.weight": torch.zeros(522, 768),
            "head.1.bias": torch.zeros(522),
            "head_dist.weight": torch.zeros(522, 768),
            "head_dist.bias": torch.zeros(522),
        }
        torch.save(
            {"state_dict": native_state, "metadata": {"architecture": "maest_522l_pytorch"}},
            source_dir / "native-maest-522.ckpt",
        )
        (source_dir / "feature_extraction_maest.py").write_text(
            "# SPDX-License-Identifier: Apache-2.0\n",
            encoding="utf-8",
        )
        (source_dir / "preprocessor_config.json").write_text(
            json.dumps({"feature_extractor_type": "MAESTFeatureExtractor"}),
            encoding="utf-8",
        )
        (source_dir / "README.md").write_text(
            "---\nlicense: other\n---\n# Model\n",
            encoding="utf-8",
        )
        (source_dir / "evaluation.json").write_text(
            json.dumps({"legacy_gates_passed": True, "test_evaluated": True}),
            encoding="utf-8",
        )
        (source_dir / "parity.json").write_text(
            json.dumps({"passed": True, "windows": 30, "tracks": 10}),
            encoding="utf-8",
        )
        teacher_path = root / "teacher.ckpt"
        torch.save(
            {
                "head.1.weight": torch.zeros(519, 768),
                "head.1.bias": torch.zeros(519),
                "head_dist.weight": torch.zeros(519, 768),
                "head_dist.bias": torch.zeros(519),
            },
            teacher_path,
        )
        trained_path = root / "trained.ckpt"
        trained_path.write_bytes(b"trained")
        manifest_path = root / "training.jsonl"
        manifest_path.write_text('{"track_id":"sha256:a"}\n', encoding="utf-8")
        return ReleaseInputs(
            source_dir=source_dir,
            teacher_checkpoint=teacher_path,
            trained_checkpoint=trained_path,
            dataset_manifest=manifest_path,
            split_audit_sha256="b" * 64,
            git_commit="abc123",
            version="0.1.0",
        )

    def test_stages_allowlisted_files_provenance_and_sorted_hashes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = self._fixture(root)
            output = root / "release"

            report = stage_release(inputs, output)

            self.assertEqual(report.file_count, 11)
            self.assertTrue((output / "provenance.json").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())
            names = [path.name for path in output.iterdir()]
            self.assertEqual(set(names), report.files)
            sums = (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            self.assertEqual(sums, sorted(sums, key=lambda line: line.split("  ", 1)[1]))

    def test_rejects_failed_gates_parity_labels_paths_and_unknown_files(self) -> None:
        mutations = ("evaluation", "parity", "labels", "path", "unknown")
        for mutation in mutations:
            with self.subTest(mutation=mutation), TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                inputs = self._fixture(root)
                source = inputs.source_dir
                if mutation == "evaluation":
                    (source / "evaluation.json").write_text(
                        json.dumps({"legacy_gates_passed": False}), encoding="utf-8"
                    )
                elif mutation == "parity":
                    (source / "parity.json").write_text(
                        json.dumps({"passed": False, "windows": 30, "tracks": 10}),
                        encoding="utf-8",
                    )
                elif mutation == "labels":
                    (source / "labels-522.txt").write_text("wrong\n", encoding="utf-8")
                elif mutation == "path":
                    (source / "README.md").write_text(
                        "private C:\\Users\\Example\\audio.wav",
                        encoding="utf-8",
                    )
                else:
                    (source / "secret.txt").write_text("secret", encoding="utf-8")
                with self.assertRaises((ValueError, RuntimeError)):
                    stage_release(inputs, root / "release")

    def test_rejects_non_519_teacher_head(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = self._fixture(root)
            teacher = torch.load(
                inputs.teacher_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            teacher["head.1.weight"] = teacher["head.1.weight"][:-1]
            torch.save(teacher, inputs.teacher_checkpoint)

            with self.assertRaisesRegex(ValueError, "head.1.weight"):
                stage_release(inputs, root / "release")
