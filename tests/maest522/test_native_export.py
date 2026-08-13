import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from tools.maest522.native_export import (
    export_native_release,
    merge_native_state_dict,
)


class NativeExportTests(TestCase):
    def _teacher(self) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(519)
        return {
            "backbone.weight": torch.randn(2, 2, generator=generator),
            "head.0.weight": torch.randn(768, generator=generator),
            "head.0.bias": torch.randn(768, generator=generator),
            "head.1.weight": torch.randn(519, 768, generator=generator),
            "head.1.bias": torch.randn(519, generator=generator),
            "head_dist.weight": torch.randn(519, 768, generator=generator),
            "head_dist.bias": torch.randn(519, generator=generator),
        }

    def _student(self) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(522)
        return {
            "backbone.backbone.weight": torch.randn(2, 2, generator=generator),
            "head_norm.weight": torch.randn(768, generator=generator),
            "head_norm.bias": torch.randn(768, generator=generator),
            "legacy_head.weight": torch.randn(519, 768, generator=generator),
            "legacy_head.bias": torch.randn(519, generator=generator),
            "extension_head.weight": torch.randn(3, 768, generator=generator),
            "extension_head.bias": torch.randn(3, generator=generator),
            "extension_dist_head.weight": torch.randn(3, 768, generator=generator),
            "extension_dist_head.bias": torch.randn(3, generator=generator),
            "legacy_dist_weight": torch.randn(519, 768, generator=generator),
            "legacy_dist_bias": torch.randn(519, generator=generator),
        }

    def test_merge_uses_teacher_legacy_rows_and_trained_backbone_extension(self) -> None:
        teacher = self._teacher()
        student = self._student()

        merged = merge_native_state_dict(teacher, student)

        for prefix in ("head.1", "head_dist"):
            torch.testing.assert_close(
                merged[f"{prefix}.weight"][:519],
                teacher[f"{prefix}.weight"],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                merged[f"{prefix}.bias"][:519],
                teacher[f"{prefix}.bias"],
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            merged["head.1.weight"][519:],
            student["extension_head.weight"],
        )
        torch.testing.assert_close(
            merged["head_dist.weight"][519:],
            student["extension_dist_head.weight"],
        )
        torch.testing.assert_close(
            merged["backbone.weight"],
            student["backbone.backbone.weight"],
        )
        torch.testing.assert_close(merged["head.0.weight"], student["head_norm.weight"])
        self.assertEqual(tuple(merged["head.1.weight"].shape), (522, 768))

    def test_export_writes_metadata_labels_and_digest_report(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher_path = root / "teacher.ckpt"
            trained_path = root / "trained.ckpt"
            output_path = root / "release" / "native-maest-522.ckpt"
            labels_path = root / "release" / "labels-522.txt"
            torch.save(self._teacher(), teacher_path)
            torch.save(
                {
                    "model_state_dict": self._student(),
                    "input_digests": {"annotation": "a" * 64},
                },
                trained_path,
            )
            evaluation_path = root / "evaluation.json"
            evaluation_path.write_text(
                json.dumps({"gates_passed": True}),
                encoding="utf-8",
            )

            report = export_native_release(
                teacher_path,
                trained_path,
                output_path,
                labels_path,
                evaluation_path=evaluation_path,
            )

            artifact = torch.load(output_path, map_location="cpu", weights_only=True)
            self.assertEqual(artifact["metadata"]["architecture"], "maest_522l_pytorch")
            self.assertEqual(tuple(artifact["state_dict"]["head.1.weight"].shape), (522, 768))
            self.assertEqual(len(labels_path.read_text(encoding="utf-8").splitlines()), 522)
            self.assertEqual(len(report.output_sha256), 64)
            self.assertEqual(len(report.label_sha256), 64)

    def test_rejects_invalid_teacher_or_missing_student_rows(self) -> None:
        teacher = self._teacher()
        teacher["head.1.weight"] = teacher["head.1.weight"][:-1]
        with self.assertRaisesRegex(ValueError, "head.1.weight"):
            merge_native_state_dict(teacher, self._student())
        student = self._student()
        del student["extension_head.weight"]
        with self.assertRaisesRegex(ValueError, "extension_head.weight"):
            merge_native_state_dict(self._teacher(), student)
