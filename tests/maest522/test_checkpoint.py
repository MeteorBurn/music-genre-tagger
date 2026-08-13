import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from tools.maest522.checkpoint import (
    CLASSIFIER_HEAD_SHAPES_519,
    convert_checkpoint,
    expand_classifier_state_dict,
)


class CheckpointExpansionTests(TestCase):
    def _state_dict(self) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(7)
        return {
            "backbone.weight": torch.randn(4, 4, generator=generator),
            "head.1.weight": torch.randn(519, 768, generator=generator),
            "head.1.bias": torch.randn(519, generator=generator),
            "head_dist.weight": torch.randn(519, 768, generator=generator),
            "head_dist.bias": torch.randn(519, generator=generator),
        }

    def test_expansion_preserves_every_legacy_value_and_is_deterministic(self) -> None:
        source = self._state_dict()

        first = expand_classifier_state_dict(source, seed=129, prior_probability=0.01)
        second = expand_classifier_state_dict(source, seed=129, prior_probability=0.01)

        for key, tensor in source.items():
            legacy_rows = 519 if key in CLASSIFIER_HEAD_SHAPES_519 else None
            if legacy_rows is None:
                torch.testing.assert_close(first[key], tensor, rtol=0, atol=0)
            else:
                torch.testing.assert_close(
                    first[key][:legacy_rows], tensor, rtol=0, atol=0
                )
            self.assertNotEqual(first[key].data_ptr(), tensor.data_ptr())
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

        self.assertEqual(tuple(first["head.1.weight"].shape), (522, 768))
        self.assertEqual(tuple(first["head.1.bias"].shape), (522,))
        self.assertEqual(tuple(first["head_dist.weight"].shape), (522, 768))
        self.assertEqual(tuple(first["head_dist.bias"].shape), (522,))
        expected_bias = math.log(0.01 / 0.99)
        torch.testing.assert_close(
            first["head.1.bias"][519:],
            torch.full((3,), expected_bias),
        )
        torch.testing.assert_close(
            first["head_dist.bias"][519:],
            torch.full((3,), expected_bias),
        )
        torch.testing.assert_close(
            source["head.1.weight"], self._state_dict()["head.1.weight"]
        )

    def test_rejects_checkpoint_with_wrong_classifier_shape(self) -> None:
        source = self._state_dict()
        source["head.1.weight"] = source["head.1.weight"][:-1]

        with self.assertRaisesRegex(ValueError, "head.1.weight"):
            expand_classifier_state_dict(source)

    def test_conversion_writes_atomic_artifacts_and_hash_report(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "source.ckpt"
            output_dir = root / "expanded"
            torch.save(self._state_dict(), source_path)

            report = convert_checkpoint(
                source_path,
                output_dir,
                seed=129,
                prior_probability=0.01,
            )

            checkpoint_path = output_dir / "expanded-init.ckpt"
            labels_path = output_dir / "labels-522.txt"
            report_path = output_dir / "expansion-report.json"
            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue(labels_path.is_file())
            self.assertTrue(report_path.is_file())
            labels = labels_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(labels), 522)
            self.assertEqual(
                labels[-3:],
                [
                    "Electronic---Minimal-Deep-Tech",
                    "Electronic---Microhouse",
                    "Electronic---RoMinimal",
                ],
            )
            self.assertEqual(labels[-3:], list(report["new_labels"]))
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted, report)
            self.assertEqual(len(report["source_sha256"]), 64)
            self.assertEqual(len(report["output_sha256"]), 64)
            self.assertNotIn(str(root), json.dumps(report))

            expanded = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            torch.testing.assert_close(
                expanded["head.1.weight"][:519],
                self._state_dict()["head.1.weight"],
                rtol=0,
                atol=0,
            )
