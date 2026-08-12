from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from maest_infer import get_maest

from tools.maest522.checkpoint import expand_classifier_state_dict
from tools.maest522.native_model import get_maest_522, predict_522


class NativeModelTests(TestCase):
    def test_strict_reload_returns_identical_522_logits(self) -> None:
        teacher_path = Path(
            r"E:\Projects\music-genre-tagger\src\models\discogs-maest-30s-pw-129e-519l-swa.ckpt"
        )
        if not teacher_path.is_file():
            self.skipTest("local MAEST checkpoint is unavailable")
        teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
        expanded = expand_classifier_state_dict(teacher, seed=522)
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "native.ckpt"
            torch.save(
                {
                    "state_dict": expanded,
                    "metadata": {"architecture": "maest_522l_pytorch"},
                },
                checkpoint_path,
            )
            original = get_maest(
                "discogs-maest-30s-pw-129e-519l",
                pretrained=False,
            )
            original.head[1] = torch.nn.Linear(768, 522)
            original.head_dist = torch.nn.Linear(768, 522)
            original.labels = [str(index) for index in range(522)]
            original.load_state_dict(expanded, strict=True)
            original.eval()
            reloaded = get_maest_522(checkpoint_path)
            mel = torch.randn(1, 1, 96, 1875, generator=torch.Generator().manual_seed(7))

            with torch.no_grad():
                expected_logits = original(
                    mel,
                    melspectrogram_input=True,
                )[0]
                actual_logits = reloaded(
                    mel,
                    melspectrogram_input=True,
                )[0]
            torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
            torch.testing.assert_close(
                predict_522(reloaded, mel),
                torch.sigmoid(expected_logits),
                rtol=0,
                atol=0,
            )
