from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy
import torch
from maest_infer.helpers.melspectrogram import MelSpectrogram
from transformers import AutoFeatureExtractor

from tools.maest522.feature_extraction_maest import MAESTFeatureExtractor
from tools.maest522.hf_feature_extractor import write_feature_extractor_release


class HuggingFaceFeatureExtractorTests(TestCase):
    def test_waveform_to_mel_matches_installed_native_frontend(self) -> None:
        waveform = torch.randn(48_000, generator=torch.Generator().manual_seed(522))
        native = MelSpectrogram()(waveform).transpose(0, 1)
        extractor = MAESTFeatureExtractor(max_length=400)

        batch = extractor(
            waveform.numpy(),
            sampling_rate=16_000,
            return_tensors="pt",
        )

        actual = batch["input_values"][0, : native.shape[0]]
        torch.testing.assert_close(actual, native, rtol=0, atol=0)
        torch.testing.assert_close(
            batch["input_values"][0, native.shape[0] :],
            torch.zeros(400 - native.shape[0], 96),
        )

    def test_rejects_wrong_sample_rate_or_multichannel_shape(self) -> None:
        extractor = MAESTFeatureExtractor(max_length=100)
        with self.assertRaisesRegex(ValueError, "16000"):
            extractor(numpy.zeros(1600), sampling_rate=44_100)
        with self.assertRaisesRegex(ValueError, "mono"):
            extractor(numpy.zeros((2, 1600)), sampling_rate=16_000)

    def test_writes_remote_code_and_preprocessor_contract(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            report = write_feature_extractor_release(output_dir)

            self.assertTrue((output_dir / "feature_extraction_maest.py").is_file())
            self.assertTrue((output_dir / "preprocessor_config.json").is_file())
            self.assertIn("Apache-2.0", (output_dir / "feature_extraction_maest.py").read_text(encoding="utf-8"))
            self.assertEqual(report["feature_extractor_type"], "MAESTFeatureExtractor")
            reloaded = AutoFeatureExtractor.from_pretrained(
                output_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.assertEqual(type(reloaded).__name__, "MAESTFeatureExtractor")
            self.assertEqual(reloaded.sampling_rate, 16_000)
