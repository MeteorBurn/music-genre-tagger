import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline import (
    aggregate_window_predictions,
    analyze_audio_file,
    process_predictions,
)


class AggregateWindowPredictionsTests(unittest.TestCase):
    def test_averages_scores_before_top_n_selection(self):
        predictions = [
            (np.array([0.9, 0.2, 0.2, 0.0]), ["A", "B", "C", "D"]),
            (np.array([0.0, 0.8, 0.2, 0.1]), ["A", "B", "C", "D"]),
            (np.array([0.0, 0.2, 0.9, 0.2]), ["A", "B", "C", "D"]),
        ]

        scores, labels = aggregate_window_predictions(predictions)

        np.testing.assert_allclose(scores, [0.3, 0.4, 0.43333333, 0.1])
        self.assertEqual(labels, ["A", "B", "C", "D"])
        self.assertEqual(
            process_predictions(scores, labels, 3),
            [("C", scores[2]), ("B", scores[1]), ("A", scores[0])],
        )

    def test_rejects_changed_label_order(self):
        predictions = [
            (np.array([0.2, 0.8]), ["A", "B"]),
            (np.array([0.8, 0.2]), ["B", "A"]),
        ]

        with self.assertRaisesRegex(ValueError, "label vocabulary"):
            aggregate_window_predictions(predictions)

    def test_rejects_non_vector_scores(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            aggregate_window_predictions([(np.array([[0.2, 0.8]]), ["A", "B"])])


class AnalyzeAudioFileWindowInferenceTests(unittest.TestCase):
    def test_returns_top_three_from_mean_of_three_audio_windows(self):
        class FakeMaest:
            def __init__(self):
                self._predictions = iter(
                    [
                        np.array([0.9, 0.2, 0.2, 0.0]),
                        np.array([0.0, 0.8, 0.2, 0.1]),
                        np.array([0.0, 0.2, 0.9, 0.2]),
                    ]
                )

            def predict_labels(self, wav):
                return next(self._predictions), [
                    "genre---A",
                    "genre---B",
                    "genre---C",
                    "genre---D",
                ]

        config = {
            "convert_to_wav": False,
            "sample_rate": 10,
            "ffmpeg_path": "ffmpeg",
            "maest_result_key": "maest_519l_pytorch",
            "audio_window_duration": 30.0,
            "audio_window_positions": (0.2, 0.5, 0.8),
            "num_genres": 3,
        }
        models = {"maest": {"fake": {"model": FakeMaest(), "device": "cpu"}}}
        audio = np.arange(600, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "track.wav"
            audio_path.touch()
            with patch("pipeline.load_mono_16k", return_value=audio):
                result = analyze_audio_file(audio_path, models, config)

        self.assertNotIn("error", result)
        self.assertEqual(result["genres"], {
            "labels": ["C", "B", "A"],
            "confidences": [0.4333, 0.4, 0.3],
            "model": "fake",
        })
        self.assertEqual(result["analysis_config"], {
            "audio_segment_offsets": [0.0, 15.0, 30.0],
            "audio_segment_duration": 30.0,
            "audio_segment_count": 3,
            "aggregation": "mean",
        })
