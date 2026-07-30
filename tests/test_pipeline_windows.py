import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline import select_audio_windows


class SelectAudioWindowsTests(unittest.TestCase):
    def test_selects_duration_aware_window_starts(self):
        cases = [
            (20, [0.0]),
            (30, [0.0]),
            (40, [0.0, 5.0, 10.0]),
            (60, [0.0, 15.0, 30.0]),
            (180, [21.0, 75.0, 129.0]),
            (300, [45.0, 135.0, 225.0]),
        ]
        sample_rate = 10
        for duration, expected_starts in cases:
            with self.subTest(duration=duration):
                audio = np.arange(duration * sample_rate, dtype=np.float32)
                windows = select_audio_windows(
                    audio, sample_rate, 30.0, (0.2, 0.5, 0.8)
                )
                self.assertEqual([offset for offset, _ in windows], expected_starts)

    def test_slices_using_sample_boundaries(self):
        sample_rate = 10
        audio = np.arange(60 * sample_rate, dtype=np.float32)
        windows = select_audio_windows(audio, sample_rate, 30.0, (0.2, 0.5, 0.8))
        self.assertEqual([window[0] for _, window in windows], [0.0, 150.0, 300.0])
        self.assertTrue(all(len(window) == 300 for _, window in windows))

    def test_rejects_empty_audio(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            select_audio_windows(np.array([], dtype=np.float32), 16000, 30.0, (0.2, 0.5, 0.8))
