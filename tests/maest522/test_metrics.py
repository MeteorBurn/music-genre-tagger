from unittest import TestCase

import numpy

from tools.maest522.metrics import (
    binary_classification_metrics,
    expected_calibration_error,
    select_label_threshold,
)


class ClassificationMetricsTests(TestCase):
    def test_binary_metrics_return_exact_confusion_and_rates(self) -> None:
        targets = numpy.array([1, 1, 0, 0], dtype=numpy.float64)
        probabilities = numpy.array([0.9, 0.4, 0.6, 0.1])

        metrics = binary_classification_metrics(targets, probabilities, 0.5)

        self.assertEqual(metrics.true_positive, 1)
        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.true_negative, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.f1, 0.5)
        self.assertAlmostEqual(metrics.average_precision, 5.0 / 6.0)

    def test_threshold_selection_maximizes_f1_subject_to_precision_floor(self) -> None:
        targets = numpy.array([1, 1, 0, 0])
        probabilities = numpy.array([0.9, 0.6, 0.7, 0.1])

        threshold = select_label_threshold(
            targets,
            probabilities,
            minimum_precision=0.8,
        )
        metrics = binary_classification_metrics(targets, probabilities, threshold)

        self.assertEqual(threshold, 0.9)
        self.assertGreaterEqual(metrics.precision, 0.8)
        self.assertAlmostEqual(metrics.f1, 2.0 / 3.0)

    def test_expected_calibration_error_uses_probability_bins(self) -> None:
        targets = numpy.array([0, 0, 1, 1])
        probabilities = numpy.array([0.1, 0.2, 0.8, 0.9])

        ece = expected_calibration_error(targets, probabilities, bin_count=2)

        self.assertAlmostEqual(ece, 0.15)

    def test_metrics_reject_unsupervised_or_degenerate_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "both positive and negative"):
            select_label_threshold(
                numpy.ones(3),
                numpy.array([0.1, 0.2, 0.3]),
            )
        with self.assertRaisesRegex(ValueError, "same shape"):
            binary_classification_metrics(
                numpy.ones(2),
                numpy.ones(3),
                0.5,
            )
