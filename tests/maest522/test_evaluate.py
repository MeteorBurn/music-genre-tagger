import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy

from tools.maest522.evaluate import (
    EvaluationAlreadyRunError,
    EvaluationPermissionError,
    aggregate_track_windows,
    evaluate_legacy_regression,
    evaluate_new_labels,
    record_evaluation,
)


class EvaluationTests(TestCase):
    def test_teacher_self_comparison_is_zero_drift_parity(self) -> None:
        generator = numpy.random.default_rng(522)
        logits = generator.normal(size=(30, 519))
        embeddings = generator.normal(size=(30, 768))

        report = evaluate_legacy_regression(
            logits,
            logits.copy(),
            embeddings,
            embeddings.copy(),
        )

        self.assertAlmostEqual(report.mean_probability_drift, 0.0)
        self.assertAlmostEqual(report.mean_top10_overlap, 1.0)
        self.assertAlmostEqual(report.teacher_top3_in_student_top5, 1.0)
        self.assertAlmostEqual(report.embedding_cosine, 1.0)
        self.assertTrue(report.gates_passed)

    def test_regression_gates_detect_old_output_and_embedding_drift(self) -> None:
        teacher = numpy.zeros((30, 519))
        student = teacher.copy()
        student[:, :20] = 2.0
        teacher_embeddings = numpy.ones((30, 4))
        student_embeddings = -teacher_embeddings

        report = evaluate_legacy_regression(
            teacher,
            student,
            teacher_embeddings,
            student_embeddings,
        )

        self.assertGreater(report.mean_probability_drift, 0.01)
        self.assertLess(report.embedding_cosine, 0.98)
        self.assertFalse(report.gates_passed)

    def test_track_aggregation_means_windows_and_publishes_variance(self) -> None:
        track_ids = numpy.array(["a", "a", "b"])
        group_ids = numpy.array(["ga", "ga", "gb"])
        probabilities = numpy.array([[0.2, 0.4], [0.8, 0.6], [0.1, 0.9]])
        targets = numpy.array([[1, 0], [1, 0], [0, 1]])
        mask = numpy.ones_like(targets)

        aggregated = aggregate_track_windows(
            track_ids,
            group_ids,
            probabilities,
            targets,
            mask,
        )

        self.assertEqual(aggregated.track_ids.tolist(), ["a", "b"])
        numpy.testing.assert_allclose(aggregated.probabilities[0], [0.5, 0.5])
        numpy.testing.assert_allclose(aggregated.window_variance[0], [0.09, 0.01])
        self.assertEqual(aggregated.group_ids.tolist(), ["ga", "gb"])

    def test_new_label_metrics_honor_uncertainty_and_hard_negative_subset(self) -> None:
        targets = numpy.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        mask = numpy.ones_like(targets)
        mask[0, 2] = 0
        probabilities = targets * 0.8 + 0.1
        hard_negative_mask = numpy.zeros_like(targets, dtype=bool)
        hard_negative_mask[1, 0] = True

        report = evaluate_new_labels(
            targets,
            mask,
            probabilities,
            minimum_precision=0.8,
            hard_negative_mask=hard_negative_mask,
        )

        self.assertEqual(len(report.thresholds), 3)
        self.assertAlmostEqual(report.macro_average_precision, 1.0)
        self.assertAlmostEqual(report.macro_f1, 1.0)
        self.assertEqual(report.hard_negative_counts[0], 1)

    def test_test_evaluation_requires_permission_and_is_recorded_once(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "run-manifest.json").write_text(
                json.dumps({"format_version": 1, "test_evaluated": False}),
                encoding="utf-8",
            )
            report = {"split": "test", "macro_f1": 0.7}

            with self.assertRaises(EvaluationPermissionError):
                record_evaluation(run_dir, "test", report, allow_test=False)

            record_evaluation(run_dir, "test", report, allow_test=True)
            manifest = json.loads(
                (run_dir / "run-manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["test_evaluated"])
            self.assertTrue((run_dir / "evaluation.json").is_file())
            self.assertTrue((run_dir / "metrics.csv").is_file())
            self.assertTrue((run_dir / "evaluation.md").is_file())

            with self.assertRaises(EvaluationAlreadyRunError):
                record_evaluation(run_dir, "test", report, allow_test=True)
