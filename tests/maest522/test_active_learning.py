import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy

from tools.maest522.active_learning import (
    AcquisitionCandidate,
    export_acquisition_scores,
    rank_training_candidates,
)


class ActiveLearningTests(TestCase):
    def _candidates(self) -> list[AcquisitionCandidate]:
        return [
            AcquisitionCandidate(
                track_id="train-a",
                group_id="group-a",
                split="train",
                reviewed=False,
                label_credit_counts=(10, 10, 10),
                probabilities=numpy.array([[0.49, 0.51, 0.1], [0.7, 0.3, 0.2]]),
                hard_negative_score=0.8,
                embedding=numpy.array([1.0, 0.0]),
            ),
            AcquisitionCandidate(
                track_id="train-b",
                group_id="group-b",
                split="train",
                reviewed=False,
                label_credit_counts=(10, 10, 10),
                probabilities=numpy.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
                hard_negative_score=0.1,
                embedding=numpy.array([0.0, 1.0]),
            ),
            AcquisitionCandidate(
                track_id="reviewed",
                group_id="group-c",
                split="train",
                reviewed=True,
                label_credit_counts=(0, 0, 0),
                probabilities=numpy.full((2, 3), 0.5),
                hard_negative_score=1.0,
                embedding=numpy.array([1.0, 1.0]),
            ),
            AcquisitionCandidate(
                track_id="validation",
                group_id="group-d",
                split="val",
                reviewed=False,
                label_credit_counts=(0, 0, 0),
                probabilities=numpy.full((2, 3), 0.5),
                hard_negative_score=1.0,
                embedding=numpy.array([-1.0, 0.0]),
            ),
            AcquisitionCandidate(
                track_id="sibling",
                group_id="reviewed-group",
                split="train",
                reviewed=False,
                label_credit_counts=(0, 0, 0),
                probabilities=numpy.full((2, 3), 0.5),
                hard_negative_score=1.0,
                embedding=numpy.array([0.0, -1.0]),
            ),
            AcquisitionCandidate(
                track_id="capped",
                group_id="group-e",
                split="train",
                reviewed=False,
                label_credit_counts=(1000, 1000, 1000),
                probabilities=numpy.full((2, 3), 0.5),
                hard_negative_score=1.0,
                embedding=numpy.array([-1.0, -1.0]),
            ),
        ]

    def test_excludes_reviewed_nontrain_siblings_and_fully_capped_tracks(self) -> None:
        ranked = rank_training_candidates(
            self._candidates(),
            reviewed_group_ids={"reviewed-group"},
            limit=10,
            seed=522,
        )

        self.assertEqual(set(ranked.scores), {"train-a", "train-b"})
        self.assertEqual(ranked.excluded_counts["reviewed"], 1)
        self.assertEqual(ranked.excluded_counts["non_train"], 1)
        self.assertEqual(ranked.excluded_counts["reviewed_group_sibling"], 1)
        self.assertEqual(ranked.excluded_counts["label_cap"], 1)
        self.assertGreaterEqual(ranked.scores["train-a"], 0.0)
        self.assertLessEqual(ranked.scores["train-a"], 1.0)

    def test_ranking_is_reproducible_and_export_contains_only_ids_and_scores(self) -> None:
        candidates = self._candidates()[:2]
        first = rank_training_candidates(candidates, set(), limit=2, seed=522)
        second = rank_training_candidates(candidates, set(), limit=2, seed=522)
        self.assertEqual(first.scores, second.scores)

        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "scores.json"
            diagnostic = export_acquisition_scores(first, output)
            payload = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(set(payload), set(first.scores))
            self.assertTrue(all(isinstance(value, float) for value in payload.values()))
            self.assertNotIn("probabilities", json.dumps(payload))
            self.assertTrue(diagnostic.is_file())
