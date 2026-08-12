import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy
import torch
from torch import nn

from tools.maest522.active_learning import (
    AcquisitionCandidate,
    export_acquisition_scores,
    rank_training_candidates,
)
from tools.maest522.cache import CacheKey, TeacherCache, TeacherRecord
from tools.maest522.checkpoint import expand_classifier_state_dict
from tools.maest522.continual_model import ContinualMaest522, TrainingStage
from tools.maest522.metrics import select_label_threshold
from tools.maest522.train import (
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)


class TinyMaest(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(768, 768) for _ in range(12)])
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(nn.Identity(), nn.Linear(768, 519))
        self.head_dist = nn.Linear(768, 519)

    def forward_features(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return values, values * 0.5


class TrainingWorkflowTests(TestCase):
    def test_expansion_cache_stage1_resume_threshold_and_acquisition(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generator = torch.Generator().manual_seed(522)
            source = {
                "head.1.weight": torch.randn(519, 768, generator=generator),
                "head.1.bias": torch.randn(519, generator=generator),
                "head_dist.weight": torch.randn(519, 768, generator=generator),
                "head_dist.bias": torch.randn(519, generator=generator),
            }
            expanded = expand_classifier_state_dict(source, seed=522)
            model = ContinualMaest522(TinyMaest(), expanded)
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=1e-3,
            )

            cache = TeacherCache(root / "cache")
            key = CacheKey("track-a", 5000, "a" * 64, "mel-v1")
            teacher = TeacherRecord(
                legacy_logits=torch.randn(519, generator=generator),
                pooled_embedding=torch.randn(768, generator=generator),
            )
            cache.put_atomic(key, teacher)
            restored_teacher = cache.get(key)
            self.assertIsNotNone(restored_teacher)

            batch = torch.randn(2, 768, generator=generator)
            targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            before = model.extension_head.weight.detach().clone()
            output = model(batch)
            loss = nn.functional.binary_cross_entropy_with_logits(
                output.new_logits,
                targets,
            )
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.extension_head.weight))
            self.assertIsNone(model.legacy_head.weight.grad)

            progress = TrainingProgress(
                TrainingStage.EXTENSION_HEADS,
                epoch=1,
                global_step=1,
                best_metrics={"macro_average_precision": 0.5},
            )
            checkpoint_path = root / "last.ckpt"
            digests = {"teacher": "a", "annotation": "b", "replay": "c"}
            save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                progress,
                digests,
            )
            resumed = load_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                digests,
            )
            self.assertEqual(resumed, progress)

            threshold = select_label_threshold(
                numpy.array([1, 1, 0, 0]),
                numpy.array([0.9, 0.8, 0.4, 0.1]),
                minimum_precision=0.8,
            )
            self.assertGreaterEqual(threshold, 0.4)

            acquisition = rank_training_candidates(
                [
                    AcquisitionCandidate(
                        track_id="train-a",
                        group_id="group-a",
                        split="train",
                        reviewed=False,
                        label_credit_counts=(0, 0, 0),
                        probabilities=numpy.array([[0.5, 0.4, 0.6]]),
                        hard_negative_score=0.5,
                        embedding=numpy.array([1.0, 0.0]),
                    )
                ],
                reviewed_group_ids=set(),
                limit=1,
            )
            scores_path = root / "acquisition.json"
            export_acquisition_scores(acquisition, scores_path)
            exported = json.loads(scores_path.read_text(encoding="utf-8"))
            self.assertEqual(set(exported), {"train-a"})
