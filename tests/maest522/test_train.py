import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from torch import nn

from tools.maest522.continual_model import TrainingStage
from tools.maest522.train import (
    EarlyStopping,
    TrainingConfig,
    TrainingProgress,
    build_optimizer,
    load_training_checkpoint,
    save_training_checkpoint,
    select_better_validation_result,
    train_accumulated_batches,
    run_training_stage,
)


class TinyStagedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extension_head = nn.Linear(2, 1, bias=False)
        self.extension_dist_head = nn.Linear(2, 1, bias=False)
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(12)])
        self.backbone.norm = nn.LayerNorm(2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.extension_head(values)


class TrainerTests(TestCase):
    def _config(self, root: Path) -> TrainingConfig:
        teacher = root / "teacher.ckpt"
        annotation = root / "annotation.jsonl"
        replay = root / "replay.jsonl"
        teacher.write_bytes(b"teacher")
        annotation.write_bytes(b"annotation")
        replay.write_bytes(b"replay")
        return TrainingConfig(
            teacher_checkpoint=teacher,
            annotation_manifest=annotation,
            replay_manifest=replay,
            output_dir=root / "run",
            batch_size=2,
            accumulation_steps=2,
            head_learning_rate=1e-2,
            backbone_learning_rate=1e-3,
            max_epochs_per_stage=3,
            patience=2,
        )

    def test_optimizer_groups_follow_named_stage_policy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = self._config(Path(temp_dir))
            model = TinyStagedModel()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for parameter in model.extension_head.parameters():
                parameter.requires_grad_(True)
            for parameter in model.extension_dist_head.parameters():
                parameter.requires_grad_(True)

            stage_one = build_optimizer(model, config, TrainingStage.EXTENSION_HEADS)

            self.assertEqual(len(stage_one.param_groups), 1)
            self.assertEqual(stage_one.param_groups[0]["name"], "extension_heads")
            self.assertEqual(stage_one.param_groups[0]["lr"], 1e-2)

            for block_index in (10, 11):
                for parameter in model.backbone.blocks[block_index].parameters():
                    parameter.requires_grad_(True)
            for parameter in model.backbone.norm.parameters():
                parameter.requires_grad_(True)
            stage_two = build_optimizer(model, config, TrainingStage.BLOCKS_10_11)

            self.assertEqual(
                [group["name"] for group in stage_two.param_groups],
                ["extension_heads", "backbone"],
            )
            self.assertEqual(stage_two.param_groups[1]["lr"], 1e-3)

    def test_gradient_accumulation_steps_once_per_two_microbatches(self) -> None:
        torch.manual_seed(522)
        model = TinyStagedModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        batches = [
            (torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0]])),
            (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0]])),
            (torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0]])),
        ]

        result = train_accumulated_batches(
            model,
            batches,
            optimizer,
            loss_function=lambda output, target: nn.functional.mse_loss(output, target),
            accumulation_steps=2,
            device=torch.device("cpu"),
            max_gradient_norm=1.0,
        )

        self.assertEqual(result.microbatches, 3)
        self.assertEqual(result.optimizer_steps, 2)
        self.assertGreater(result.mean_loss, 0.0)

    def test_checkpoint_resume_restores_model_optimizer_progress_and_rng(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._config(root)
            model = TinyStagedModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            progress = TrainingProgress(
                stage=TrainingStage.EXTENSION_HEADS,
                epoch=2,
                global_step=7,
                best_metrics={"macro_average_precision": 0.75},
            )
            torch.manual_seed(99)
            checkpoint_path = config.output_dir / "last.ckpt"
            expected_next_random = None

            save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                progress,
                config.input_digests(),
            )
            expected_next_random = torch.rand(3)
            with torch.no_grad():
                model.extension_head.weight.add_(10.0)
            torch.manual_seed(1234)

            restored = load_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config.input_digests(),
            )

            self.assertEqual(restored, progress)
            torch.testing.assert_close(torch.rand(3), expected_next_random)
            self.assertLess(model.extension_head.weight.abs().max().item(), 10.0)

            annotation = config.annotation_manifest
            annotation.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest"):
                load_training_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    config.input_digests(),
                )

    def test_validation_selection_and_early_stopping_are_lexicographic(self) -> None:
        failing = {
            "regression_gates_passed": False,
            "macro_average_precision": 0.99,
            "macro_f1": 0.99,
            "legacy_probability_drift": 0.0,
        }
        passing = {
            "regression_gates_passed": True,
            "macro_average_precision": 0.70,
            "macro_f1": 0.60,
            "legacy_probability_drift": 0.01,
        }
        better_ap = dict(passing, macro_average_precision=0.71, macro_f1=0.1)
        better_f1 = dict(passing, macro_f1=0.61)
        lower_drift = dict(passing, legacy_probability_drift=0.005)

        self.assertTrue(select_better_validation_result(passing, failing))
        self.assertTrue(select_better_validation_result(better_ap, passing))
        self.assertTrue(select_better_validation_result(better_f1, passing))
        self.assertTrue(select_better_validation_result(lower_drift, passing))

        stopper = EarlyStopping(patience=2)
        self.assertFalse(stopper.update(True))
        self.assertFalse(stopper.update(False))
        self.assertTrue(stopper.update(False))
        self.assertEqual(stopper.epochs_without_improvement, 2)

    def test_resolved_config_is_serializable_without_absolute_digest_ambiguity(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = self._config(Path(temp_dir))

            payload = config.to_dict()

            encoded = json.dumps(payload)
            self.assertIn("teacher_checkpoint", encoded)
            self.assertEqual(payload["seed"], 522)
            self.assertEqual(set(config.input_digests()), {"teacher", "annotation", "replay"})

    def test_stage_runner_writes_last_best_and_metrics_with_early_stop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._config(root)
            model = TinyStagedModel()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for module in (model.extension_head, model.extension_dist_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            optimizer = build_optimizer(model, config, TrainingStage.EXTENSION_HEADS)
            validation_results = iter(
                [
                    {
                        "regression_gates_passed": True,
                        "macro_average_precision": 0.7,
                        "macro_f1": 0.6,
                        "legacy_probability_drift": 0.01,
                    },
                    {
                        "regression_gates_passed": True,
                        "macro_average_precision": 0.69,
                        "macro_f1": 0.7,
                        "legacy_probability_drift": 0.005,
                    },
                    {
                        "regression_gates_passed": True,
                        "macro_average_precision": 0.68,
                        "macro_f1": 0.8,
                        "legacy_probability_drift": 0.001,
                    },
                ]
            )
            batches = [
                (torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0]])),
                (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0]])),
            ]

            progress = run_training_stage(
                model=model,
                batches_for_epoch=lambda epoch: list(batches),
                optimizer=optimizer,
                loss_function=lambda output, target: nn.functional.mse_loss(
                    output,
                    target,
                ),
                validate=lambda current: next(validation_results),
                config=config,
                stage=TrainingStage.EXTENSION_HEADS,
                input_digests=config.input_digests(),
                device=torch.device("cpu"),
            )

            self.assertEqual(progress.epoch, 3)
            self.assertEqual(progress.global_step, 3)
            self.assertEqual(progress.best_metrics["macro_average_precision"], 0.7)
            self.assertTrue((config.output_dir / "last.ckpt").is_file())
            self.assertTrue((config.output_dir / "best.ckpt").is_file())
            metric_rows = (config.output_dir / "metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(metric_rows), 3)
