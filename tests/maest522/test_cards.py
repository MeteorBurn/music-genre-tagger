from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.maest522.cards import render_release_cards


def complete_metadata() -> dict[str, object]:
    return {
        "version": "0.1.0",
        "source_checkpoint": "discogs-maest-30s-pw-129e-519l-swa.ckpt",
        "source_sha256": "a" * 64,
        "license_review_status": "pending exact checkpoint grant review",
        "evaluation": {
            "macro_average_precision": 0.81,
            "macro_f1": 0.74,
            "legacy_gates_passed": True,
        },
        "parity": {"windows": 30, "tracks": 10, "passed": True},
        "dataset_summary": {
            "rows_by_split": {"train": 700, "val": 150, "test": 150},
            "label_state_counts": {"positive": 500, "negative": 400, "uncertain": 100},
        },
        "split_audit_sha256": "b" * 64,
        "annotation_protocol": "Three independent per-label states with blind holdouts.",
        "intended_use": "Music-library tagging and research on adjacent minimal-house styles.",
        "limitations": "Adjacent minimal-house styles overlap and human judgments are subjective.",
        "audio_rights": "No source audio is redistributed.",
    }


class CardGenerationTests(TestCase):
    def test_renders_complete_deterministic_model_and_dataset_cards(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            report = render_release_cards(complete_metadata(), output_dir)

            model_card = report.model_card.read_text(encoding="utf-8")
            dataset_card = report.dataset_card.read_text(encoding="utf-8")
            self.assertIn("license: other", model_card)
            self.assertIn("mtg-upf/discogs-maest-30s-pw-129e-519l", model_card)
            self.assertIn("Electronic---Microhouse", model_card)
            self.assertIn("519", model_card)
            self.assertIn("hard negatives", dataset_card.lower())
            self.assertIn("uncertain", dataset_card)
            self.assertIn("No source audio", dataset_card)

            first_model = model_card
            render_release_cards(complete_metadata(), output_dir)
            self.assertEqual(
                report.model_card.read_text(encoding="utf-8"),
                first_model,
            )

    def test_rejects_missing_provenance_license_evaluation_or_annotation(self) -> None:
        required = (
            "source_sha256",
            "license_review_status",
            "evaluation",
            "intended_use",
            "limitations",
            "annotation_protocol",
        )
        for key in required:
            metadata = complete_metadata()
            del metadata[key]
            with TemporaryDirectory() as temp_dir:
                with self.assertRaisesRegex(ValueError, key):
                    render_release_cards(metadata, Path(temp_dir))
