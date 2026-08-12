from unittest import TestCase

from maest_infer.discogs_labels import discogs_519labels

from tools.maest522.model_labels import (
    NEW_LABELS,
    build_522_labels,
    load_official_519_labels,
)


class ModelLabelsTests(TestCase):
    def test_appends_three_labels_without_reordering_official_labels(self) -> None:
        official = load_official_519_labels()

        labels = build_522_labels(official)

        self.assertEqual(official, tuple(discogs_519labels))
        self.assertEqual(len(official), 519)
        self.assertEqual(labels[:519], official)
        self.assertEqual(labels[519:], NEW_LABELS)
        self.assertEqual(len(labels), 522)
        self.assertEqual(len(set(labels)), 522)

    def test_rejects_invalid_legacy_label_contract(self) -> None:
        official = load_official_519_labels()

        with self.assertRaisesRegex(ValueError, "exactly 519"):
            build_522_labels(official[:-1])
        with self.assertRaisesRegex(ValueError, "unique"):
            build_522_labels((*official[:-1], official[0]))
        with self.assertRaisesRegex(ValueError, "already contains"):
            build_522_labels((*official[:-1], NEW_LABELS[0]))
