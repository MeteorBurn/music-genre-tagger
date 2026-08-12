from pathlib import Path
from unittest import TestCase

from tools.maest522.constants import NEW_LABELS


class AnnotationUiAssetTests(TestCase):
    def test_contains_single_label_controls_and_trusted_playlist_workflow(self) -> None:
        static_dir = Path("tools/maest522/static")
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        javascript = (static_dir / "app.js").read_text(encoding="utf-8")
        styles = (static_dir / "styles.css").read_text(encoding="utf-8")
        combined = html + javascript

        for label in NEW_LABELS:
            self.assertIn(label, combined)
        for keyboard_code in (
            "Space",
            "KeyP",
            "KeyN",
            "KeyU",
            "KeyJ",
            "KeyK",
            "KeyL",
            "Enter",
            "Backspace",
        ):
            self.assertIn(keyboard_code, javascript)
        for obsolete_code in ("Digit1", "Digit2", "Digit3", "KeyX"):
            self.assertNotIn(obsolete_code, javascript)
        self.assertIn("active-label", html)
        self.assertIn("positive-target", html)
        self.assertIn("negative-target", html)
        self.assertIn("confirmed-progress", html)
        self.assertIn("trusted-playlist-path", html)
        self.assertIn("trusted-playlist-state", html)
        self.assertIn("trusted-preflight", html)
        self.assertIn("trusted-commit", html)
        self.assertIn("confirmed-batches", html)
        self.assertIn("/trusted-playlists/preflight", javascript)
        self.assertIn("/trusted-playlists/commit", javascript)
        self.assertIn("/confirmed-progress", javascript)
        self.assertIn("/confirmed-batches/", javascript)
        self.assertIn("/goals/", javascript)
        self.assertIn("export-manifest", html)
        self.assertIn("/export", javascript)
        self.assertIn("audio", html)
        self.assertIn("0.2", javascript)
        self.assertIn("0.5", javascript)
        self.assertIn("0.8", javascript)
        self.assertNotIn("teacher_score", combined)
        self.assertNotIn("student_score", combined)
        self.assertNotIn("Для каждого трека укажите состояние всех трёх стилей", html)
        self.assertIn("--accent", styles)
