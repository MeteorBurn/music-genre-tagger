from pathlib import Path
from unittest import TestCase

from tools.maest522.constants import NEW_LABELS


class AnnotationUiAssetTests(TestCase):
    def test_contains_review_controls_hotkeys_and_playlist_upload(self) -> None:
        static_dir = Path("tools/maest522/static")
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        javascript = (static_dir / "app.js").read_text(encoding="utf-8")
        styles = (static_dir / "styles.css").read_text(encoding="utf-8")
        combined = html + javascript

        for label in NEW_LABELS:
            self.assertIn(label, combined)
        for keyboard_code in (
            "Space",
            "Digit1",
            "Digit2",
            "Digit3",
            "KeyP",
            "KeyN",
            "KeyU",
            "KeyX",
            "KeyJ",
            "KeyK",
            "KeyL",
            "Enter",
            "Backspace",
        ):
            self.assertIn(keyboard_code, javascript)
        self.assertIn('accept=".m3u,.m3u8"', html)
        self.assertIn("base-directory", html)
        self.assertIn("export-manifest", html)
        self.assertIn("/export", javascript)
        self.assertIn("audio", html)
        self.assertIn("0.2", javascript)
        self.assertIn("0.5", javascript)
        self.assertIn("0.8", javascript)
        self.assertNotIn("teacher_score", combined)
        self.assertNotIn("student_score", combined)
        self.assertIn("--accent", styles)
