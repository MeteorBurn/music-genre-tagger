import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.confirmed_labels import update_label_goal
from tools.maest522.constants import NEW_LABELS
from tools.maest522.fingerprints import (
    FingerprintResult,
    calculate_fingerprint,
    fingerprint_project,
)


class FingerprintTests(TestCase):
    @patch("tools.maest522.fingerprints.subprocess.run")
    def test_invokes_fpcalc_without_shell_and_parses_json(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"duration": 123.4, "fingerprint": "1,2,3"}),
            stderr="",
        )
        audio_path = Path("D:/Music/test track.flac")
        fpcalc_path = Path("C:/Utils/tools/fpcalc.exe")

        result = calculate_fingerprint(audio_path, fpcalc_path)

        self.assertEqual(result.status, "available")
        self.assertEqual(result.fingerprint, "1,2,3")
        self.assertEqual(result.duration_seconds, 123.4)
        run_mock.assert_called_once_with(
            [str(fpcalc_path), "-json", str(audio_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )

    @patch(
        "tools.maest522.fingerprints.subprocess.run",
        side_effect=FileNotFoundError("fpcalc missing"),
    )
    def test_reports_missing_fpcalc_as_unavailable(self, _run_mock) -> None:
        result = calculate_fingerprint(
            Path("D:/Music/test.flac"),
            Path("C:/Utils/tools/fpcalc.exe"),
        )

        self.assertEqual(result.status, "unavailable")
        self.assertIsNone(result.fingerprint)
        self.assertIn("fpcalc missing", result.detail)

    @patch("tools.maest522.fingerprints.calculate_fingerprint")
    def test_persists_project_fingerprint_outcomes(self, calculate_mock) -> None:
        calculate_mock.side_effect = [
            FingerprintResult("available", "fingerprint-a", 180.0, ""),
            FingerprintResult("unavailable", None, None, "fpcalc unavailable"),
        ]
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = AnnotationStore(root / "annotations.db")
            store.initialize()
            project_id = store.create_project("fingerprint-test")
            now = datetime.now(timezone.utc).isoformat()
            with store.connection() as connection:
                for track_index in range(2):
                    cursor = connection.execute(
                        "INSERT INTO tracks("
                        "project_id, path, exact_sha256, duration_seconds, created_at"
                        ") VALUES (?, ?, ?, 180, ?)",
                        (
                            project_id,
                            str(root / f"track-{track_index}.wav"),
                            f"hash-{track_index}",
                            now,
                        ),
                    )
                    for label in NEW_LABELS:
                        connection.execute(
                            "INSERT INTO confirmed_label_events("
                            "project_id, track_id, label, state, event_kind, "
                            "batch_id, note, created_at) VALUES (?, ?, ?, ?, "
                            "'trusted_import', NULL, 'fixture', ?)",
                            (
                                project_id,
                                int(cursor.lastrowid),
                                label,
                                "positive" if track_index == 0 else "negative",
                                now,
                            ),
                        )
            for label in NEW_LABELS:
                update_label_goal(store, project_id, label, 1, 1)

            summary = fingerprint_project(
                store,
                project_id,
                Path("C:/Utils/tools/fpcalc.exe"),
            )

            self.assertEqual(summary.available, 1)
            self.assertEqual(summary.unavailable, 1)
            self.assertEqual(summary.errors, 0)
            with store.connection() as connection:
                tracks = connection.execute(
                    "SELECT acoustic_fingerprint FROM tracks ORDER BY id"
                ).fetchall()
                audit_statuses = [
                    row[0]
                    for row in connection.execute(
                        "SELECT status FROM fingerprint_audit ORDER BY track_id"
                    ).fetchall()
                ]
            self.assertEqual(tracks[0][0], "fingerprint-a")
            self.assertIsNone(tracks[1][0])
            self.assertEqual(audit_statuses, ["available", "unavailable"])

    def test_refuses_to_run_before_all_label_goals_are_complete(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = AnnotationStore(Path(temp_dir) / "annotations.db")
            store.initialize()
            project_id = store.create_project("incomplete")

            with self.assertRaisesRegex(RuntimeError, "all label goals"):
                fingerprint_project(store, project_id, Path("fpcalc"))
