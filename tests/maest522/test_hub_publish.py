import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

from tools.maest522.hub_publish import main, publish_model, validate_for_hub
from tools.maest522.release import MODEL_ALLOWLIST


class HubPublicationTests(TestCase):
    def _release(self, root: Path) -> Path:
        release = root / "release"
        release.mkdir()
        for name in MODEL_ALLOWLIST - {"SHA256SUMS"}:
            content = b"placeholder"
            if name == "README.md":
                content = b"---\nlicense: other\n---\n# MAEST 522\n"
            elif name == "evaluation.json":
                content = b'{"legacy_gates_passed":true,"test_evaluated":true}'
            elif name == "parity.json":
                content = b'{"passed":true,"windows":30,"tracks":10}'
            (release / name).write_bytes(content)
        lines = []
        for path in sorted(release.iterdir(), key=lambda item: item.name):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.name}")
        (release / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return release

    @patch("tools.maest522.hub_publish.HfApi")
    def test_default_command_validates_only_without_network_mutation(self, api_class) -> None:
        with TemporaryDirectory() as temp_dir:
            release = self._release(Path(temp_dir))

            exit_code = main(["--release-dir", str(release)])

            self.assertEqual(exit_code, 0)
            api_class.assert_not_called()

    def test_push_requires_repo_id(self) -> None:
        with TemporaryDirectory() as temp_dir:
            release = self._release(Path(temp_dir))
            with self.assertRaisesRegex(ValueError, "repo-id"):
                main(["--release-dir", str(release), "--push"])

    @patch("tools.maest522.hub_publish.HfApi")
    def test_publish_defaults_private_and_never_deletes_remote_files(self, api_class) -> None:
        with TemporaryDirectory() as temp_dir:
            release = self._release(Path(temp_dir))
            api = MagicMock()
            api.create_repo.return_value = "https://huggingface.co/org/model"
            api_class.return_value = api

            url = publish_model(
                release,
                "org/model",
                private=True,
                revision="main",
                version="0.1.0",
            )

            self.assertEqual(url, "https://huggingface.co/org/model")
            api.create_repo.assert_called_once_with(
                repo_id="org/model",
                repo_type="model",
                private=True,
                exist_ok=True,
            )
            api.upload_folder.assert_called_once()
            upload_kwargs = api.upload_folder.call_args.kwargs
            self.assertNotIn("delete_patterns", upload_kwargs)
            self.assertIn("0.1.0", upload_kwargs["commit_message"])

    def test_validation_detects_hash_or_allowlist_drift(self) -> None:
        with TemporaryDirectory() as temp_dir:
            release = self._release(Path(temp_dir))
            report = validate_for_hub(release)
            self.assertEqual(report.file_count, len(MODEL_ALLOWLIST))
            (release / "README.md").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validate_for_hub(release)
