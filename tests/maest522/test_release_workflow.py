from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from safetensors.torch import load_file

from tests.maest522 import test_release as release_fixtures
from tools.maest522.hub_publish import main as hub_main
from tools.maest522.hub_publish import validate_for_hub
from tools.maest522.native_model import load_native_state
from tools.maest522.release import MODEL_ALLOWLIST, stage_release


class ReleaseWorkflowTests(TestCase):
    @patch("tools.maest522.hub_publish.HfApi")
    def test_rehearses_native_hf_stage_and_offline_hub_validation(self, api_class) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = release_fixtures.ReleaseStagingTests()._fixture(root)

            release = root / "release"
            stage_report = stage_release(inputs, release)
            native = load_native_state(release / "native-maest-522.ckpt")
            hf_state = load_file(release / "model.safetensors", device="cpu")
            hub_report = validate_for_hub(release)
            exit_code = hub_main(["--release-dir", str(release)])

            self.assertEqual(tuple(native["head.1.weight"].shape), (522, 768))
            self.assertEqual(
                tuple(hf_state["classifier.dense.weight"].shape),
                (522, 768),
            )
            self.assertEqual(stage_report.files, MODEL_ALLOWLIST)
            self.assertEqual(hub_report.file_count, len(MODEL_ALLOWLIST))
            self.assertEqual(exit_code, 0)
            api_class.assert_not_called()
