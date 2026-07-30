import importlib.metadata
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from environment import _check_maest_version


class MaestVersionCheckTests(unittest.TestCase):
    @patch("importlib.metadata.version", return_value="0.1.0")
    def test_rejects_installed_version_below_required_release(self, version):
        ok, installed_version, required_version = _check_maest_version()

        self.assertFalse(ok)
        self.assertEqual(installed_version, "0.1.0")
        self.assertEqual(required_version, "0.2.0")

    @patch("importlib.metadata.version", return_value="0.2.0")
    def test_accepts_installed_version_matching_required_release(self, version):
        ok, installed_version, required_version = _check_maest_version()

        self.assertTrue(ok)
        self.assertEqual(installed_version, "0.2.0")
        self.assertEqual(required_version, "0.2.0")
