import importlib.metadata
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from environment import _check_maest_version
from environment import _check_python_version


class VersionInfo(tuple):
    def __new__(cls, major: int, minor: int, micro: int) -> "VersionInfo":
        return super().__new__(cls, (major, minor, micro))

    @property
    def major(self) -> int:
        return self[0]

    @property
    def minor(self) -> int:
        return self[1]

    @property
    def micro(self) -> int:
        return self[2]


class PythonVersionCheckTests(unittest.TestCase):
    @patch("environment.sys.version_info", VersionInfo(3, 9, 18))
    def test_rejects_python_3_9(self) -> None:
        ok, current_version, required_version = _check_python_version()

        self.assertFalse(ok)
        self.assertEqual(current_version, "3.9.18")
        self.assertEqual(required_version, ">= 3.10")

    @patch("environment.sys.version_info", VersionInfo(3, 10, 0))
    def test_accepts_python_3_10(self) -> None:
        ok, current_version, required_version = _check_python_version()

        self.assertTrue(ok)
        self.assertEqual(current_version, "3.10.0")
        self.assertEqual(required_version, ">= 3.10")


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
