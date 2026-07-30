#!/usr/bin/env python3

import importlib
import importlib.metadata
import json
import logging
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple


PACKAGE_IMPORT_MAP = {
    "maest-infer": "maest_infer",
}

TORCH_PACKAGE = "torch"
TORCHAUDIO_PACKAGE = "torchaudio"
TORCHVISION_PACKAGE = "torchvision"
DEFAULT_TORCH_STACK_VERSION = "2.11.0"

CUDA_WHEEL_INDEX_CU126 = "https://download.pytorch.org/whl/cu126"
CUDA_WHEEL_INDEX_CU128 = "https://download.pytorch.org/whl/cu128"
CUDA_WHEEL_INDEX_CU130 = "https://download.pytorch.org/whl/cu130"
CPU_WHEEL_INDEX = "https://download.pytorch.org/whl/cpu"


def _check_python_version() -> Tuple[bool, str, str]:
    current = sys.version_info
    required = (3, 10)
    ok = current >= required
    version_str = f"{current.major}.{current.minor}.{current.micro}"
    return ok, version_str, f">= {required[0]}.{required[1]}"


def _check_module(import_name: str) -> Tuple[bool, str, str]:
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, str(version), ""
    except Exception as exc:
        return False, "", str(exc)


def _check_sqlite_runtime() -> Tuple[bool, str, str]:
    try:
        version = str(sqlite3.sqlite_version)
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO t(value) VALUES ('ok')")
        row = connection.execute("SELECT value FROM t WHERE id = 1").fetchone()
        connection.close()
        if not row or row[0] != "ok":
            return (
                False,
                version,
                "sqlite runtime smoke test returned unexpected result",
            )
        return True, version, ""
    except Exception as exc:
        return False, "", str(exc)


def _check_maest_api() -> Tuple[bool, str]:
    try:
        module = importlib.import_module("maest_infer")
        if not hasattr(module, "get_maest"):
            return False, "module found, but get_maest is missing"
        return True, "get_maest available"
    except Exception as exc:
        return False, str(exc)


def _check_maest_version(required_version: str = "0.2.0") -> Tuple[bool, str, str]:
    try:
        installed_version = importlib.metadata.version("maest-infer")
    except importlib.metadata.PackageNotFoundError:
        return False, "", required_version
    return installed_version == required_version, installed_version, required_version


def _check_ffmpeg() -> Tuple[bool, str]:
    ffmpeg_path = shutil.which("ffmpeg")
    return ffmpeg_path is not None, ffmpeg_path or ""


def _has_nvidia_gpu() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        completed = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0 and bool(completed.stdout.strip())
    except Exception:
        return False


def _detect_cuda_version_from_nvidia_smi() -> str:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return ""
    try:
        completed = subprocess.run(
            [nvidia_smi],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return ""
        match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", completed.stdout)
        if not match:
            return ""
        return match.group(1)
    except Exception:
        return ""


def _version_prefix_matches(version: str, expected_prefix: str) -> bool:
    if not version:
        return False
    version_core = version.split("+", 1)[0]
    expected_core = expected_prefix.split("+", 1)[0]
    return version_core.startswith(expected_core)


def _parse_cuda_version(cuda_version: str) -> Tuple[int, int]:
    match = re.match(r"^(\d+)(?:\.(\d+))?", cuda_version.strip())
    if not match:
        return 0, 0
    major = int(match.group(1))
    minor = int(match.group(2) or "0")
    return major, minor


def _parse_semver_triplet(version: str) -> Tuple[int, int, int]:
    core = version.strip().split("+", 1)[0]
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", core)
    if not match:
        return 0, 0, 0
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _same_major_minor(version_a: str, version_b: str) -> bool:
    a_major, a_minor, _ = _parse_semver_triplet(version_a)
    b_major, b_minor, _ = _parse_semver_triplet(version_b)
    return a_major == b_major and a_minor == b_minor


def _load_available_package_versions(
    package_name: str, index_url: Optional[str]
) -> List[str]:
    cmd = [sys.executable, "-m", "pip", "index", "versions", package_name]
    if index_url:
        cmd.extend(["--index-url", index_url])

    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return []

    if completed.returncode != 0:
        return []

    output = completed.stdout or ""
    available_line = ""
    for line in output.splitlines():
        if line.strip().startswith("Available versions:"):
            available_line = line
            break

    if not available_line:
        return []

    versions_raw = available_line.split(":", 1)[-1].strip()
    versions = [value.strip() for value in versions_raw.split(",") if value.strip()]
    return versions


def _resolve_torch_stack_versions(index_url: Optional[str]) -> Tuple[str, str]:
    torch_versions = _load_available_package_versions(TORCH_PACKAGE, index_url)
    torchaudio_versions = _load_available_package_versions(
        TORCHAUDIO_PACKAGE, index_url
    )

    if not torch_versions or not torchaudio_versions:
        return DEFAULT_TORCH_STACK_VERSION, DEFAULT_TORCH_STACK_VERSION

    common_versions = sorted(
        set(torch_versions).intersection(set(torchaudio_versions)),
        key=_parse_semver_triplet,
        reverse=True,
    )
    if common_versions:
        selected = common_versions[0]
        return selected, selected

    torch_sorted = sorted(torch_versions, key=_parse_semver_triplet, reverse=True)
    torchaudio_sorted = sorted(
        torchaudio_versions, key=_parse_semver_triplet, reverse=True
    )

    for torch_version in torch_sorted:
        for torchaudio_version in torchaudio_sorted:
            if _same_major_minor(torch_version, torchaudio_version):
                return torch_version, torchaudio_version

    return torch_sorted[0], torchaudio_sorted[0]


def _select_torch_index_url(
    system_name: str, has_nvidia_gpu: bool, cuda_version: str
) -> Optional[str]:
    system = system_name.strip().lower()
    if not has_nvidia_gpu:
        if system == "linux":
            return CPU_WHEEL_INDEX
        return None

    cuda_major, cuda_minor = _parse_cuda_version(cuda_version)
    if system == "windows":
        if cuda_major > 13 or (cuda_major == 13 and cuda_minor >= 0):
            return CUDA_WHEEL_INDEX_CU130
        if cuda_major == 12 and cuda_minor >= 8:
            return CUDA_WHEEL_INDEX_CU128
        if cuda_major == 12 and cuda_minor >= 6:
            return CUDA_WHEEL_INDEX_CU126
        return CUDA_WHEEL_INDEX_CU126

    if system == "linux":
        if cuda_major > 13 or (cuda_major == 13 and cuda_minor >= 0):
            return None
        if cuda_major == 12 and cuda_minor >= 8:
            return CUDA_WHEEL_INDEX_CU128
        if cuda_major == 12 and cuda_minor >= 6:
            return CUDA_WHEEL_INDEX_CU126
        return CUDA_WHEEL_INDEX_CU126

    return None


def _install_torch_stack(
    index_url: Optional[str], torch_version: str, torchaudio_version: str
) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-cache-dir",
        f"{TORCH_PACKAGE}=={torch_version}",
        f"{TORCHAUDIO_PACKAGE}=={torchaudio_version}",
        TORCHVISION_PACKAGE,
    ]
    if index_url:
        cmd.extend(["--index-url", index_url])
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode == 0:
            return True, ""
        error_text = (completed.stderr or completed.stdout or "").strip()
        return False, error_text
    except Exception as exc:
        return False, str(exc)


def _probe_torch_info() -> Tuple[bool, str, str, bool]:
    probe_code = (
        "import json\n"
        "try:\n"
        " import torch\n"
        " print(json.dumps({'ok': True, 'version': str(torch.__version__), 'cuda_build': str(torch.version.cuda), 'cuda_available': bool(torch.cuda.is_available())}))\n"
        "except Exception as exc:\n"
        " print(json.dumps({'ok': False, 'error': str(exc)}))\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_code],
            check=False,
            capture_output=True,
            text=True,
        )
        raw = (completed.stdout or "").strip()
        if not raw:
            return False, "", "", False
        data = json.loads(raw)
        if not data.get("ok"):
            return False, "", "", False
        return (
            True,
            str(data.get("version", "")),
            str(data.get("cuda_build", "")),
            bool(data.get("cuda_available", False)),
        )
    except Exception:
        return False, "", "", False


def _ensure_torch_runtime(
    has_nvidia_gpu: bool, cuda_version: str, system_name: str
) -> Tuple[bool, bool]:
    restart_required = False
    index_url = _select_torch_index_url(system_name, has_nvidia_gpu, cuda_version)
    target_torch_version, target_torchaudio_version = _resolve_torch_stack_versions(
        index_url
    )

    def _manual_fix_message() -> str:
        base_cmd = (
            f"{sys.executable} -m pip install --upgrade --force-reinstall --no-cache-dir "
            f"{TORCH_PACKAGE}=={target_torch_version} "
            f"{TORCHAUDIO_PACKAGE}=={target_torchaudio_version} {TORCHVISION_PACKAGE}"
        )
        if index_url:
            return f"{base_cmd} --index-url {index_url}"
        return base_cmd

    logging.info(
        "Torch install target: platform=%s, nvidia_gpu=%s, cuda_version=%s, index_url=%s",
        system_name,
        has_nvidia_gpu,
        cuda_version or "unknown",
        index_url or "default",
    )
    logging.info(
        "Resolved torch targets: torch=%s, torchaudio=%s",
        target_torch_version,
        target_torchaudio_version,
    )

    try:
        import torch
        import torchaudio
        import torchvision

        torch_ok = _version_prefix_matches(str(torch.__version__), target_torch_version)
        torchaudio_ok = _version_prefix_matches(
            str(torchaudio.__version__), target_torchaudio_version
        )
        torchvision_ok = bool(str(torchvision.__version__).strip())
        torch_cuda_build = bool(torch.version.cuda)

        needs_reinstall = not (torch_ok and torchaudio_ok and torchvision_ok)
        if has_nvidia_gpu and not torch_cuda_build:
            needs_reinstall = True

        if not needs_reinstall:
            return True, restart_required

        logging.warning(
            "Installed torch stack does not match required versions/build (torch=%s, torchaudio=%s, torchvision=%s, cuda_build=%s)",
            torch.__version__,
            torchaudio.__version__,
            torchvision.__version__,
            torch.version.cuda,
        )
        logging.warning("Attempting to install compatible torch stack automatically...")
        logging.warning(
            "This may take several minutes. No progress messages will appear during download — this is expected."
        )
        install_ok, install_error = _install_torch_stack(
            index_url=index_url,
            torch_version=target_torch_version,
            torchaudio_version=target_torchaudio_version,
        )
        if not install_ok:
            logging.error("Failed to install torch stack automatically")
            if install_error:
                logging.error("Installer output: %s", install_error)
            logging.error("Manual fix: %s", _manual_fix_message())
            return False, False

        probe_ok, version, cuda_build, _ = _probe_torch_info()
        if not probe_ok:
            logging.error("Torch import verification failed after installation")
            logging.error("Manual fix: %s", _manual_fix_message())
            return False, False

        if has_nvidia_gpu and (not cuda_build or cuda_build == "None"):
            logging.error("CUDA torch build verification failed after installation")
            logging.error("Manual fix: %s", _manual_fix_message())
            return False, False

        logging.warning(
            "Torch stack installed successfully: %s (CUDA build: %s)",
            version,
            cuda_build,
        )
        logging.warning("Please restart the script to use updated torch stack")
        restart_required = True
    except Exception:
        logging.warning(
            "Torch is not installed. Installing torch stack automatically..."
        )
        install_ok, install_error = _install_torch_stack(
            index_url=index_url,
            torch_version=target_torch_version,
            torchaudio_version=target_torchaudio_version,
        )
        if not install_ok:
            logging.error("Failed to install torch stack automatically")
            if install_error:
                logging.error("Installer output: %s", install_error)
            logging.error("Manual fix: %s", _manual_fix_message())
            return False, False
        logging.warning("Torch stack installed successfully")
        logging.warning("Please restart the script to use updated packages")
        restart_required = True

    return True, restart_required


def _parse_requirements(requirements_path: Path) -> List[str]:
    if not requirements_path.is_file():
        return []

    packages: List[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "--requirement", "-e", "--editable", "--")):
            continue

        line = line.split("#", 1)[0].strip()
        if not line:
            continue

        package = re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip()
        if package:
            packages.append(package)
    return packages


def _package_to_import_name(package_name: str) -> str:
    key = package_name.lower()
    if key in PACKAGE_IMPORT_MAP:
        return PACKAGE_IMPORT_MAP[key]
    return package_name.replace("-", "_")


def run_environment_checks(
    project_dir: Path,
    models_dir: str,
    checkpoint_filename: str,
    checkpoint_path_value: str,
) -> bool:
    logging.info("Environment checks started")
    logging.info("Platform: %s", platform.platform())
    logging.info("Python executable: %s", sys.executable)

    ok_py, current_py, required_py = _check_python_version()
    if ok_py:
        logging.info("Python version: %s (required %s)", current_py, required_py)
    else:
        logging.error(
            "Python version mismatch: %s (required %s)", current_py, required_py
        )

    has_nvidia_gpu = _has_nvidia_gpu()
    logging.info("NVIDIA GPU detected: %s", has_nvidia_gpu)
    cuda_version = _detect_cuda_version_from_nvidia_smi() if has_nvidia_gpu else ""
    logging.info("Detected CUDA version from nvidia-smi: %s", cuda_version or "unknown")

    system_name = platform.system()

    torch_setup_ok, torch_restart_required = _ensure_torch_runtime(
        has_nvidia_gpu, cuda_version, system_name
    )
    if not torch_setup_ok:
        logging.error("Environment checks failed")
        return False

    requirements_path = project_dir / "requirements.txt"
    packages = _parse_requirements(requirements_path)
    required_pairs: List[Tuple[str, str]]
    if packages:
        required_pairs = [
            (_package_to_import_name(package), package) for package in packages
        ]
    else:
        required_pairs = [
            ("numpy", "numpy"),
            ("soundfile", "soundfile"),
            ("torch", "torch"),
            ("torchaudio", "torchaudio"),
            ("torchvision", "torchvision"),
            ("maest_infer", "maest-infer"),
            ("pandas", "pandas"),
            ("openpyxl", "openpyxl"),
            ("mutagen", "mutagen"),
        ]

    missing_packages: List[str] = []
    for import_name, package_name in required_pairs:
        ok, version, error = _check_module(import_name)
        if ok:
            logging.info("Module OK: %s (%s)", import_name, version)
        else:
            logging.error("Module missing: %s (%s)", import_name, error)
            missing_packages.append(package_name)

    maest_version_ok, installed_maest_version, required_maest_version = (
        _check_maest_version()
    )
    if maest_version_ok:
        logging.info("MAEST version: %s", installed_maest_version)
    else:
        logging.error(
            "MAEST version mismatch: installed %s, required %s",
            installed_maest_version or "not installed",
            required_maest_version,
        )

    maest_ok, maest_msg = _check_maest_api()
    if maest_ok:
        logging.info("MAEST API check: %s", maest_msg)
    else:
        logging.error("MAEST API check failed: %s", maest_msg)

    ffmpeg_ok, ffmpeg_path = _check_ffmpeg()
    if ffmpeg_ok:
        logging.info("FFmpeg found: %s", ffmpeg_path)
    else:
        logging.warning("FFmpeg was not found in PATH")

    sqlite_ok, sqlite_version, sqlite_error = _check_sqlite_runtime()
    if sqlite_ok:
        logging.info("SQLite runtime: %s", sqlite_version)
    else:
        logging.error("SQLite runtime check failed: %s", sqlite_error)

    torch_runtime_warning = False
    try:
        import torch

        logging.info("Torch version: %s", torch.__version__)
        logging.info("Torch CUDA build: %s", torch.version.cuda)
        logging.info("Torch CUDA available: %s", torch.cuda.is_available())

        cpu_ok = False
        try:
            x_cpu = torch.tensor([1.0, 2.0, 3.0], device="cpu")
            y_cpu = (x_cpu * 2).sum().item()
            cpu_ok = abs(y_cpu - 12.0) < 1e-6
        except Exception:
            cpu_ok = False

        cuda_available = torch.cuda.is_available()
        cuda_ok = False
        if cuda_available:
            try:
                x_cuda = torch.tensor([1.0, 2.0, 3.0], device="cuda")
                y_cuda = (x_cuda * 2).sum().item()
                cuda_ok = abs(y_cuda - 12.0) < 1e-6
            except Exception:
                cuda_ok = False

        if cpu_ok:
            logging.info("Torch CPU runtime test: OK")
        elif cuda_ok:
            logging.info("Torch runtime test: CPU failed, CUDA OK")
        else:
            torch_runtime_warning = True
            logging.warning("Torch runtime test failed for both CPU and CUDA")
            logging.warning("Pipeline will continue, but audio analysis may fail")
    except Exception as exc:
        torch_runtime_warning = True
        logging.warning("Torch runtime test skipped due to import error: %s", exc)
        logging.warning("Pipeline will continue, but audio analysis may fail")

    checkpoint_path_raw = str(checkpoint_path_value).strip()
    configured_checkpoint_path = None
    if checkpoint_path_raw:
        configured_checkpoint_path = Path(checkpoint_path_raw)
        if not configured_checkpoint_path.is_absolute():
            configured_checkpoint_path = project_dir / configured_checkpoint_path

        if configured_checkpoint_path.is_file():
            logging.info(
                "Checkpoint found at configured path: %s",
                configured_checkpoint_path,
            )
        else:
            logging.error(
                "Configured checkpoint path is invalid: %s",
                configured_checkpoint_path,
            )
    elif checkpoint_filename:
        checkpoint_path = project_dir / models_dir / checkpoint_filename
        if checkpoint_path.is_file():
            logging.info("Checkpoint found: %s", checkpoint_path)
        else:
            logging.warning("Checkpoint missing: %s", checkpoint_path)
            logging.info(
                "Checkpoint will be downloaded to %s when model loading starts",
                checkpoint_path,
            )

    missing_unique = sorted(set(missing_packages))
    if missing_unique:
        joined = " ".join(missing_unique)
        logging.error("Missing Python packages: %s", joined)
        logging.error("Suggested fix: pip install %s", joined)

    if not maest_ok:
        logging.error("Fix maest_infer so `from maest_infer import get_maest` works")

    checkpoint_path_ok = (not checkpoint_path_raw) or (
        configured_checkpoint_path is not None and configured_checkpoint_path.is_file()
    )
    hard_fail = (
        (not ok_py)
        or bool(missing_unique)
        or (not maest_version_ok)
        or (not maest_ok)
        or (not sqlite_ok)
        or (not checkpoint_path_ok)
        or torch_restart_required
    )
    if hard_fail:
        logging.error("Environment checks failed")
        return False

    if torch_runtime_warning:
        logging.warning("Environment checks passed with torch runtime warnings")
    else:
        logging.info("Environment checks passed")
    return True
