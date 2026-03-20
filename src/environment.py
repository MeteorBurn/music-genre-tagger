#!/usr/bin/env python3

import importlib
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List
from typing import Tuple


PACKAGE_IMPORT_MAP = {
    "maest-infer": "maest_infer",
}

CUDA_WHEEL_INDEX = "https://download.pytorch.org/whl/cu121"


def _check_python_version() -> Tuple[bool, str, str]:
    current = sys.version_info
    required = (3, 9)
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


def _check_maest_api() -> Tuple[bool, str]:
    try:
        module = importlib.import_module("maest_infer")
        if not hasattr(module, "get_maest"):
            return False, "module found, but get_maest is missing"
        return True, "get_maest available"
    except Exception as exc:
        return False, str(exc)


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


def _install_torch_stack(use_cuda: bool) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-cache-dir",
        "torch",
        "torchaudio",
        "torchvision",
    ]
    if use_cuda:
        cmd.extend(["--index-url", CUDA_WHEEL_INDEX])
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


def _ensure_torch_runtime(has_nvidia_gpu: bool) -> Tuple[bool, bool]:
    restart_required = False
    try:
        import torch

        torch_cuda_build = bool(torch.version.cuda)
        if has_nvidia_gpu and not torch_cuda_build:
            logging.warning(
                "NVIDIA GPU detected, but CPU-only torch build is installed"
            )
            logging.warning("Attempting to install CUDA torch build automatically...")
            logging.warning(
                "This may take several minutes. No progress messages will appear during download — this is expected."
            )
            install_ok, install_error = _install_torch_stack(use_cuda=True)
            if not install_ok:
                logging.error("Failed to install CUDA torch build automatically")
                if install_error:
                    logging.error("Installer output: %s", install_error)
                logging.error(
                    "Manual fix: %s -m pip install --upgrade torch torchaudio torchvision --index-url %s",
                    sys.executable,
                    CUDA_WHEEL_INDEX,
                )
                return False, False
            probe_ok, version, cuda_build, _ = _probe_torch_info()
            if not probe_ok or not cuda_build or cuda_build == "None":
                logging.error("CUDA torch build verification failed after installation")
                logging.error(
                    "Manual fix: %s -m pip uninstall -y torch torchaudio torchvision",
                    sys.executable,
                )
                logging.error(
                    "Manual fix: %s -m pip install --upgrade --force-reinstall --no-cache-dir torch torchaudio torchvision --index-url %s",
                    sys.executable,
                    CUDA_WHEEL_INDEX,
                )
                return False, False
            logging.warning("CUDA torch build installed successfully: %s", version)
            logging.warning("Detected CUDA build: %s", cuda_build)
            logging.warning("Please restart the script to use updated torch build")
            restart_required = True
    except Exception:
        logging.warning(
            "Torch is not installed. Installing torch stack automatically..."
        )
        install_ok, install_error = _install_torch_stack(use_cuda=has_nvidia_gpu)
        if not install_ok:
            logging.error("Failed to install torch stack automatically")
            if install_error:
                logging.error("Installer output: %s", install_error)
            if has_nvidia_gpu:
                logging.error(
                    "Manual fix: %s -m pip install --upgrade torch torchaudio torchvision --index-url %s",
                    sys.executable,
                    CUDA_WHEEL_INDEX,
                )
            else:
                logging.error(
                    "Manual fix: %s -m pip install --upgrade torch torchaudio torchvision",
                    sys.executable,
                )
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

    torch_setup_ok, torch_restart_required = _ensure_torch_runtime(has_nvidia_gpu)
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
            ("maest_infer", "maest-infer"),
        ]

    missing_packages: List[str] = []
    for import_name, package_name in required_pairs:
        ok, version, error = _check_module(import_name)
        if ok:
            logging.info("Module OK: %s (%s)", import_name, version)
        else:
            logging.error("Module missing: %s (%s)", import_name, error)
            missing_packages.append(package_name)

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
        or (not maest_ok)
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
