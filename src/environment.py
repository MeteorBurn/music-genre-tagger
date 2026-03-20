#!/usr/bin/env python3

import importlib
import logging
import platform
import re
import shutil
import sys
from pathlib import Path
from typing import List
from typing import Tuple


PACKAGE_IMPORT_MAP = {
    "maest-infer": "maest_infer",
}


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
    )
    if hard_fail:
        logging.error("Environment checks failed")
        return False

    if torch_runtime_warning:
        logging.warning("Environment checks passed with torch runtime warnings")
    else:
        logging.info("Environment checks passed")
    return True
