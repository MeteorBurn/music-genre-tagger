#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import platform
import re
import shutil
import sys
from pathlib import Path


CONFIG = {
    "models_dir": "E:/Projects/MusicTagger/models",
    "checkpoint_filename": "discogs-maest-30s-pw-129e-519l-swa.ckpt",
    "requirements_path": "requirements.txt",
}


REQUIRED_MODULES = [
    ("numpy", "numpy"),
    ("soundfile", "soundfile"),
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("maest_infer", "maest_infer"),
]

PACKAGE_IMPORT_MAP = {
    "maest-infer": "maest_infer",
}


def check_python_version():
    current = sys.version_info
    required = (3, 9)
    ok = current >= required
    version_str = f"{current.major}.{current.minor}.{current.micro}"
    return ok, version_str, f">= {required[0]}.{required[1]}"


def check_module(import_name: str):
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version, ""
    except Exception as exc:
        return False, "", str(exc)


def check_maest_api():
    try:
        module = importlib.import_module("maest_infer")
        if not hasattr(module, "get_maest"):
            return False, "module found, but get_maest is missing"
        return True, "get_maest available"
    except Exception as exc:
        return False, str(exc)


def check_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")
    return (ffmpeg_path is not None), (ffmpeg_path or "")


def check_model_file(models_dir: str, checkpoint_filename: str):
    if not checkpoint_filename:
        return True, None
    ckpt = Path(models_dir) / checkpoint_filename
    return ckpt.is_file(), ckpt


def parse_requirements(requirements_path: Path):
    if not requirements_path.is_file():
        return []

    packages = []
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


def package_to_import_name(package_name: str):
    key = package_name.lower()
    if key in PACKAGE_IMPORT_MAP:
        return PACKAGE_IMPORT_MAP[key]
    return package_name.replace("-", "_")


def main():
    print("Environment check for MusicTagger (no installs, diagnostics only)")
    print(f"Platform: {platform.platform()}")
    print(f"Python executable: {sys.executable}")
    print("-" * 72)

    ok_py, current_py, required_py = check_python_version()
    print(
        f"Python: {current_py} (required {required_py}) -> {'OK' if ok_py else 'MISSING'}"
    )

    requirements_path = Path(CONFIG["requirements_path"])
    if not requirements_path.is_absolute():
        requirements_path = Path(__file__).resolve().parent / requirements_path

    requirement_packages = parse_requirements(requirements_path)
    if requirement_packages:
        required_pairs = [(package_to_import_name(p), p) for p in requirement_packages]
        print(f"\nRequired Python modules (from {requirements_path.name}):")
    else:
        required_pairs = REQUIRED_MODULES
        print("\nRequired Python modules (fallback list):")

    missing = []
    for import_name, pip_name in required_pairs:
        ok, version, error = check_module(import_name)
        if ok:
            print(f"- {import_name}: OK (version: {version})")
        else:
            print(f"- {import_name}: MISSING ({error})")
            missing.append(pip_name)

    maest_ok, maest_msg = check_maest_api()
    print(f"\nMAEST API check: {'OK' if maest_ok else 'MISSING'} ({maest_msg})")

    ffmpeg_ok, ffmpeg_path = check_ffmpeg()
    print(f"FFmpeg on PATH: {'OK' if ffmpeg_ok else 'MISSING'}")
    if ffmpeg_ok:
        print(f"- ffmpeg path: {ffmpeg_path}")

    try:
        import torch

        print("\nTorch runtime:")
        print(f"- torch.cuda.is_available(): {torch.cuda.is_available()}")
        cpu_ok = False
        try:
            x_cpu = torch.tensor([1.0, 2.0, 3.0], device="cpu")
            y_cpu = (x_cpu * 2).sum().item()
            cpu_ok = abs(y_cpu - 12.0) < 1e-6
        except Exception:
            cpu_ok = False
        print(f"- CPU test op: {'OK' if cpu_ok else 'FAILED'}")

        if torch.cuda.is_available():
            print(f"- CUDA devices: {torch.cuda.device_count()}")
            for idx in range(torch.cuda.device_count()):
                print(f"  - [{idx}] {torch.cuda.get_device_name(idx)}")
            cuda_ok = False
            try:
                x_cuda = torch.tensor([1.0, 2.0, 3.0], device="cuda")
                y_cuda = (x_cuda * 2).sum().item()
                cuda_ok = abs(y_cuda - 12.0) < 1e-6
            except Exception:
                cuda_ok = False
            print(f"- CUDA test op: {'OK' if cuda_ok else 'FAILED'}")
        else:
            print("- CUDA test op: SKIPPED (CUDA unavailable)")
    except Exception:
        pass

    model_ok, model_path = check_model_file(
        CONFIG["models_dir"], CONFIG["checkpoint_filename"]
    )
    print("\nModel file:")
    if model_path is None:
        print("- local checkpoint: OPTIONAL (auto-download mode)")
    else:
        print(f"- expected: {model_path}")
        print(
            f"- status: {'OK' if model_ok else 'MISSING (auto-download can still work)'}"
        )

    print("\n" + "-" * 72)
    if missing:
        unique_missing = sorted(set(missing))
        print("Missing Python packages to install:")
        print("- " + " ".join(unique_missing))
        print("Suggested command:")
        print("pip install " + " ".join(unique_missing))
    else:
        print("All required Python packages are available.")

    if not maest_ok:
        print(
            "Add/fix maest_infer package so `from maest_infer import get_maest` works."
        )
    if not model_ok:
        print(
            "Local checkpoint missing: script can still run via maest_infer auto-download."
        )


if __name__ == "__main__":
    main()
