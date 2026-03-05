#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import traceback
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

try:
    from maest_infer import get_maest
except ImportError:
    print("ERROR: maest_infer package/module not found in environment.")
    print(
        "Install MAEST inference package so import `from maest_infer import get_maest` works."
    )
    raise SystemExit(1)


# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
CONFIG = {
    # --- Logging ---
    "loglevel": "INFO",  # Levels: DEBUG, INFO, WARNING, ERROR
    # --- Input/Output ---
    "input_directory": "M:/Ambient",
    "output_directory": "json",
    "file_pattern": "",
    "max_files": 0,
    # --- Conversion Parameters ---
    "convert_to_wav": False,
    "ffmpeg_path": "ffmpeg",
    "temp_dir": "",
    # --- Audio Processing Parameters ---
    "audio_offset": 60,
    "audio_duration": 30,
    # --- Audio Parameters ---
    "sample_rate": 16000,
    # --- Model Paths ---
    "models_dir": "models",
    # --- MAEST Models (PyTorch) ---
    "maest_models": {
        "maest_519l_pytorch": {
            "enabled": True,
            "arch": "discogs-maest-30s-pw-129e-519l",
            "checkpoint_path": "",  # optional absolute path
            "checkpoint_filename": "",  # optional filename in models_dir
            "num_genres": 3,
        }
    },
}


# =============================================================================
# CONSTANTS
# =============================================================================
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"]
ClassificationResult = Dict[str, Any]
SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def setup_logging(level: str):
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Logging level set to: %s", level.upper())


def resolve_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return SCRIPT_DIR / p


def find_audio_files(directory: str, file_pattern: str = "") -> List[Path]:
    audio_files: List[Path] = []
    start_dir = resolve_path(directory)
    pattern_lower = file_pattern.lower() if file_pattern else None
    if not start_dir.is_dir():
        logging.error("Input directory not found: %s", start_dir)
        return []

    logging.info("Searching for all audio files in %s...", start_dir)
    for item in start_dir.rglob("*"):
        if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS:
            if pattern_lower and pattern_lower not in item.stem.lower():
                continue
            audio_files.append(item)

    audio_files.sort()
    logging.info("Found %d total audio files.", len(audio_files))
    return audio_files


def get_existing_json_stems(directory: str) -> Set[str]:
    json_stems: Set[str] = set()
    output_dir = resolve_path(directory)
    if not output_dir.is_dir():
        return json_stems

    for item in output_dir.glob("*.json"):
        if item.is_file():
            json_stems.add(item.stem)
    return json_stems


def trim_audio_segment(
    audio: np.ndarray, sample_rate: int, offset_sec: float, duration_sec: float
) -> np.ndarray:
    if audio is None or len(audio) == 0:
        return audio

    start_sample = int(offset_sec * sample_rate)
    end_sample = start_sample + int(duration_sec * sample_rate)

    if start_sample >= len(audio):
        start_sample = 0
        end_sample = int(duration_sec * sample_rate)

    if end_sample > len(audio):
        end_sample = len(audio)

    return audio[start_sample:end_sample]


def convert_audio_to_wav(
    input_path: Path, target_sr: int, ffmpeg_exec: str, temp_dir: Optional[str]
) -> Optional[Path]:
    temp_wav_path: Optional[Path] = None
    try:
        fd, temp_wav_path_str = tempfile.mkstemp(
            suffix=".wav", prefix="convert_", dir=temp_dir or None
        )
        os.close(fd)
        temp_wav_path = Path(temp_wav_path_str)

        command = [
            ffmpeg_exec,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path.resolve()),
            "-y",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(target_sr),
            "-ac",
            "1",
            str(temp_wav_path.resolve()),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

        if temp_wav_path.exists() and temp_wav_path.stat().st_size > 0:
            return temp_wav_path
        return None
    except Exception as e:
        logging.error("ffmpeg conversion error for %s: %s", input_path.name, e)
        if temp_wav_path and temp_wav_path.exists():
            temp_wav_path.unlink(missing_ok=True)
        return None


def get_platform_paths(file_path: Path) -> Dict[str, str]:
    win_path = str(file_path)
    wsl_path = win_path
    try:
        resolved = file_path.resolve()
        parts = resolved.parts
        if len(parts) >= 2:
            drive_raw = parts[0].replace(":", "")
            if len(drive_raw) == 1 and drive_raw.isalpha():
                tail = "/".join(parts[1:]).replace("\\", "/")
                wsl_path = f"/mnt/{drive_raw.lower()}/{tail}"
    except Exception:
        pass
    return {"win": win_path, "wsl": wsl_path}


def load_mono_16k(audio_path: Path, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(str(audio_path), always_2d=False)
    if isinstance(wav, np.ndarray) and wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)

    if sr != target_sr:
        wav_t = torch.from_numpy(wav)
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
        wav = wav_t.detach().cpu().numpy().astype(np.float32)

    return wav


def resolve_checkpoint_path(
    model_params: Dict[str, Any], models_dir: Path
) -> Optional[Path]:
    checkpoint_path_raw = model_params.get("checkpoint_path", "")
    if checkpoint_path_raw:
        checkpoint_path = Path(checkpoint_path_raw)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found at configured checkpoint_path: {checkpoint_path}. "
                "Clear 'checkpoint_path' to load model from models_dir/checkpoint_filename."
            )
        return checkpoint_path

    checkpoint_filename = model_params.get("checkpoint_filename", "")
    if not checkpoint_filename:
        return None

    checkpoint_path = models_dir / checkpoint_filename
    if not checkpoint_path.is_file():
        logging.warning(
            "Checkpoint not found at %s, will use maest_infer pretrained weights.",
            checkpoint_path,
        )
        return None
    return checkpoint_path


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_models(config: Dict) -> Optional[Dict[str, Any]]:
    models: Dict[str, Any] = {"maest": {}}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Using device: %s", device)
    logging.info("Using maest_infer from Python environment")
    models_base_path = resolve_path(config.get("models_dir", ""))

    logging.info("Loading MAEST models...")
    for model_name, model_params in config.get("maest_models", {}).items():
        if not model_params.get("enabled", False):
            continue

        try:
            ckpt_path = resolve_checkpoint_path(model_params, models_base_path)
        except Exception as path_err:
            logging.error(
                "Checkpoint resolution error for '%s': %s", model_name, path_err
            )
            raise

        arch = model_params.get("arch", "discogs-maest-30s-pw-129e-519l")
        try:
            use_pretrained = ckpt_path is None
            model = get_maest(arch, pretrained=use_pretrained).eval().to(device)

            if ckpt_path is not None:
                state = torch.load(str(ckpt_path), map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)

            if device == "cuda" and hasattr(model, "init_melspectrogram"):
                model.init_melspectrogram()
                if hasattr(model, "melspectrogram"):
                    model.melspectrogram = model.melspectrogram.cuda()

            models["maest"][model_name] = {
                "model": model,
                "arch": arch,
                "checkpoint": str(ckpt_path)
                if ckpt_path is not None
                else "pretrained:auto",
                "device": device,
            }
            logging.info(" -> Loaded MAEST model: '%s' (arch: %s)", model_name, arch)
        except Exception as e:
            logging.error("Failed to load MAEST model '%s': %s", model_name, e)
            logging.debug(traceback.format_exc())

    if not models["maest"]:
        logging.warning("No active MAEST models were loaded.")

    return models


# =============================================================================
# PREDICTION PROCESSING
# =============================================================================
def process_predictions(
    scores: np.ndarray, labels: List[str], top_n: int
) -> List[Tuple[str, float]]:
    if scores is None or labels is None:
        return []

    scores_np = np.asarray(scores).squeeze()
    if scores_np.ndim != 1:
        scores_np = np.mean(scores_np, axis=0)

    if len(scores_np) != len(labels):
        logging.warning(
            "Prediction/label size mismatch (%d vs %d).", len(scores_np), len(labels)
        )
        return []

    idx = np.argsort(scores_np)[-top_n:][::-1]
    return [(labels[i], float(scores_np[i])) for i in idx]


def process_labels(raw_labels: List[str]) -> List[str]:
    return [label.split("---", 1)[-1] for label in raw_labels]


# =============================================================================
# CORE ANALYSIS FUNCTION
# =============================================================================
def analyze_audio_file(
    original_audio_path: Path,
    models: Dict[str, Any],
    config: Dict,
    output_dir: Path,
) -> Optional[ClassificationResult]:
    path_to_analyze = original_audio_path
    temp_wav_file: Optional[Path] = None
    needs_cleanup = False

    if config["convert_to_wav"] and original_audio_path.suffix.lower() != ".wav":
        temp_wav_file = convert_audio_to_wav(
            original_audio_path,
            config["sample_rate"],
            config["ffmpeg_path"],
            config.get("temp_dir"),
        )
        if temp_wav_file:
            path_to_analyze, needs_cleanup = temp_wav_file, True
        else:
            return None

    json_data: ClassificationResult = {
        "file_path": get_platform_paths(original_audio_path),
        "file_name": original_audio_path.stem,
        "file_extension": original_audio_path.suffix,
        "timestamp": datetime.now().isoformat(),
        "analysis_config": {
            "audio_segment_offset": config["audio_offset"],
            "audio_segment_duration": config["audio_duration"],
        },
        "analysis_results": {},
    }

    try:
        audio = load_mono_16k(path_to_analyze, config["sample_rate"])
        if audio is None or len(audio) < config["sample_rate"] * 0.1:
            raise ValueError("Audio is empty or too short.")

        audio = trim_audio_segment(
            audio,
            config["sample_rate"],
            config["audio_offset"],
            config["audio_duration"],
        )
        if audio is None or len(audio) < config["sample_rate"] * 0.1:
            raise ValueError("Audio segment is too short after trimming.")

        audio_tensor = torch.from_numpy(audio).float()
        all_results: Dict[str, Any] = {}

        if models.get("maest"):
            for name, maest_data in models["maest"].items():
                try:
                    model_params = config.get("maest_models", {}).get(name, {})
                    top_n = model_params.get("num_genres", 5)
                    model = maest_data["model"]
                    device = maest_data["device"]

                    wav = audio_tensor.to(device)
                    with torch.no_grad():
                        scores, labels = model.predict_labels(wav)
                        if device == "cuda":
                            torch.cuda.synchronize()

                    cleaned_labels = process_labels(list(labels))
                    structured_result = process_predictions(
                        np.asarray(scores), cleaned_labels, top_n
                    )
                    all_results[name] = {
                        "labels": [label for label, _score in structured_result],
                        "confidences": [
                            round(score, 4) for _label, score in structured_result
                        ],
                    }
                except Exception as maest_err:
                    logging.error("Error in MAEST model '%s': %s", name, maest_err)
                    all_results[name] = {"error": str(maest_err)}

        json_data["analysis_results"] = all_results

    except Exception as e:
        logging.error("Failed to analyze %s: %s", original_audio_path.name, e)
        logging.debug(traceback.format_exc())
        json_data["error"] = str(e)

    finally:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            json_filename = output_dir / (original_audio_path.stem + ".json")
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        except Exception as json_err:
            logging.error(
                "Could not save JSON for %s: %s", original_audio_path.name, json_err
            )

        if needs_cleanup and temp_wav_file and temp_wav_file.exists():
            temp_wav_file.unlink(missing_ok=True)

    return json_data


# =============================================================================
# NUMPY JSON ENCODER
# =============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def format_log_summary(results: Dict) -> str:
    summary_parts: List[str] = []

    model_log_order = ["maest_519l_pytorch"]
    log_labels = {"maest_519l_pytorch": "Genre(MAEST)"}

    for model_name in model_log_order:
        result = results.get(model_name)
        if not result:
            continue
        if "labels" in result and result["labels"]:
            summary_parts.append(
                f"{log_labels.get(model_name, model_name)}: {result['labels'][0]}"
            )

    if not summary_parts:
        return "No conclusive tags found"
    return " | ".join(summary_parts)


def run_analysis(config: Dict):
    start_time = datetime.now()
    logging.info(
        "%s Starting Universal Audio Analyzer (PyTorch MAEST) %s", "=" * 20, "=" * 20
    )

    output_dir = resolve_path(config["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    models = load_models(config)
    if not models:
        logging.critical("Model loading failed. Exiting.")
        return

    all_audio_files = find_audio_files(
        config["input_directory"], config["file_pattern"]
    )
    if not all_audio_files:
        return

    existing_stems = get_existing_json_stems(config["output_directory"])
    unprocessed_files = [f for f in all_audio_files if f.stem not in existing_stems]

    if not unprocessed_files:
        logging.info(
            "All audio files already have corresponding JSON results. Nothing to do."
        )
        return

    max_files_to_process = config.get("max_files", 0)
    if max_files_to_process > 0:
        files_to_process = unprocessed_files[:max_files_to_process]
        logging.info(
            "Processing the next batch of %d out of %d remaining files.",
            len(files_to_process),
            len(unprocessed_files),
        )
    else:
        files_to_process = unprocessed_files
        logging.info("Processing all %d remaining files.", len(unprocessed_files))

    total_files = len(files_to_process)
    processed_count, error_count = 0, 0

    for i, audio_path in enumerate(files_to_process):
        file_start_time = datetime.now()
        result = analyze_audio_file(audio_path, models, config, output_dir)
        file_duration = (datetime.now() - file_start_time).total_seconds()

        if result and not result.get("error"):
            processed_count += 1
            summary = format_log_summary(result.get("analysis_results", {}))
            logging.info(
                "[%d/%d] %s - (%s) (time: %.2fs)",
                i + 1,
                total_files,
                audio_path.name,
                summary,
                file_duration,
            )
        else:
            error_count += 1
            logging.warning(
                "[%d/%d] %s [ERROR] (time: %.2fs)",
                i + 1,
                total_files,
                audio_path.name,
                file_duration,
            )

    total_duration = (datetime.now() - start_time).total_seconds()
    logging.info("%s Analysis Complete %s", "=" * 20, "=" * 20)
    logging.info("Successfully processed: %d files", processed_count)
    logging.info("Failed with errors: %d files", error_count)
    logging.info("Total execution time: %s", timedelta(seconds=total_duration))


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    setup_logging(CONFIG.get("loglevel", "INFO"))

    if not resolve_path(CONFIG["input_directory"]).is_dir():
        logging.critical(
            "Input directory does not exist: %s", CONFIG["input_directory"]
        )
        raise SystemExit(1)

    try:
        run_analysis(CONFIG)
    except Exception as e:
        logging.critical("A critical error occurred during execution: %s", e)
        logging.critical(traceback.format_exc())
        raise SystemExit(1)
