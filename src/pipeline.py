#!/usr/bin/env python3

import hashlib
import logging
import os
import re
import subprocess
import tempfile
import traceback
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from decimal import ROUND_DOWN
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from mutagen import File as MutagenFile

from extractor import create_excel_report
from report import load_best_available_library_summary
from report import load_cumulative_timings
from report import summarize_database_library
from report import write_markdown_report
from storage import export_tracks_json
from storage import get_existing_hashes
from storage import init_db
from storage import upsert_track
from storage import update_track_statuses
from tagger import run_genre_tagging

try:
    from maest_infer import get_maest
except ImportError:
    get_maest = None


DEFAULT_AUDIO_EXTENSIONS = [
    ".flac",
    ".wav",
    ".aiff",
    ".aif",
    ".m4a",
    ".dsf",
    ".ape",
    ".wv",
    ".mp3",
]

SOUNDFILE_DIRECT_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3"}

DEFAULT_MODEL_KEY = "maest_519l_pytorch"
DEFAULT_MODEL_ARCH = "discogs-maest-30s-pw-129e-519l"
DEFAULT_MODELS_DIR = "src/models"
DEFAULT_CHECKPOINT_FILENAME = "discogs-maest-30s-pw-129e-519l-swa.ckpt"
DEFAULT_CHECKPOINT_URL = "https://github.com/palonso/MAEST/releases/download/v0.0.0-beta/discogs-maest-30s-pw-129e-519l-swa.ckpt"


@dataclass
class RuntimePaths:
    input_dir: Optional[Path]
    output_base: Optional[Path]
    meta_root: Optional[Path]
    db_path: Path
    tracks_json_path: Path
    excel_path: Path
    report_path: Path


def setup_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _resolve_path(script_dir: Path, path_value: str) -> Path:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return script_dir / path_obj


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "input"


def apply_cli_overrides(base_config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    config = deepcopy(base_config)

    if args.loglevel:
        config["loglevel"] = args.loglevel
    if args.input_directory is not None:
        config["input_directory"] = args.input_directory
    if args.output_directory is not None:
        config["output_directory"] = args.output_directory
    if args.file_pattern is not None:
        config["file_pattern"] = args.file_pattern
    if args.max_files is not None:
        config["max_files"] = args.max_files
    if args.convert_to_wav:
        config["convert_to_wav"] = True
    if args.write_json:
        config["write_json"] = True
    if args.tag_yes:
        config["tag_mode"] = "yes"
    elif args.tag_no:
        config["tag_mode"] = "no"
    else:
        config["tag_mode"] = "ask"
    return config


def apply_model_runtime_defaults(
    config: Dict[str, Any], base_dir: Path
) -> Dict[str, Any]:
    runtime = deepcopy(config)

    model_file_path_raw = str(runtime.get("model_file_path", "")).strip()
    user_model_key = str(runtime.get("model_key", "")).strip()

    checkpoint_path = ""
    result_key = DEFAULT_MODEL_KEY
    if model_file_path_raw:
        custom_model_path = Path(model_file_path_raw)
        if not custom_model_path.is_absolute():
            custom_model_path = base_dir / custom_model_path
        checkpoint_path = str(custom_model_path)
        result_key = user_model_key or DEFAULT_MODEL_KEY

    runtime["maest_result_key"] = result_key
    runtime["maest_models"] = {
        result_key: {
            "enabled": True,
            "arch": DEFAULT_MODEL_ARCH,
            "checkpoint_path": checkpoint_path,
            "checkpoint_filename": DEFAULT_CHECKPOINT_FILENAME,
            "checkpoint_url": DEFAULT_CHECKPOINT_URL,
        }
    }
    return runtime


def _prompt_for_path(prompt_text: str, allow_empty: bool) -> str:
    while True:
        raw = input(prompt_text).strip()
        if raw:
            return raw
        if allow_empty:
            return ""
        logging.error("Path value cannot be empty")


def _determine_input_dir(
    script_dir: Path, config: Dict[str, Any], stage: str, non_interactive: bool
) -> Optional[Path]:
    needs_input = stage in ["all", "analyze", "excel", "tag"]
    configured = str(config.get("input_directory", "")).strip()
    if not needs_input and not configured:
        return None

    path_value = configured
    if needs_input and not path_value:
        if non_interactive:
            raise RuntimeError(
                "input_directory is empty; provide it in config.py or via --input-directory"
            )
        path_value = _prompt_for_path("Input directory path: ", allow_empty=False)

    if not path_value:
        return None

    input_dir = _resolve_path(script_dir, path_value)
    if not input_dir.is_dir():
        raise RuntimeError(f"Input directory not found: {input_dir}")
    return input_dir


def _determine_output_base(
    script_dir: Path, config: Dict[str, Any], stage: str, non_interactive: bool
) -> Optional[Path]:
    uses_meta = stage in ["all", "analyze", "excel", "tag"]
    configured = str(config.get("output_directory", "")).strip()
    if not uses_meta and not configured:
        return None

    path_value = configured
    if uses_meta and not path_value:
        if non_interactive:
            return script_dir
        answer = _prompt_for_path(
            "Output base directory (empty to use project root): ",
            allow_empty=True,
        )
        path_value = answer
        if not path_value:
            return script_dir

    if not path_value:
        return None

    output_base = _resolve_path(script_dir, path_value)
    output_base.mkdir(parents=True, exist_ok=True)
    return output_base


def build_runtime_paths(
    script_dir: Path, config: Dict[str, Any], stage: str, non_interactive: bool
) -> RuntimePaths:
    input_dir = _determine_input_dir(script_dir, config, stage, non_interactive)
    output_base = _determine_output_base(script_dir, config, stage, non_interactive)

    meta_root: Optional[Path] = None
    if input_dir and output_base:
        meta_root = output_base / _slugify(input_dir.name)

    if not meta_root:
        raise RuntimeError(
            "Unable to build runtime paths: input_directory and output_directory are required"
        )

    meta_root.mkdir(parents=True, exist_ok=True)
    db_path = meta_root / "tracks.db"
    tracks_json_path = meta_root / "tracks.json"
    excel_path = meta_root / "genres.xlsx"
    report_path = meta_root / "report.md"

    return RuntimePaths(
        input_dir=input_dir,
        output_base=output_base,
        meta_root=meta_root,
        db_path=db_path,
        tracks_json_path=tracks_json_path,
        excel_path=excel_path,
        report_path=report_path,
    )


def find_audio_files(
    directory: Path,
    file_pattern: str = "",
    audio_extensions: Optional[List[str]] = None,
) -> List[Path]:
    audio_files: List[Path] = []
    pattern_lower = file_pattern.lower() if file_pattern else None
    extension_set = {
        str(ext).strip().lower()
        for ext in (audio_extensions or DEFAULT_AUDIO_EXTENSIONS)
        if str(ext).strip()
    }

    for item in directory.rglob("*"):
        if item.is_file() and item.suffix.lower() in extension_set:
            if pattern_lower and pattern_lower not in item.stem.lower():
                continue
            audio_files.append(item)
    audio_files.sort()
    return audio_files


def build_audio_hash(audio_path: Path) -> str:
    try:
        normalized_path = audio_path.resolve().as_posix()
    except Exception:
        normalized_path = str(audio_path).replace("\\", "/")
    return hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()[:16]


def select_audio_windows(
    audio: np.ndarray,
    sample_rate: int,
    window_duration_sec: float,
    positions: Sequence[float],
) -> List[Tuple[float, np.ndarray]]:
    if audio is None or len(audio) == 0:
        raise ValueError("Audio cannot be empty.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    if window_duration_sec <= 0:
        raise ValueError("Window duration must be positive.")
    if len(positions) > 3:
        raise ValueError("at most three window positions are supported.")
    if any(position < 0 or position > 1 for position in positions):
        raise ValueError("Window positions must be between 0 and 1.")

    audio_duration_sec = len(audio) / sample_rate
    if audio_duration_sec <= window_duration_sec:
        return [(0.0, audio)]

    window_samples = max(1, int(window_duration_sec * sample_rate))
    max_start_sec = audio_duration_sec - window_duration_sec
    start_samples = {
        int(
            max(
                0.0,
                min(
                    position * audio_duration_sec - window_duration_sec / 2,
                    max_start_sec,
                ),
            )
            * sample_rate
        )
        for position in positions
    }
    return [
        (start_sample / sample_rate, audio[start_sample : start_sample + window_samples])
        for start_sample in sorted(start_samples)
    ]


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
    except Exception as exc:
        logging.error("ffmpeg conversion error for %s: %s", input_path.name, exc)
        if temp_wav_path and temp_wav_path.exists():
            temp_wav_path.unlink(missing_ok=True)
        return None


def _truncate_two_decimals(value: float) -> float:
    if value <= 0:
        return 0.0
    return float(Decimal(str(value)).quantize(Decimal("0.00"), rounding=ROUND_DOWN))


def _format_duration_hhmmss(total_seconds: float) -> str:
    seconds_int = max(0, int(total_seconds))
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    seconds = seconds_int % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _resolve_audio_duration_seconds(audio_path: Path) -> Optional[float]:
    try:
        audio_file = MutagenFile(str(audio_path))
        if audio_file is None or not hasattr(audio_file, "info"):
            return None
        duration = getattr(audio_file.info, "length", None)
        if duration is None:
            return None
        duration_float = float(duration)
        return duration_float if duration_float > 0 else None
    except Exception:
        return None


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
    model_params: Dict[str, Any], base_dir: Path
) -> Optional[Path]:
    checkpoint_path_raw = model_params.get("checkpoint_path", "")
    if checkpoint_path_raw:
        checkpoint_path = Path(checkpoint_path_raw)
        if not checkpoint_path.is_absolute():
            checkpoint_path = base_dir / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found at configured checkpoint_path: {checkpoint_path}. "
                "Provide a valid MODEL_FILE_PATH or clear it to use the default src/models path."
            )
        return checkpoint_path

    checkpoint_filename = str(model_params.get("checkpoint_filename", "")).strip()
    if not checkpoint_filename:
        return None

    checkpoint_url = str(model_params.get("checkpoint_url", "")).strip()
    checkpoint_dir = base_dir / DEFAULT_MODELS_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_filename

    if checkpoint_path.is_file():
        return checkpoint_path

    if not checkpoint_url:
        raise RuntimeError(
            f"Checkpoint missing at {checkpoint_path} and no checkpoint_url was provided"
        )

    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    logging.info("Checkpoint not found: %s", checkpoint_path)
    logging.info("Downloading checkpoint to %s", checkpoint_path)
    try:
        urllib.request.urlretrieve(checkpoint_url, str(tmp_path))
        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise RuntimeError("Downloaded checkpoint file is empty")
        tmp_path.replace(checkpoint_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download checkpoint from {checkpoint_url}: {exc}"
        ) from exc

    return checkpoint_path


def load_models(config: Dict[str, Any], script_dir: Path) -> Dict[str, Any]:
    if get_maest is None:
        raise RuntimeError(
            "maest_infer is unavailable; install package so get_maest can be imported"
        )

    models: Dict[str, Any] = {"maest": {}}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model_name, model_params in config.get("maest_models", {}).items():
        if not model_params.get("enabled", False):
            continue
        ckpt_path = resolve_checkpoint_path(model_params, script_dir)
        if ckpt_path is None:
            raise RuntimeError("Resolved checkpoint path is empty")
        arch = model_params.get("arch", "discogs-maest-30s-pw-129e-519l")
        model = get_maest(arch, pretrained=False).eval().to(device)
        logging.info("Model loaded on device: %s", device)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
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
            "checkpoint": str(ckpt_path),
            "device": device,
        }

    if not models["maest"]:
        raise RuntimeError("No active MAEST models loaded")
    return models


def process_predictions(
    scores: np.ndarray, labels: List[str], top_n: int
) -> List[Tuple[str, float]]:
    if scores is None or labels is None:
        return []
    scores_np = np.asarray(scores).squeeze()
    if scores_np.ndim != 1:
        scores_np = np.mean(scores_np, axis=0)
    if len(scores_np) != len(labels):
        return []
    idx = np.argsort(scores_np)[-top_n:][::-1]
    return [(labels[index], float(scores_np[index])) for index in idx]


def aggregate_window_predictions(
    window_predictions: Sequence[Tuple[np.ndarray, Sequence[str]]],
) -> Tuple[np.ndarray, List[str]]:
    if not window_predictions:
        raise ValueError("At least one window prediction is required.")

    score_vectors: List[np.ndarray] = []
    common_labels: Optional[List[str]] = None
    for scores, labels in window_predictions:
        score_vector = np.asarray(scores)
        if score_vector.ndim != 1:
            raise ValueError("Window prediction scores must be one-dimensional.")

        label_list = list(labels)
        if len(score_vector) != len(label_list):
            raise ValueError("Window prediction score count must match label count.")
        if common_labels is None:
            common_labels = label_list
        elif label_list != common_labels:
            raise ValueError("Window predictions must use the same label vocabulary.")
        score_vectors.append(score_vector)

    return np.mean(np.stack(score_vectors), axis=0), common_labels or []


def process_labels(raw_labels: List[str]) -> List[str]:
    return [label.split("---", 1)[-1] for label in raw_labels]


def analyze_audio_file(
    original_audio_path: Path,
    models: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    path_to_analyze = original_audio_path
    temp_wav_file: Optional[Path] = None
    needs_cleanup = False

    file_ext = original_audio_path.suffix.lower()
    force_convert = file_ext not in SOUNDFILE_DIRECT_EXTENSIONS
    should_convert = (config["convert_to_wav"] and file_ext != ".wav") or force_convert

    if should_convert:
        temp_wav_file = convert_audio_to_wav(
            original_audio_path,
            config["sample_rate"],
            config["ffmpeg_path"],
            None,
        )
        if temp_wav_file:
            path_to_analyze = temp_wav_file
            needs_cleanup = True
        else:
            raise RuntimeError(
                f"Failed to convert audio file: {original_audio_path}. "
                "Install ffmpeg and ensure this format is supported by your ffmpeg build."
            )

    duration_raw = _resolve_audio_duration_seconds(original_audio_path)
    duration_seconds = _truncate_two_decimals(duration_raw or 0.0)

    try:
        file_size = int(original_audio_path.stat().st_size)
    except Exception:
        file_size = 0
    file_size_mb = _truncate_two_decimals(file_size / (1024.0 * 1024.0))

    json_data: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "hash": build_audio_hash(original_audio_path),
        "file": {
            "path": str(original_audio_path.resolve()),
            "name": original_audio_path.stem,
            "extension": original_audio_path.suffix,
            "size": {
                "megabytes": file_size_mb,
                "bytes": file_size,
            },
            "duration": {
                "time": _format_duration_hhmmss(duration_seconds),
                "seconds": duration_seconds,
            },
        },
        "genres": {
            "labels": [],
            "confidences": [],
            "model": str(config.get("maest_result_key", DEFAULT_MODEL_KEY)),
        },
        "analysis_config": {
            "audio_segment_offsets": [],
            "audio_segment_duration": config["audio_window_duration"],
            "audio_segment_count": 0,
            "aggregation": "mean",
        },
        "_converted_with_ffmpeg": bool(should_convert),
    }

    try:
        audio = load_mono_16k(path_to_analyze, config["sample_rate"])
        if audio is None or len(audio) < config["sample_rate"] * 0.1:
            raise ValueError("Audio is empty or too short")

        if duration_seconds <= 0:
            duration_seconds = _truncate_two_decimals(
                len(audio) / float(config["sample_rate"])
            )
            json_data["file"]["duration"] = {
                "time": _format_duration_hhmmss(duration_seconds),
                "seconds": duration_seconds,
            }

        windows = select_audio_windows(
            audio,
            config["sample_rate"],
            config["audio_window_duration"],
            config["audio_window_positions"],
        )
        offsets = [offset for offset, _ in windows]
        json_data["analysis_config"] = {
            "audio_segment_offsets": offsets,
            "audio_segment_duration": config["audio_window_duration"],
            "audio_segment_count": len(windows),
            "aggregation": "mean",
        }
        best_result: Dict[str, Any] = {
            "labels": [],
            "confidences": [],
            "model": str(config.get("maest_result_key", DEFAULT_MODEL_KEY)),
        }
        for name, maest_data in models.get("maest", {}).items():
            top_n = int(config.get("num_genres", 3) or 3)
            model = maest_data["model"]
            device = maest_data["device"]
            window_predictions: List[Tuple[np.ndarray, Sequence[str]]] = []
            for _, window in windows:
                wav = torch.from_numpy(window).float().to(device)
                with torch.no_grad():
                    scores, labels = model.predict_labels(wav)
                    if device == "cuda":
                        torch.cuda.synchronize()
                window_predictions.append((scores, labels))

            mean_scores, raw_labels = aggregate_window_predictions(window_predictions)
            cleaned_labels = process_labels(raw_labels)
            structured = process_predictions(mean_scores, cleaned_labels, top_n)
            best_result = {
                "labels": [label for label, _ in structured],
                "confidences": [round(score, 4) for _, score in structured],
                "model": name,
            }
        json_data["genres"] = best_result
    except Exception as exc:
        json_data["error"] = str(exc)
    finally:
        if needs_cleanup and temp_wav_file and temp_wav_file.exists():
            temp_wav_file.unlink(missing_ok=True)

    return json_data


def run_analysis_stage(
    config: Dict[str, Any], script_dir: Path, input_dir: Optional[Path], db_path: Path
) -> Dict[str, Any]:
    if input_dir is None:
        raise RuntimeError("Analyze stage requires input directory")

    init_db(db_path)

    all_audio_files = find_audio_files(
        input_dir,
        str(config.get("file_pattern", "")),
        config.get("audio_extensions"),
    )
    if not all_audio_files:
        raise RuntimeError(f"No audio files found in {input_dir}")

    existing_hashes = get_existing_hashes(db_path)
    unprocessed_files = [
        file_path
        for file_path in all_audio_files
        if build_audio_hash(file_path) not in existing_hashes
    ]

    max_files_to_process = int(config.get("max_files", 0) or 0)
    if max_files_to_process > 0:
        files_to_process = unprocessed_files[:max_files_to_process]
    else:
        files_to_process = unprocessed_files

    if not files_to_process:
        logging.info("Resume check: all tracks are already analyzed")
        return {
            "audio_files": len(all_audio_files),
            "processed": 0,
            "errors": 0,
            "skipped_existing": len(all_audio_files),
        }

    models = load_models(config, script_dir)

    processed = 0
    errors = 0
    total_elapsed_seconds = 0.0
    for index, audio_path in enumerate(files_to_process, start=1):
        started_at = perf_counter()
        try:
            result = analyze_audio_file(audio_path, models, config)
            upsert_track(db_path, result)
            elapsed_seconds = perf_counter() - started_at
            total_elapsed_seconds += elapsed_seconds
            if result.get("error"):
                errors += 1
                logging.error(
                    "[%d/%d] Analyze error: %s - %s (time: %.2fs)",
                    index,
                    len(files_to_process),
                    audio_path.name,
                    result.get("error"),
                    elapsed_seconds,
                )
            else:
                processed += 1
                conversion_suffix = ""
                if result.get("_converted_with_ffmpeg"):
                    conversion_suffix = " > .wav [ffmpeg]"
                logging.info(
                    "[%d/%d] Analyzed: %s%s (time: %.2fs)",
                    index,
                    len(files_to_process),
                    audio_path.name,
                    conversion_suffix,
                    elapsed_seconds,
                )
        except Exception as exc:
            errors += 1
            elapsed_seconds = perf_counter() - started_at
            total_elapsed_seconds += elapsed_seconds
            logging.error(
                "[%d/%d] Analyze exception: %s - %s (time: %.2fs)",
                index,
                len(files_to_process),
                audio_path.name,
                exc,
                elapsed_seconds,
            )
            logging.debug(traceback.format_exc())

    files_processed_count = len(files_to_process)
    return {
        "audio_files": len(all_audio_files),
        "processed": processed,
        "errors": errors,
        "skipped_existing": len(all_audio_files) - len(files_to_process),
        "files_attempted": files_processed_count,
        "elapsed_seconds": round(total_elapsed_seconds, 2),
        "average_seconds": round(total_elapsed_seconds / files_processed_count, 2)
        if files_processed_count
        else 0.0,
    }


def run_excel_stage(
    config: Dict[str, Any], db_path: Path, excel_path: Path
) -> Dict[str, Any]:
    if not db_path.is_file():
        raise RuntimeError(f"Database file not found: {db_path}")
    return create_excel_report(db_path, excel_path)


def run_tag_stage(
    config: Dict[str, Any], db_path: Path, excel_path: Path
) -> Dict[str, Any]:
    if not db_path.is_file():
        raise RuntimeError(f"Database file not found: {db_path}")

    tag_stats = run_genre_tagging(excel_path, config["tagger"])
    update_track_statuses(db_path, tag_stats.get("status_updates", []))
    return tag_stats


def _prompt_overwrite() -> bool:
    answer = input("Overwrite existing genre tags? [y/N]: ").strip().lower()
    return answer in ["y", "yes"]


def should_run_tag_stage(
    stage: str,
    tag_mode: str,
    non_interactive: bool,
    config: Dict[str, Any],
) -> bool:
    if stage == "tag":
        return True
    if stage != "all":
        return False
    if tag_mode == "yes":
        if not non_interactive:
            config["tagger"]["overwrite_existing"] = _prompt_overwrite()
        return True
    if tag_mode == "no":
        return False
    if non_interactive:
        return False
    answer = input("Run tagging stage now? [y/N]: ").strip().lower()
    if answer not in ["y", "yes"]:
        return False
    config["tagger"]["overwrite_existing"] = _prompt_overwrite()
    return True


def run_pipeline(
    config: Dict[str, Any], stage: str, script_dir: Path, non_interactive: bool
) -> int:
    config = apply_model_runtime_defaults(config, script_dir)
    runtime_paths = build_runtime_paths(script_dir, config, stage, non_interactive)
    analysis_stats: Optional[Dict[str, Any]] = None
    excel_stats: Optional[Dict[str, Any]] = None
    tag_stats: Optional[Dict[str, Any]] = None
    library_summary: Optional[Dict[str, Any]] = None
    cumulative_timings = load_cumulative_timings(runtime_paths.report_path)

    success = False
    error_text = ""
    report_status = "running"

    def refresh_tracks_json() -> None:
        if not config.get("write_json", False):
            return
        if not runtime_paths.db_path.is_file():
            return
        export_tracks_json(runtime_paths.db_path, runtime_paths.tracks_json_path)

    def write_current_report() -> None:
        current_library_summary = library_summary
        if current_library_summary is None:
            current_library_summary = load_best_available_library_summary(
                runtime_paths.db_path,
                runtime_paths.excel_path,
            )

        write_markdown_report(
            runtime_paths.report_path,
            stage,
            runtime_paths.input_dir,
            runtime_paths.db_path,
            runtime_paths.tracks_json_path if config.get("write_json", False) else None,
            runtime_paths.excel_path,
            analysis_stats,
            excel_stats,
            tag_stats,
            current_library_summary,
            report_status,
            cumulative_timings,
            success,
            error_text,
        )

    try:
        if stage in ["all", "analyze"]:
            stage_started_at = perf_counter()
            analysis_stats = run_analysis_stage(
                config, script_dir, runtime_paths.input_dir, runtime_paths.db_path
            )
            elapsed_seconds = round(perf_counter() - stage_started_at, 2)
            cumulative_timings["analyze_seconds"] += elapsed_seconds
            cumulative_timings["total_runtime_seconds"] += elapsed_seconds
            library_summary = summarize_database_library(runtime_paths.db_path)
            refresh_tracks_json()
            write_current_report()

        if stage in ["all", "excel"]:
            stage_started_at = perf_counter()
            excel_stats = run_excel_stage(
                config, runtime_paths.db_path, runtime_paths.excel_path
            )
            elapsed_seconds = round(perf_counter() - stage_started_at, 2)
            cumulative_timings["excel_seconds"] += elapsed_seconds
            cumulative_timings["total_runtime_seconds"] += elapsed_seconds
            library_summary = excel_stats.get("library_summary")
            write_current_report()

        if should_run_tag_stage(
            stage, config.get("tag_mode", "ask"), non_interactive, config
        ):
            stage_started_at = perf_counter()
            tag_stats = run_tag_stage(
                config, runtime_paths.db_path, runtime_paths.excel_path
            )
            elapsed_seconds = round(perf_counter() - stage_started_at, 2)
            cumulative_timings["tag_seconds"] += elapsed_seconds
            cumulative_timings["total_runtime_seconds"] += elapsed_seconds
            library_summary = summarize_database_library(runtime_paths.db_path)
            refresh_tracks_json()
            write_current_report()

        success = True
        report_status = "completed"
    except Exception as exc:
        error_text = str(exc)
        report_status = "failed"
        logging.error("Pipeline failed: %s", exc)
        logging.debug(traceback.format_exc())
    finally:
        refresh_tracks_json()
        write_current_report()
        logging.info("Report written: %s", runtime_paths.report_path)

    return 0 if success else 1
