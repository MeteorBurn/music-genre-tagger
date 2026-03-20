#!/usr/bin/env python3

import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import traceback
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from extractor import create_excel_report
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

DEFAULT_MODELS_DIR = "src/models"
DEFAULT_MODEL_KEY = "maest_519l_pytorch"
DEFAULT_MODEL_ARCH = "discogs-maest-30s-pw-129e-519l"
DEFAULT_CHECKPOINT_FILENAME = "discogs-maest-30s-pw-129e-519l-swa.ckpt"
DEFAULT_CHECKPOINT_URL = "https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e-519l/resolve/main/discogs-maest-30s-pw-129e-519l-swa.ckpt"


@dataclass
class RuntimePaths:
    input_dir: Optional[Path]
    output_base: Optional[Path]
    meta_root: Optional[Path]
    json_dir: Path
    excel_path: Path
    report_path: Path


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def setup_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)-8s - %(message)s",
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

    runtime["models_dir"] = DEFAULT_MODELS_DIR
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
            return script_dir / "meta"
        answer = _prompt_for_path(
            "Output base directory (empty to use project relative meta): ",
            allow_empty=True,
        )
        path_value = answer
        if not path_value:
            return script_dir / "meta"

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

    json_dir = meta_root / "json"
    excel_path = meta_root / "tracks_genres.xlsx"
    report_path = meta_root / "report.md"

    json_dir.parent.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(
        input_dir=input_dir,
        output_base=output_base,
        meta_root=meta_root,
        json_dir=json_dir,
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


def build_audio_json_key(audio_path: Path) -> str:
    try:
        normalized_path = audio_path.resolve().as_posix()
    except Exception:
        normalized_path = str(audio_path).replace("\\", "/")
    path_hash = hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()[:16]
    return f"{audio_path.stem}__{path_hash}"


def get_existing_json_stems(directory: Path) -> Set[str]:
    json_stems: Set[str] = set()
    if not directory.is_dir():
        return json_stems

    for item in directory.glob("*.json"):
        if item.is_file():
            json_stems.add(item.stem)
            if "__" in item.stem:
                continue
            try:
                data = json.loads(item.read_text(encoding="utf-8"))
                file_path_data = data.get("file_path", {})
                win_path = file_path_data.get("win", "")
                if win_path:
                    json_stems.add(build_audio_json_key(Path(win_path)))
            except Exception:
                continue
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
    except Exception as exc:
        logging.error("ffmpeg conversion error for %s: %s", input_path.name, exc)
        if temp_wav_path and temp_wav_path.exists():
            temp_wav_path.unlink(missing_ok=True)
        return None


def get_platform_paths(file_path: Path) -> Dict[str, str]:
    win_path = str(file_path)
    wsl_path = ""
    linux_path = file_path.as_posix()

    try:
        resolved = file_path.resolve()
        parts = resolved.parts
        if len(parts) >= 2:
            drive_raw = parts[0].replace(":", "")
            if len(drive_raw) == 1 and drive_raw.isalpha():
                tail = "/".join(parts[1:]).replace("\\", "/")
                wsl_path = f"/mnt/{drive_raw.lower()}/{tail}"
    except Exception:
        wsl_path = ""

    return {
        "win": win_path,
        "wsl": wsl_path,
        "linux": linux_path,
    }


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
    model_params: Dict[str, Any], models_dir: Path, base_dir: Path
) -> Optional[Path]:
    checkpoint_path_raw = model_params.get("checkpoint_path", "")
    if checkpoint_path_raw:
        checkpoint_path = Path(checkpoint_path_raw)
        if not checkpoint_path.is_absolute():
            checkpoint_path = base_dir / checkpoint_path
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
    if checkpoint_path.is_file():
        return checkpoint_path

    checkpoint_url = str(model_params.get("checkpoint_url", "")).strip()
    if checkpoint_url:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Checkpoint not found at %s", checkpoint_path)
        logging.info("Downloading checkpoint from %s", checkpoint_url)
        try:
            urllib.request.urlretrieve(checkpoint_url, str(checkpoint_path))
        except Exception as exc:
            raise RuntimeError(
                f"Unable to download checkpoint from {checkpoint_url}: {exc}"
            ) from exc
        logging.info("Checkpoint downloaded to %s", checkpoint_path)
        return checkpoint_path

    logging.warning(
        "Checkpoint not found at %s, using maest_infer pretrained mode",
        checkpoint_path,
    )
    return None


def load_models(config: Dict[str, Any], script_dir: Path) -> Dict[str, Any]:
    if get_maest is None:
        raise RuntimeError(
            "maest_infer is unavailable; install package so get_maest can be imported"
        )

    models: Dict[str, Any] = {"maest": {}}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_base_path = _resolve_path(
        script_dir, config.get("models_dir", DEFAULT_MODELS_DIR)
    )

    for model_name, model_params in config.get("maest_models", {}).items():
        if not model_params.get("enabled", False):
            continue
        ckpt_path = resolve_checkpoint_path(model_params, models_base_path, script_dir)
        arch = model_params.get("arch", "discogs-maest-30s-pw-129e-519l")
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


def process_labels(raw_labels: List[str]) -> List[str]:
    return [label.split("---", 1)[-1] for label in raw_labels]


def analyze_audio_file(
    original_audio_path: Path,
    models: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
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

    json_data: Dict[str, Any] = {
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
            raise ValueError("Audio is empty or too short")

        audio = trim_audio_segment(
            audio,
            config["sample_rate"],
            config["audio_offset"],
            config["audio_duration"],
        )
        if audio is None or len(audio) < config["sample_rate"] * 0.1:
            raise ValueError("Audio segment is too short after trimming")

        audio_tensor = torch.from_numpy(audio).float()
        all_results: Dict[str, Any] = {}
        for name, maest_data in models.get("maest", {}).items():
            top_n = int(config.get("num_genres", 3) or 3)
            model = maest_data["model"]
            device = maest_data["device"]
            wav = audio_tensor.to(device)

            with torch.no_grad():
                scores, labels = model.predict_labels(wav)
                if device == "cuda":
                    torch.cuda.synchronize()

            cleaned_labels = process_labels(list(labels))
            structured = process_predictions(np.asarray(scores), cleaned_labels, top_n)
            all_results[name] = {
                "labels": [label for label, _ in structured],
                "confidences": [round(score, 4) for _, score in structured],
            }
        json_data["analysis_results"] = all_results
    except Exception as exc:
        json_data["error"] = str(exc)
    finally:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_filename = output_dir / f"{build_audio_json_key(original_audio_path)}.json"
        json_filename.write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2, cls=NumpyEncoder),
            encoding="utf-8",
        )
        if needs_cleanup and temp_wav_file and temp_wav_file.exists():
            temp_wav_file.unlink(missing_ok=True)

    return json_data


def run_analysis_stage(
    config: Dict[str, Any], script_dir: Path, input_dir: Optional[Path], json_dir: Path
) -> Dict[str, int]:
    if input_dir is None:
        raise RuntimeError("Analyze stage requires input directory")

    all_audio_files = find_audio_files(
        input_dir,
        str(config.get("file_pattern", "")),
        config.get("audio_extensions"),
    )
    if not all_audio_files:
        raise RuntimeError(f"No audio files found in {input_dir}")

    existing_stems = get_existing_json_stems(json_dir)
    unprocessed_files = [
        file_path
        for file_path in all_audio_files
        if build_audio_json_key(file_path) not in existing_stems
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
    for index, audio_path in enumerate(files_to_process, start=1):
        try:
            result = analyze_audio_file(audio_path, models, config, json_dir)
            if result.get("error"):
                errors += 1
                logging.error(
                    "[%d/%d] Analyze error: %s - %s",
                    index,
                    len(files_to_process),
                    audio_path.name,
                    result.get("error"),
                )
            else:
                processed += 1
                logging.info(
                    "[%d/%d] Analyzed: %s",
                    index,
                    len(files_to_process),
                    audio_path.name,
                )
        except Exception as exc:
            errors += 1
            logging.error(
                "[%d/%d] Analyze exception: %s - %s",
                index,
                len(files_to_process),
                audio_path.name,
                exc,
            )
            logging.debug(traceback.format_exc())

    return {
        "audio_files": len(all_audio_files),
        "processed": processed,
        "errors": errors,
        "skipped_existing": len(all_audio_files) - len(files_to_process),
    }


def run_excel_stage(
    config: Dict[str, Any], json_dir: Path, excel_path: Path
) -> Dict[str, int]:
    if not json_dir.is_dir():
        raise RuntimeError(f"JSON directory not found: {json_dir}")
    return create_excel_report(json_dir, excel_path, config["maest_result_key"])


def run_tag_stage(config: Dict[str, Any], excel_path: Path) -> Dict[str, int]:
    return run_genre_tagging(excel_path, config["tagger"])


def should_run_tag_stage(stage: str, tag_mode: str, non_interactive: bool) -> bool:
    if stage == "tag":
        return True
    if stage != "all":
        return False
    if tag_mode == "yes":
        return True
    if tag_mode == "no":
        return False
    if non_interactive:
        return False
    answer = input("Run tagging stage now? [y/N]: ").strip().lower()
    return answer in ["y", "yes"]


def write_markdown_report(
    report_path: Path,
    stage: str,
    runtime_paths: RuntimePaths,
    analysis_stats: Optional[Dict[str, int]],
    excel_stats: Optional[Dict[str, int]],
    tag_stats: Optional[Dict[str, int]],
    success: bool,
    error_text: str,
) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# MusicTagger Report",
        "",
        f"- timestamp: {timestamp}",
        f"- stage: {stage}",
        f"- success: {str(success).lower()}",
        f"- input_dir: {runtime_paths.input_dir if runtime_paths.input_dir else ''}",
        f"- json_dir: {runtime_paths.json_dir}",
        f"- excel_path: {runtime_paths.excel_path}",
        f"- report_path: {report_path}",
    ]

    if error_text:
        lines.extend(["", "## Error", "", error_text])

    if analysis_stats is not None:
        lines.extend(
            [
                "",
                "## Analyze",
                "",
                f"- audio_files: {analysis_stats.get('audio_files', 0)}",
                f"- processed: {analysis_stats.get('processed', 0)}",
                f"- skipped_existing: {analysis_stats.get('skipped_existing', 0)}",
                f"- errors: {analysis_stats.get('errors', 0)}",
            ]
        )

    if excel_stats is not None:
        lines.extend(
            [
                "",
                "## Excel",
                "",
                f"- json_files: {excel_stats.get('json_files', 0)}",
                f"- rows_added: {excel_stats.get('rows_added', 0)}",
                f"- rows_total: {excel_stats.get('rows_total', 0)}",
            ]
        )

    if tag_stats is not None:
        lines.extend(
            [
                "",
                "## Tag",
                "",
                f"- success: {tag_stats.get('success', 0)}",
                f"- skipped: {tag_stats.get('skipped', 0)}",
                f"- error: {tag_stats.get('error', 0)}",
                f"- already_processed: {tag_stats.get('already_processed', 0)}",
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    config: Dict[str, Any], stage: str, script_dir: Path, non_interactive: bool
) -> int:
    config = apply_model_runtime_defaults(config, script_dir)
    runtime_paths = build_runtime_paths(script_dir, config, stage, non_interactive)
    analysis_stats: Optional[Dict[str, int]] = None
    excel_stats: Optional[Dict[str, int]] = None
    tag_stats: Optional[Dict[str, int]] = None

    success = False
    error_text = ""

    try:
        if stage in ["all", "analyze"]:
            analysis_stats = run_analysis_stage(
                config, script_dir, runtime_paths.input_dir, runtime_paths.json_dir
            )

        if stage in ["all", "excel"]:
            excel_stats = run_excel_stage(
                config, runtime_paths.json_dir, runtime_paths.excel_path
            )

        if should_run_tag_stage(stage, config.get("tag_mode", "ask"), non_interactive):
            tag_stats = run_tag_stage(config, runtime_paths.excel_path)

        success = True
    except Exception as exc:
        error_text = str(exc)
        logging.error("Pipeline failed: %s", exc)
        logging.debug(traceback.format_exc())
    finally:
        write_markdown_report(
            runtime_paths.report_path,
            stage,
            runtime_paths,
            analysis_stats,
            excel_stats,
            tag_stats,
            success,
            error_text,
        )
        logging.info("Report written: %s", runtime_paths.report_path)

    return 0 if success else 1
