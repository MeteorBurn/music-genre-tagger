#!/usr/bin/env python3

from typing import Any
from typing import Dict


# Run: python src/main.py

# Paths
INPUT_DIRECTORY = ""  # Music library path
OUTPUT_DIRECTORY = ""  # Metadata base path

# Runtime
LOG_LEVEL = "INFO"  # DEBUG|INFO|WARNING|ERROR|CRITICAL
FILE_PATTERN = ""  # Filename substring filter, empty = all
MAX_FILES = 0  # 0 = no limit
CONVERT_TO_WAV = False  # Convert non-WAV before analysis
FFMPEG_PATH = "ffmpeg"  # ffmpeg executable

# Analysis
AUDIO_OFFSET = 60  # Segment start in seconds
AUDIO_DURATION = 30  # Segment length in seconds
SAMPLE_RATE = 16000  # Target sample rate
NUM_GENRES = 3  # Top-N genres per track

# Model
MODELS_DIR = "src/models"  # Auto-download target when needed
MODEL_KEY = "maest_519l_pytorch"  # JSON result key
MODEL_ARCH = "discogs-maest-30s-pw-129e-519l"  # MAEST architecture
CHECKPOINT_PATH = ""  # Optional explicit checkpoint file path
CHECKPOINT_FILENAME = "discogs-maest-30s-pw-129e-519l-swa.ckpt"  # Default checkpoint
CHECKPOINT_URL = "https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e-519l/resolve/main/discogs-maest-30s-pw-129e-519l-swa.ckpt"  # Download URL

# Tagging
OVERWRITE_EXISTING = False  # Keep existing tags
GENRE_SEPARATOR = "; "  # Separator in written tag
MAX_TAG_GENRES = 3  # Max genres written to tag


def get_config() -> Dict[str, Any]:
    return {
        "loglevel": LOG_LEVEL,
        "input_directory": INPUT_DIRECTORY,
        "output_directory": OUTPUT_DIRECTORY,
        "file_pattern": FILE_PATTERN,
        "max_files": MAX_FILES,
        "convert_to_wav": CONVERT_TO_WAV,
        "ffmpeg_path": FFMPEG_PATH,
        "audio_offset": AUDIO_OFFSET,
        "audio_duration": AUDIO_DURATION,
        "sample_rate": SAMPLE_RATE,
        "num_genres": NUM_GENRES,
        "models_dir": MODELS_DIR,
        "maest_result_key": MODEL_KEY,
        "maest_models": {
            MODEL_KEY: {
                "enabled": True,
                "arch": MODEL_ARCH,
                "checkpoint_path": CHECKPOINT_PATH,
                "checkpoint_filename": CHECKPOINT_FILENAME,
                "checkpoint_url": CHECKPOINT_URL,
            }
        },
        "tagger": {
            "genre_source_field": "genres_maest",
            "file_path_field": "file_path",
            "status_field": "status",
            "genre_separator": GENRE_SEPARATOR,
            "max_genres": MAX_TAG_GENRES,
            "overwrite_existing": OVERWRITE_EXISTING,
            "max_rows": None,
        },
    }
