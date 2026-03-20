#!/usr/bin/env python3

from typing import Any
from typing import Dict


# -----------------------------------------------------------------------------
# Project Configuration
# -----------------------------------------------------------------------------
# 1) Fill values below.
# 2) Run: python src/main.py
# 3) If INPUT_DIRECTORY or OUTPUT_DIRECTORY are empty, CLI will ask interactively
#    (unless --non-interactive is used).


# INPUT_DIRECTORY: path to the source music library for analysis.
# Example: "path/to/music_library_demo"
INPUT_DIRECTORY = ""

# OUTPUT_DIRECTORY: path to the base metadata directory.
# Example: "path/to/output_meta_demo"
# Final structure is created automatically:
# <OUTPUT_DIRECTORY>/<input_folder_name>/json
# <OUTPUT_DIRECTORY>/<input_folder_name>/tracks_genres.xlsx
# <OUTPUT_DIRECTORY>/<input_folder_name>/report.md
OUTPUT_DIRECTORY = ""

# LOG_LEVEL: logging verbosity.
# Allowed: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"

# FILE_PATTERN: optional substring filter for filenames during analysis.
# Empty value means process all supported audio files.
FILE_PATTERN = ""

# MAX_FILES: max number of tracks to analyze in one run.
# 0 means no limit (process all pending tracks).
MAX_FILES = 0

# CONVERT_TO_WAV: convert non-WAV files to temporary WAV before analysis.
CONVERT_TO_WAV = False

# FFMPEG_PATH: ffmpeg executable name or absolute path.
FFMPEG_PATH = "ffmpeg"

# AUDIO_OFFSET: start position (seconds) for analysis segment.
AUDIO_OFFSET = 60

# AUDIO_DURATION: segment length (seconds) for analysis.
AUDIO_DURATION = 30

# SAMPLE_RATE: target sample rate used for model input.
SAMPLE_RATE = 16000

# NUM_GENRES: number of top genres to save per track.
NUM_GENRES = 3

# MODELS_DIR: local directory for model checkpoints.
# If CHECKPOINT_PATH is empty and checkpoint is missing, it is downloaded here.
MODELS_DIR = "src/models"

# MODEL_KEY: result key stored in JSON and used by extractor.
MODEL_KEY = "maest_519l_pytorch"

# MODEL_ARCH: MAEST architecture name.
MODEL_ARCH = "discogs-maest-30s-pw-129e-519l"

# CHECKPOINT_PATH: optional custom checkpoint file path.
# If empty, pipeline uses MODELS_DIR/CHECKPOINT_FILENAME.
CHECKPOINT_PATH = ""

# CHECKPOINT_FILENAME: default checkpoint filename.
CHECKPOINT_FILENAME = "discogs-maest-30s-pw-129e-519l-swa.ckpt"

# CHECKPOINT_URL: source URL for automatic checkpoint download.
CHECKPOINT_URL = "https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e-519l/resolve/main/discogs-maest-30s-pw-129e-519l-swa.ckpt"

# OVERWRITE_EXISTING: if False, do not overwrite existing genre tags.
OVERWRITE_EXISTING = False

# GENRE_SEPARATOR: separator used when writing multiple genres to tags.
GENRE_SEPARATOR = "; "

# MAX_TAG_GENRES: max number of genres written to tags.
MAX_TAG_GENRES = 3


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
