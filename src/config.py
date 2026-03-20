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
AUDIO_EXTENSIONS = [
    ".flac",
    ".wav",
    ".aiff",
    ".aif",
    ".m4a",
    ".dsf",
    ".ape",
    ".wv",
    ".mp3",
]  # Formats included in analysis; remove an extension to exclude that format

# Model
MODEL_FILE_PATH = ""  # Optional custom checkpoint file path
MODEL_KEY = ""  # Optional custom result key (used only with MODEL_FILE_PATH)

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
        "audio_extensions": AUDIO_EXTENSIONS,
        "num_genres": NUM_GENRES,
        "model_file_path": MODEL_FILE_PATH,
        "model_key": MODEL_KEY,
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
