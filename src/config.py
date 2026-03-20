#!/usr/bin/env python3

from pathlib import Path
from typing import Any
from typing import Dict


SCRIPT_DIR = Path(__file__).resolve().parent


CONFIG: Dict[str, Any] = {
    "loglevel": "INFO",
    "input_directory": "",
    "output_directory": "",
    "json_directory": "",
    "excel_path": "",
    "report_path": "",
    "file_pattern": "",
    "max_files": 0,
    "convert_to_wav": False,
    "ffmpeg_path": "ffmpeg",
    "temp_dir": "",
    "audio_offset": 60,
    "audio_duration": 30,
    "sample_rate": 16000,
    "models_dir": "src/models",
    "maest_models": {
        "maest_519l_pytorch": {
            "enabled": True,
            "arch": "discogs-maest-30s-pw-129e-519l",
            "checkpoint_path": "",
            "checkpoint_filename": "discogs-maest-30s-pw-129e-519l-swa.ckpt",
            "checkpoint_url": "https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e-519l/resolve/main/discogs-maest-30s-pw-129e-519l-swa.ckpt",
            "num_genres": 3,
        }
    },
    "maest_result_key": "maest_519l_pytorch",
    "tagger": {
        "genre_source_field": "genres_maest",
        "file_path_field": "file_path",
        "status_field": "status",
        "genre_separator": "; ",
        "max_genres": 3,
        "overwrite_existing": False,
        "max_rows": None,
    },
}
