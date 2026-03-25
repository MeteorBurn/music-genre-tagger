# Music Genre Tagger

Automatically detect genres for large audio libraries with the MAEST model, write them into file tags, and keep the full analysis state in a local `SQLite` database.

## What it does

- analyzes audio files with [MAEST](https://github.com/palonso/MAEST) via [maest-infer](https://github.com/openmirlab/maest-infer)
- stores per-track analysis results in `tracks.db`
- rebuilds `genres.xlsx` from the database on demand
- optionally exports a combined `tracks.json` snapshot
- writes genre tags into audio file metadata
- refreshes `report.md` after each stage

## Pipeline

```text
Your Music Library
       |
       v
1. ANALYZE
   - scan audio files
   - skip hashes already in tracks.db
   - decode audio and run inference
   - upsert results into tracks.db
   - optionally refresh tracks.json

2. EXCEL
   - read tracks.db
   - fully rebuild genres.xlsx

3. TAG (optional)
   - read genres.xlsx
   - write genre tags into files
   - sync updated status values back to tracks.db
   - optionally refresh tracks.json

4. REPORT
   - summarize tracks.db
   - write report.md
```

## Output structure

```text
<output_directory>/
└── <music_folder_name>/
    ├── tracks.db
    ├── genres.xlsx
    ├── report.md
    └── tracks.json   # optional, skipped with --no-tracks-json
```

## Track payload format

Each database row stores the same track-level analysis payload that is exported back into `tracks.json`:

```json
{
  "timestamp": "2026-03-25T12:34:56.123456",
  "hash": "537667f4e6b49acb",
  "file": {
    "path": "C:\\Music\\track.flac",
    "name": "track",
    "extension": ".flac",
    "size": {
      "megabytes": 42.31,
      "bytes": 44368789
    },
    "duration": {
      "time": "00:05:43",
      "seconds": 343.0
    }
  },
  "genres": {
    "labels": ["Techno", "House", "Electro"],
    "confidences": [0.8213, 0.1104, 0.0403],
    "model": "maest_519l_pytorch"
  },
  "analysis_config": {
    "audio_segment_offset": 60,
    "audio_segment_duration": 30
  }
}
```

If inference fails, an `error` key is added and `labels` / `confidences` stay empty.

## Excel columns

`genres.xlsx` always contains these columns in this order:

| Column | Description |
|---|---|
| `status` | `analysis_success`, `analysis_error`, `tag_success`, `tag_skipped_existing`, `tag_error` |
| `path` | Absolute audio file path |
| `name` | Filename stem |
| `duration` | Track length in `[h]:mm:ss` format |
| `genres` | Top genres, comma-separated |
| `is_broken_beat` | Broken-beat flag |
| `model_key` | Model result key |
| `confidences` | Confidence scores, comma-separated |

## Configuration

Edit `src/config.py`:

```python
INPUT_DIRECTORY = ""
OUTPUT_DIRECTORY = ""

FILE_PATTERN = ""
MAX_FILES = 0

AUDIO_OFFSET = 60
AUDIO_DURATION = 30
SAMPLE_RATE = 16000
NUM_GENRES = 3

CONVERT_TO_WAV = False
FFMPEG_PATH = "ffmpeg"

MODEL_FILE_PATH = ""
MODEL_KEY = ""

OVERWRITE_EXISTING = False
GENRE_SEPARATOR = "; "
MAX_TAG_GENRES = 3

LOG_LEVEL = "INFO"
WRITE_TRACKS_JSON = True
```

## CLI

```bash
python src/main.py [options]
```

| Option | Description |
|---|---|
| `--stage all|analyze|excel|tag` | Run full pipeline or one stage |
| `--input-directory <path>` | Source music library |
| `--output-directory <path>` | Metadata output directory |
| `--file-pattern <text>` | Filter filenames by substring |
| `--max-files <N>` | Limit analyzed tracks |
| `--convert-to-wav` | Force WAV conversion before inference |
| `--tag-yes` | Auto-run tag stage |
| `--tag-no` | Skip tag stage |
| `--non-interactive` | Disable prompts |
| `--loglevel <level>` | Logging level |
| `--no-tracks-json` | Skip `tracks.json` export |

Examples:

```bash
python src/main.py --input-directory "path/to/music" --output-directory "path/to/meta"
python src/main.py --stage analyze --max-files 5 --input-directory "path/to/music"
python src/main.py --stage excel --input-directory "path/to/music" --output-directory "path/to/meta"
python src/main.py --stage tag --input-directory "path/to/music" --output-directory "path/to/meta"
python src/main.py --non-interactive --tag-no --no-tracks-json --input-directory "path/to/music"
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate with `./.venv/Scripts/Activate.ps1`.

## Supported formats

| Extension | Read method | Tag format |
|---|---|---|
| `.flac` | soundfile | Vorbis comment |
| `.wav` | soundfile | ID3 |
| `.aiff`, `.aif` | soundfile | ID3 |
| `.mp3` | soundfile | ID3 |
| `.m4a` | ffmpeg -> temp WAV | MP4 |
| `.dsf` | ffmpeg -> temp WAV | ID3 |
| `.ape` | ffmpeg -> temp WAV | APE |
| `.wv` | ffmpeg -> temp WAV | APE |

## Verification

Run before finishing changes:

```bash
python -m py_compile src/config.py src/environment.py src/pipeline.py src/extractor.py src/tagger.py src/report.py src/main.py src/storage.py
python src/main.py --help
```
