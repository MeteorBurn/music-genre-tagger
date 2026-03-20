# 🎵 PyTorch MAEST Tagger

> Automatic genre detection and metadata tagging for music libraries using the MAEST deep learning model.

---

## 🧠 What this project does

This pipeline analyzes your music library, detects genres for each track using the **MAEST** model (PyTorch), exports results to Excel, and optionally writes genre tags directly into audio files.

**Why this is useful:**

- ⏱️ Saves hours of manual genre tagging for large libraries
- 🎯 Consistent genre predictions from the same model across all tracks
- 📊 Creates structured metadata (`json`, `tracks_genres.xlsx`, `report.md`) easy to review
- 🎧 Writes genres into audio tags so DJ software and music players can filter by genre
- 🔄 Supports incremental reruns — only new tracks are analyzed on next run

---

## 🗂️ Project structure

```
project/
├── src/
│   ├── main.py          # Entrypoint
│   ├── pipeline.py      # Core flow orchestration
│   ├── environment.py   # Environment and dependency checks
│   ├── extractor.py     # JSON → Excel extraction
│   ├── tagger.py        # Audio tag writing
│   └── config.py        # User configuration
├── requirements.txt
├── README.md
└── AGENTS.md
```

---

## ⚙️ Setup

**Step 1 — Create virtual environment:**

```powershell
# Windows (PowerShell)
cd path\to\project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / WSL / macOS
cd path/to/project
python -m venv .venv
source .venv/bin/activate
```

**Step 2 — Install dependencies:**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 🚀 **GPU note:** On first run, startup automatically detects your NVIDIA GPU and installs a matching CUDA torch build (`cu121`). After install, rerun `python src/main.py` once to continue with updated packages.

---

## 🚀 Quick start

```bash
python src/main.py
```

The pipeline will:
1. Check environment and dependencies
2. Ask for input/output paths if not set in `config.py`
3. Analyze all audio tracks → write JSON files
4. Aggregate JSON into `tracks_genres.xlsx`
5. Optionally write genres into audio tags
6. Generate `report.md`

---

## 🔁 Pipeline flow

| Stage | What it does |
|-------|-------------|
| **Environment** | Checks Python, dependencies, GPU/CPU, model checkpoint |
| **Analyze** | Scans audio files, runs MAEST inference, writes per-track JSON |
| **Excel** | Aggregates JSON files into `tracks_genres.xlsx` with deduplication |
| **Tag** | Reads Excel and writes genres into audio file metadata tags |
| **Report** | Generates `report.md` summary in the metadata folder |

---

## 📁 Output structure

For each music library, metadata is stored in a dedicated folder:

```
<output_directory>/
└── <music_folder_name>/
    ├── json/
    │   ├── track_name__hash.json
    │   └── ...
    ├── tracks_genres.xlsx
    └── report.md
```

> If `output_directory` is not set, project root is used as base.

---

## 🔧 Configuration

Edit `src/config.py` to customize behavior:

```python
# Paths
INPUT_DIRECTORY = ""   # Path to your music library
OUTPUT_DIRECTORY = ""  # Path to metadata output folder

# Analysis
NUM_GENRES = 3         # Top-N genres per track
AUDIO_EXTENSIONS = [".flac", ".wav", ".aiff", ".aif", ".m4a", ".dsf", ".ape", ".wv", ".mp3"]

# Model
MODEL_FILE_PATH = ""   # Custom checkpoint path (optional)
MODEL_KEY = ""         # Custom result key (used only with MODEL_FILE_PATH)

# Tagging
OVERWRITE_EXISTING = False  # Keep existing genre tags
MAX_TAG_GENRES = 3          # Max genres written to tags
```

**Model rules:**
- `MODEL_FILE_PATH` empty → default checkpoint auto-downloaded to `src/models/`
- `MODEL_FILE_PATH` set → your checkpoint is used
- `MODEL_KEY` is only used with a custom `MODEL_FILE_PATH`

---

## 💻 CLI reference

```bash
python src/main.py [options]
```

| Option | Description |
|--------|-------------|
| `--stage all\|analyze\|excel\|tag` | Pipeline scope (default: `all`) |
| `--input-directory <path>` | Source music library path |
| `--output-directory <path>` | Metadata output base path |
| `--file-pattern <text>` | Filter filenames by substring |
| `--max-files <N>` | Limit tracks analyzed (test mode) |
| `--convert-to-wav` | Force WAV conversion before inference |
| `--tag-yes` | Auto-confirm tagging without prompt |
| `--tag-no` | Auto-skip tagging without prompt |
| `--non-interactive` | Disable all prompts |
| `--loglevel <level>` | Verbosity: `DEBUG\|INFO\|WARNING\|ERROR\|CRITICAL` |

**Example commands:**

```bash
# Full run with explicit paths
python src/main.py --input-directory "path/to/music" --output-directory "path/to/meta"

# Analyze only first 5 tracks (smoke test)
python src/main.py --stage analyze --input-directory "path/to/music" --max-files 5

# Rebuild Excel from existing JSON
python src/main.py --stage excel --input-directory "path/to/music" --output-directory "path/to/meta"

# Tag only (after Excel is ready)
python src/main.py --stage tag --input-directory "path/to/music" --output-directory "path/to/meta"

# Non-interactive CI mode, skip tagging
python src/main.py --non-interactive --tag-no --input-directory "path/to/music"
```

---

## 🎵 Supported audio formats

| Format | Notes |
|--------|-------|
| `.flac` | Direct |
| `.wav` | Direct |
| `.aiff` / `.aif` | Direct |
| `.mp3` | Direct |
| `.m4a` | Via ffmpeg |
| `.dsf` | Via ffmpeg |
| `.ape` | Via ffmpeg |
| `.wv` | Via ffmpeg |

> ⚠️ Formats marked **Via ffmpeg** require `ffmpeg` in your system PATH. If unavailable, those tracks are skipped with a per-file error while the pipeline continues.

---

## 🔒 Invariants

The following behaviors are preserved and must not be changed:

- MAEST result key: `maest_519l_pytorch`
- Label cleanup: `A---B → B`
- Extractor fields: `file_path`, `file_name`, `genres_maest`, `confidences`, `is_broken_beat`, `model_key`
- Tag safety default: `overwrite_existing = False`
