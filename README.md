# 🎵 Music Genre Tagger

> Automatically detect and tag music genres in large audio libraries using the **MAEST** deep learning model — trained on millions of tracks from the Discogs database.

No more manual tagging. Point it at your music folder, and every track gets its genre written directly into the file metadata — ready for Rekordbox, Traktor, Serato, or any music player.

https://github.com/user-attachments/assets/ae40e272-0067-4a59-93cf-8f9d62e8e763

---

## 🎯 Who is this for?

| Audience | Why it helps |
|----------|-------------|
| 🎧 **DJs** | Auto-tag your entire library so Rekordbox / Traktor / Serato can filter by genre |
| 📦 **Music archivists** | Bring structure to thousands of untagged FLAC/WAV files in one run |
| 🔬 **Researchers** | Get structured genre metadata (Excel + JSON) across large collections |
| 🛠️ **Developers** | Working example of end-to-end audio ML inference with metadata writing |

---

## ✨ What this project does

**Music Genre Tagger** is an automated ML pipeline that:

- 🔍 **Analyzes** each track with the [MAEST](https://github.com/palonso/MAEST) transformer model (trained on Discogs) and predicts the top-N genres with confidence scores — inference is powered by [maest-infer](https://github.com/openmirlab/maest-infer), a lightweight PyTorch-native repackaging of MAEST
- 📊 **Aggregates** all results into `tracks_genres.xlsx` — with genre names, confidence scores, duration, and a broken-beat flag for DJ-friendly rhythm filtering
- 🏷️ **Writes** genres directly into audio file metadata — ID3 for MP3/WAV, Vorbis comments for FLAC, MP4 tags, and APE tags
- 📋 **Generates** a cumulative `report.md` after every run — refreshed after each stage with library stats, broken beat summary, and timing totals

**Key properties:**

- ⚡ **Incremental** — already-processed tracks are skipped on reruns, only new files are analyzed
- 🚀 **Fast** — with CUDA, inference takes 0.25–1 second per track, making large libraries practical to process
- 🛡️ **Non-destructive by default** — existing genre tags are never overwritten unless you explicitly allow it
- 🌐 **Cross-platform** — runs on Windows, Linux, WSL; GPU (CUDA) auto-detected and used when available
- 🎼 **Format-wide** — FLAC, WAV, AIFF, MP3, M4A, DSF, APE, WavPack

---

## ⚙️ How it works

```
Your Music Library
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  1. ANALYZE                                         │
│  For each unprocessed track:                        │
│  • Convert to 16kHz mono (ffmpeg if needed)         │
│  • Trim to analysis window (default: 60s–90s)       │
│  • Run MAEST inference → top-N genres + confidence  │
│  • Save result to per-track JSON                    │
└───────────────────────┬─────────────────────────────┘
                        │  report.md updated ✓
                        ▼
┌─────────────────────────────────────────────────────┐
│  2. EXCEL                                           │
│  • Read all JSON files                              │
│  • Deduplicate against existing rows                │
│  • Write / update  tracks_genres.xlsx               │
└───────────────────────┬─────────────────────────────┘
                        │  report.md updated ✓
                        ▼
┌─────────────────────────────────────────────────────┐
│  3. TAG  (optional)                                 │
│  • Read tracks_genres.xlsx                          │
│  • Write genre string into audio file metadata      │
│  • Update status column (incremental-safe)          │
└───────────────────────┬─────────────────────────────┘
                        │  report.md updated ✓
                        ▼
                   report.md
```

---

## 📁 Project structure

```
music-genre-tagger/
├── src/
│   ├── main.py          # Entrypoint & CLI
│   ├── pipeline.py      # Core flow orchestration
│   ├── environment.py   # Environment and dependency checks
│   ├── extractor.py     # JSON → Excel extraction
│   ├── tagger.py        # Audio tag writing
│   ├── report.py        # Report generation and library summaries
│   └── config.py        # User configuration
├── requirements.txt
├── README.md
└── AGENTS.md
```

---

## 🚀 Setup

**Step 1 — Create virtual environment:**

```powershell
# Windows (PowerShell)
cd path\to\music-genre-tagger
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / WSL / macOS
cd path/to/music-genre-tagger
python -m venv .venv
source .venv/bin/activate
```

**Step 2 — Install dependencies:**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 💡 **GPU note:** On first run, the pipeline automatically detects your NVIDIA GPU and installs a matching CUDA torch build (`cu121`). After install, rerun `python src/main.py` once to continue with updated packages.

---

## ▶️ Quick start

```bash
python src/main.py
```

The pipeline will:
1. Check environment and dependencies
2. Ask for input/output paths if not set in `config.py`
3. Analyze all audio tracks → write per-track JSON files
4. Aggregate JSON into `tracks_genres.xlsx`
5. Optionally write genres into audio file tags
6. Generate `report.md`

---

## 📂 Output structure

```
<output_directory>/
└── <music_folder_name>/
    ├── json/
    │   ├── track_name__hash.json   ← per-track inference result
    │   └── ...
    ├── tracks_genres.xlsx          ← aggregated genre table
    └── report.md                   ← cumulative run and library summary
```

> If `output_directory` is not set, the project root is used as the base.

---

## 📋 Report

`report.md` is rewritten after each completed stage and accumulates timing totals across runs for the same output folder.

```md
# MusicTagger Report

## Run Overview
- timestamp: 2026-03-25T22:41:18
- status: completed
- stage: all
- total_runtime_seconds: 1842.55    ← cumulative across all runs
- analyze_seconds: 1710.30          ← cumulative across all runs
- excel_seconds: 84.15              ← cumulative across all runs
- tag_seconds: 48.10                ← cumulative across all runs

## Analyze
- audio_files_found: 1240
- processed_now: 620
- skipped_existing: 620
- analysis_errors_now: 3

## Excel
- json_files_seen: 1240
- rows_added_now: 620
- rows_total: 1240

## Tag
- status: completed
- tag_success_now: 605
- tag_skipped_existing_now: 12
- tag_error_now: 3
- already_processed_now: 620

## Library Summary
- total_tracks: 1240
- total_duration: 97:24:18
- average_duration: 00:04:43
- top_genres: Techno (428), House (301), Electro (97)
- extensions: .flac (812), .mp3 (241), .wav (96)

## Broken Beat Summary
- broken_beat_tracks: 186
- non_broken_beat_tracks: 1054
- broken_beat_share: 15.00%

## Status Breakdown
- statuses: tag_success (1225), tag_skipped_existing (12), tag_error (3)
```

---

## 🗂️ Per-track JSON format

Each track produces a JSON file with file metadata and genre predictions:

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
      "seconds": 343.00
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

If inference fails, an `"error"` key is added at the top level and `genres.labels` / `genres.confidences` remain empty lists.

`hash` is the first 16 characters of the SHA-1 of the resolved file path and matches the suffix used in the JSON filename (`track_name__hash.json`).

---

## 📊 Excel columns

`tracks_genres.xlsx` contains one row per track with these columns:

| Column | Description |
|--------|-------------|
| `status` | Pipeline status: `analysis_success`, `analysis_error`, `tag_success`, `tag_skipped_existing`, `tag_error` |
| `path` | Absolute path to the audio file |
| `name` | Filename without extension |
| `duration` | Track length in `[h]:mm:ss` format — sortable |
| `genres` | Top genres, comma-separated |
| `is_broken_beat` | `True` / `False` — broken-beat rhythm flag for DJ filtering |
| `model_key` | Model that produced the prediction |
| `confidences` | Confidence scores per genre, comma-separated |

The table has styled headers (dark blue), auto-sized columns, column filters, and a frozen header row.

---

## ⚙️ Configuration

Edit `src/config.py` to customize behavior:

```python
# Paths
INPUT_DIRECTORY = ""   # Path to your music library
OUTPUT_DIRECTORY = ""  # Path to metadata output folder

# Analysis
NUM_GENRES = 3         # Top-N genres per track
AUDIO_OFFSET = 60      # Start of analysis window (seconds)
AUDIO_DURATION = 30    # Length of analysis window (seconds)
AUDIO_EXTENSIONS = [".flac", ".wav", ".aiff", ".aif", ".m4a", ".dsf", ".ape", ".wv", ".mp3"]

# Model
MODEL_FILE_PATH = ""   # Custom checkpoint path (optional, auto-downloaded if empty)
MODEL_KEY = ""         # Custom result key (used only with MODEL_FILE_PATH)

# Tagging
OVERWRITE_EXISTING = False  # Keep existing genre tags (safe default)
MAX_TAG_GENRES = 3          # Max genres written to tags
GENRE_SEPARATOR = "; "      # Separator between genres in the tag string
```

**Model rules:**
- `MODEL_FILE_PATH` empty → default MAEST checkpoint auto-downloaded to `src/models/`
- `MODEL_FILE_PATH` set → your checkpoint is used instead
- `MODEL_KEY` is only relevant when using a custom `MODEL_FILE_PATH`

---

## 💻 CLI reference

```bash
python src/main.py [options]
```

| Option | Description |
|--------|-------------|
| `--stage all\|analyze\|excel\|tag` | Run the full pipeline or a single stage (default: `all`) |
| `--input-directory <path>` | Path to source music library |
| `--output-directory <path>` | Path for metadata output |
| `--file-pattern <text>` | Filter filenames by substring |
| `--max-files <N>` | Limit number of tracks analyzed (useful for testing) |
| `--convert-to-wav` | Force WAV conversion before inference |
| `--tag-yes` | Auto-confirm genre tagging without prompt |
| `--tag-no` | Auto-skip genre tagging without prompt |
| `--non-interactive` | Disable all prompts (for CI/automation) |
| `--loglevel <level>` | Verbosity: `DEBUG\|INFO\|WARNING\|ERROR\|CRITICAL` |

**Example commands:**

```bash
# Full run with explicit paths
python src/main.py --input-directory "path/to/music" --output-directory "path/to/meta"

# Analyze only first 5 tracks (smoke test)
python src/main.py --stage analyze --input-directory "path/to/music" --max-files 5

# Rebuild Excel from existing JSON without re-running inference
python src/main.py --stage excel --input-directory "path/to/music" --output-directory "path/to/meta"

# Write tags only (after Excel is ready)
python src/main.py --stage tag --input-directory "path/to/music" --output-directory "path/to/meta"

# Non-interactive CI mode, skip tagging
python src/main.py --non-interactive --tag-no --input-directory "path/to/music"
```

---

## 🎼 Supported audio formats

| Format | Read via | Tag format written |
|--------|----------|--------------------|
| `.flac` | Direct | Vorbis comment |
| `.wav` | Direct | ID3 |
| `.aiff` / `.aif` | Direct | ID3 |
| `.mp3` | Direct | ID3 |
| `.m4a` | ffmpeg | MP4 (`©gen`) |
| `.dsf` | ffmpeg | ID3 |
| `.ape` | ffmpeg | APE tag |
| `.wv` | ffmpeg | APE tag |

> Formats requiring **ffmpeg** need `ffmpeg` available in your system PATH. If unavailable, those tracks are skipped with a per-file error while the rest of the pipeline continues normally.

---

## 🙏 Credits

This project is built on top of the MAEST model and ecosystem:

- **[MAEST](https://github.com/palonso/MAEST)** — Music Audio Efficient Spectrogram Transformer, the original research model by Pablo Alonso-Jiménez and colleagues at the Music Technology Group (MTG), Universitat Pompeu Fabra. Trained on the Discogs dataset for multi-label music genre classification.

- **[maest-infer](https://github.com/openmirlab/maest-infer)** — A lightweight, dependency-minimal repackaging of MAEST focused solely on inference, with native PyTorch and torchaudio support. This is the package used directly by Music Genre Tagger.
