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
| 💻 **Developers** | Working example of end-to-end audio ML inference with metadata writing |

---

## 🧠 What this project does

**Music Genre Tagger** is an automated ML pipeline that:

- 🔍 **Analyzes** each track with the [MAEST](https://github.com/palonso/MAEST) transformer model (trained on Discogs) and predicts the top-N genres with confidence scores — inference is powered by [maest-infer](https://github.com/openmirlab/maest-infer), a lightweight PyTorch-native repackaging of MAEST
- 📊 **Aggregates** all results into `tracks_genres.xlsx` — with genre names, confidence scores, and a broken-beat flag for DJ-friendly rhythm filtering
- 🏷️ **Writes** genres directly into audio file metadata — ID3 for MP3/WAV, Vorbis comments for FLAC, MP4 tags, and APE tags
- 📝 **Generates** a `report.md` summary after every run

**Key properties:**

- ⚡ **Incremental** — already-processed tracks are skipped on reruns, only new files are analyzed
- 🛡️ **Non-destructive by default** — existing genre tags are never overwritten unless you explicitly allow it
- 🖥️ **Cross-platform** — runs on Windows, Linux, WSL; GPU (CUDA) auto-detected and used when available
- 🎵 **Format-wide** — FLAC, WAV, AIFF, MP3, M4A, DSF, APE, WavPack

---

## ⚙️ How it works

```
Your Music Library
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  ANALYZE                                            │
│  For each unprocessed track:                        │
│  • Convert to 16kHz mono (ffmpeg if needed)         │
│  • Trim to analysis window (default: 60s–90s)       │
│  • Run MAEST inference → top-N genres + confidence  │
│  • Save result to per-track JSON                    │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  EXCEL                                              │
│  • Read all JSON files                              │
│  • Deduplicate against existing rows                │
│  • Write / update  tracks_genres.xlsx               │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  TAG  (optional)                                    │
│  • Read tracks_genres.xlsx                          │
│  • Write genre string into audio file metadata      │
│  • Update status column (incremental-safe)          │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
                   report.md
```

---

## 🗂️ Project structure

```
music-genre-tagger/
├── src/
│   ├── main.py          # Entrypoint & CLI
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

> 🚀 **GPU note:** On first run, the pipeline automatically detects your NVIDIA GPU and installs a matching CUDA torch build (`cu121`). After install, rerun `python src/main.py` once to continue with updated packages.

---

## 🚀 Quick start

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

## 📁 Output structure

```
<output_directory>/
└── <music_folder_name>/
    ├── json/
    │   ├── track_name__hash.json   ← per-track inference result
    │   └── ...
    ├── tracks_genres.xlsx          ← aggregated genre table
    └── report.md                   ← run summary
```

> If `output_directory` is not set, the project root is used as the base.

---

## 🔧 Configuration

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

## 🎵 Supported audio formats

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

> ⚠️ Formats requiring **ffmpeg** need `ffmpeg` available in your system PATH. If unavailable, those tracks are skipped with a per-file error while the rest of the pipeline continues normally.

---

## 🙏 Credits

This project is built on top of the MAEST model and ecosystem:

- **[MAEST](https://github.com/palonso/MAEST)** — Music Audio Efficient Spectrogram Transformer, the original research model by Pablo Alonso-Jiménez and colleagues at the Music Technology Group (MTG), Universitat Pompeu Fabra. Trained on the Discogs dataset for multi-label music genre classification.

- **[maest-infer](https://github.com/openmirlab/maest-infer)** — A lightweight, dependency-minimal repackaging of MAEST focused solely on inference, with native PyTorch and torchaudio support. This is the package used directly by Music Genre Tagger.
