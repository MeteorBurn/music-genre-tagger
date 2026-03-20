# AGENTS

Guidance for autonomous coding agents (AI assistants, LLMs, automated tools) working in this repository.

---

## Project overview

Music Genre Tagger is a script-based Python pipeline that automatically detects music genres for audio files using the MAEST deep learning model and writes the results into audio file metadata tags.

The pipeline is designed to be:
- Incremental: already-processed files are skipped on reruns at every stage.
- Non-destructive by default: existing genre tags are never overwritten unless explicitly configured.
- Cross-platform: paths are stored in three forms (Windows, WSL, POSIX) inside every JSON result file.

---

## Repository map

```
src/
  main.py          - CLI entrypoint; argparse; calls environment checks then run_pipeline()
  pipeline.py      - Core orchestration; all stage runners; audio loading; inference; reporting
  environment.py   - Pre-flight checks; Python version, GPU detection, torch/CUDA auto-install
  extractor.py     - Reads per-track JSON files; writes/updates tracks_genres.xlsx
  tagger.py        - Reads tracks_genres.xlsx; writes genre tags into audio file metadata
  config.py        - All user-facing constants; exposes get_config() -> dict

requirements.txt   - Python package dependencies
AGENTS.md          - This file
README.md          - User-facing documentation
```

---

## Pipeline stages and data flow

```
[src/main.py]
  parse CLI args
  apply_cli_overrides(config, args)       <- pipeline.py
  run_environment_checks(config)          <- environment.py
  run_pipeline(config)                    <- pipeline.py

[run_pipeline] stage: analyze
  find_audio_files(input_dir)
    -> list of Path objects (sorted, filtered by extension and optional pattern)
  get_existing_json_stems(json_dir)
    -> set of already-processed stems (for incremental skip)
  load_models(config)
    -> (maest_model, mel_model, labels, model_key)
  for each unprocessed file:
    analyze_audio_file(path, model, mel, labels, config, json_dir)
      load_mono_16k(path) or convert_audio_to_wav(path) for unsupported formats
      trim_audio_segment(wav, offset, duration, sample_rate)
      model.predict_labels(wav_segment)   <- maest_infer
      process_predictions(scores, labels, num_genres)
        -> [(genre_label, confidence), ...]   labels cleaned via process_labels()
      write per-track JSON to json_dir/<stem>__<sha1[:16]>.json

[run_pipeline] stage: excel
  extractor.create_excel_report(json_dir, excel_path)
    find_json_files(json_dir)
    for each JSON: process_json_file() -> row dict
    _filter_new_records(new_rows, existing_df)   <- dedup by file_path
    write/append tracks_genres.xlsx

[run_pipeline] stage: tag
  tagger.run_genre_tagging(excel_path, config)
    for each row in DataFrame:
      skip if status == "success"
      skip if existing genre tag and overwrite_existing == False
      prepare_genre_string(genres, max_genres, separator)
      set_genre_tag(path, genre_string)   <- format-specific dispatch
      update status column in Excel

[run_pipeline] always (finally block)
  write_markdown_report(report_path, stats)
```

---

## Key data structures

### Per-track JSON file

Written to `<output>/<input_folder_name>/json/<stem>__<sha1[:16]>.json`.

```json
{
  "file_path": {
    "win":   "C:\\Music\\track.flac",
    "wsl":   "/mnt/c/Music/track.flac",
    "linux": "/Music/track.flac"
  },
  "file_name": "track",
  "maest_519l_pytorch": {
    "genres": ["Techno", "House", "Electro"],
    "confidences": [0.82, 0.11, 0.04]
  },
  "model_key": "maest_519l_pytorch",
  "error": null
}
```

If inference fails, `"error"` holds the exception string and genre fields are absent. The pipeline continues to the next file.

### Excel file: tracks_genres.xlsx

One row per audio file. Columns:

| Column | Type | Description |
|--------|------|-------------|
| `file_path` | str | OS-appropriate absolute path to the audio file |
| `file_name` | str | Filename stem without extension |
| `genres_maest` | str | Top genres as a comma-separated string |
| `confidences` | str | Confidence scores matching each genre, comma-separated |
| `is_broken_beat` | bool | True if top genres include broken-beat rhythm patterns |
| `model_key` | str | Key identifying which model produced the result |
| `status` | str | Tagging status: empty / "success" / "skipped" / "error" |

---

## Behavior invariants

These behaviors are fixed. Do not change them.

- Default model result key is `maest_519l_pytorch`.
  - Custom `MODEL_KEY` in config applies only when `MODEL_FILE_PATH` is also set.
- Label cleanup: hierarchical Discogs labels are stripped to leaf only.
  - Pattern: `"Electronic---Techno"` becomes `"Techno"`.
  - Implementation: `label.split("---", 1)[-1]`
- Extractor output columns must remain exactly:
  - `file_path`, `file_name`, `genres_maest`, `confidences`, `is_broken_beat`, `model_key`
- Tag safety default: `OVERWRITE_EXISTING = False`.
  - Never change this default. The user must explicitly set it to `True`.
- Output directory structure derived from input path:
  - `<output_directory>/<input_folder_name>/json/`
  - `<output_directory>/<input_folder_name>/tracks_genres.xlsx`
  - `<output_directory>/<input_folder_name>/report.md`
- JSON stem format: `{file_stem}__{sha1_of_resolved_posix_path[:16]}`.
  - This hash provides collision resistance across folders with identical filenames.
- Path triad in JSON: every result file stores three path variants: `win`, `wsl`, `linux`.
  - Do not remove any of the three keys.

---

## Module contracts

### config.py

- Exposes `get_config() -> dict` that returns all constants as a flat dictionary.
- Direct attribute access (e.g. `config.NUM_GENRES`) is also valid in the codebase.
- Add new configuration keys here and to `apply_cli_overrides()` in `pipeline.py` if they need CLI exposure.

### environment.py

- `run_environment_checks(config) -> bool`
  - Returns `True` only if all checks pass and no restart is required.
  - Hard failures: wrong Python version, missing packages, broken MAEST API, invalid checkpoint path.
  - Soft behavior: if NVIDIA GPU found and torch is CPU-only, reinstalls torch from CUDA index and sets `restart_required = True` (returns `False`).
  - FFmpeg absence is a warning only, not a hard failure.

### pipeline.py

- `run_pipeline(config) -> int` — main entry; returns 0 on success, 1 on failure.
- `apply_cli_overrides(config, args)` — merges argparse namespace into config dict.
- `build_runtime_paths(config) -> RuntimePaths` — resolves input/output dirs and derives all output paths.
- `load_models(config) -> tuple` — resolves checkpoint, loads MAEST, returns `(model, mel, labels, model_key)`.
- `analyze_audio_file(...)` — per-file inference; always writes a JSON (with error field on failure).
- `write_markdown_report(...)` — called in `finally` block; always executes regardless of pipeline success.

### extractor.py

- `create_excel_report(json_dir, excel_path) -> dict` — idempotent; safe to rerun; deduplicates by `file_path`.
- `check_broken_beat(genres: list[str]) -> bool` — returns True if any genre matches a broken-beat pattern.
- `_preferred_path(path_dict) -> str` — selects `win` on Windows, `linux`/`wsl` on Linux/WSL.

### tagger.py

- `run_genre_tagging(excel_path, config) -> dict` — reads Excel, tags files, updates status column in-place.
- `set_genre_tag(path, genre_string, config)` — dispatches to format-specific writer based on file suffix.
- `get_existing_genre(path) -> str | None` — reads current genre tag; returns None if tag absent.
- Format dispatch table: `.mp4`/`.m4a` -> MP4 `©gen`; `.flac` -> Vorbis `GENRE`; `.mp3`/`.wav`/`.aiff` -> ID3 `TCON`; `.ape`/`.wv` -> APE `Genre`.

---

## Model and checkpoint policy

- Default checkpoint: `discogs-maest-30s-pw-129e-519l-swa.ckpt`
- Default checkpoint location: `src/models/` (relative to project root)
- If the checkpoint file is missing and `MODEL_FILE_PATH` is empty, it is downloaded automatically from GitHub Releases on first run.
- Download uses `urllib.request.urlretrieve` to a `.tmp` file, then atomically renamed.
- Keep `src/models/` out of git (covered by `.gitignore`).
- Do not change the default model architecture (`maest_519l_pytorch`) or the default checkpoint filename without updating all references to the result key throughout the codebase.

---

## Dependency policy

- Do not upgrade dependency versions unless explicitly requested.
- `requirements.txt` pins baseline `torch` and `torchaudio` without CUDA specifiers.
- At runtime, `environment.py` detects NVIDIA GPU via `nvidia-smi` and auto-upgrades torch to a CUDA build (`cu121` index) if needed.
- After a CUDA torch reinstall, one manual rerun is required. This is by design.
- Do not remove or bypass the torch auto-upgrade logic in `environment.py`.

---

## Supported audio formats

| Extension | Read method | Tag format |
|-----------|-------------|------------|
| `.flac` | soundfile | Vorbis comment |
| `.wav` | soundfile | ID3 |
| `.aiff` / `.aif` | soundfile | ID3 |
| `.mp3` | soundfile | ID3 |
| `.m4a` | ffmpeg -> temp WAV | MP4 (`©gen`) |
| `.dsf` | ffmpeg -> temp WAV | ID3 |
| `.ape` | ffmpeg -> temp WAV | APE tag |
| `.wv` | ffmpeg -> temp WAV | APE tag |

Formats requiring ffmpeg produce a temporary WAV file in the system temp directory; it is deleted after inference completes.

---

## Code style conventions

- Imports: stdlib first, then third-party, then local. No wildcard imports.
- Formatting: PEP 8, 4-space indentation.
- Type hints on all new and edited function signatures.
- Use `pathlib.Path` for all path operations; never use raw string concatenation for paths.
- Use `logging`, not `print`, for all operational output.
  - `info` for normal progress; `warning` for degraded/partial success; `error` for failures.
- Per-file errors in loops: log and continue. Never let a single file failure abort the full run.
- Catch `Exception` explicitly; avoid bare `except`.
- Include actionable context in error messages (file path, stage name, key).
- Constants in `UPPER_SNAKE_CASE`; functions and variables in `snake_case`.

---

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Bash)
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run commands

```bash
# Full pipeline (all stages)
python src/main.py

# With explicit paths
python src/main.py --input-directory "path/to/music" --output-directory "path/to/meta"

# Single stage
python src/main.py --stage analyze
python src/main.py --stage excel
python src/main.py --stage tag

# Non-interactive, skip tagging (CI/automation)
python src/main.py --non-interactive --tag-no --input-directory "path/to/music"
```

---

## Verification checklist

Run before finishing any change:

1. Syntax check all source files:
   `python -m py_compile src/config.py src/environment.py src/pipeline.py src/extractor.py src/tagger.py src/main.py`

2. CLI help check:
   `python src/main.py --help`

3. If analysis logic changed: run a one-file smoke test:
   `python src/main.py --stage analyze --max-files 1 --input-directory "path/to/music" --output-directory "path/to/meta"`

4. If extractor changed: verify Excel output contains all required columns and deduplication by `file_path` works correctly.

5. If tagger changed: verify that status column is updated and that files with existing genre tags are not overwritten when `OVERWRITE_EXISTING = False`.

6. If environment logic changed: verify the torch detection flow, CUDA reinstall path, and the `restart_required` flag behavior.

7. Ensure no hardcoded machine-specific absolute paths appear in any source file or example.
