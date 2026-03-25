# AGENTS

Guidance for autonomous coding agents (AI assistants, LLMs, automated tools) working in this repository.

---

## Project overview

Music Genre Tagger is a script-based Python pipeline that automatically detects music genres for audio files using the MAEST deep learning model and writes the results into audio file metadata tags.

The pipeline is designed to be:
- Incremental: already-processed files are skipped on reruns at every stage.
- Non-destructive by default: existing genre tags are never overwritten unless explicitly configured.
- Cross-platform: runs on Windows, Linux, and WSL; paths are stored as a single OS-native absolute string.

---

## Repository map

```
src/
  main.py          - CLI entrypoint; argparse; calls environment checks then run_pipeline()
  pipeline.py      - Core orchestration; all stage runners; audio loading; inference
  environment.py   - Pre-flight checks; Python version, GPU detection, torch/CUDA auto-install
  extractor.py     - Reads per-track JSON files; writes/updates tracks_genres.xlsx
  tagger.py        - Reads tracks_genres.xlsx; writes genre tags into audio file metadata
  report.py        - Builds report.md and library summary aggregates; loads cumulative timings
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
  apply_cli_overrides(config, args)                    <- pipeline.py
  run_environment_checks(project_dir, models_dir,
    checkpoint_filename, checkpoint_path_value)        <- environment.py
  run_pipeline(config, stage, script_dir,
    non_interactive)                                   <- pipeline.py

[run_pipeline] stage: analyze
  find_audio_files(input_dir)
    -> list of Path objects (sorted, filtered by extension and optional pattern)
  get_existing_json_stems(json_dir)
    -> set of already-processed stems (for incremental skip)
  load_models(config, script_dir)
    -> {"maest": {model_name: {"model", "arch", "checkpoint", "device"}}}
  for each unprocessed file:
    analyze_audio_file(path, models, config, json_dir)
      _resolve_audio_duration_seconds(path)  <- mutagen; fallback from decoded audio
      load_mono_16k(path) or convert_audio_to_wav(path) for unsupported formats
      trim_audio_segment(wav, offset, duration, sample_rate)
      model.predict_labels(wav_segment)      <- maest_infer
      process_predictions(scores, labels, num_genres)
        -> [(genre_label, confidence), ...]  labels cleaned via process_labels()
      write per-track JSON to json_dir/<stem>__<sha1[:16]>.json

[run_pipeline] stage: excel
  extractor.create_excel_report(json_dir, excel_path)
    find_json_files(json_dir)
    for each JSON: process_json_file() -> row dict
    _filter_new_records(new_rows, existing_df)   <- dedup by path
    write/append tracks_genres.xlsx

[run_pipeline] stage: tag
  tagger.run_genre_tagging(excel_path, config)
    for each row in DataFrame:
      skip if status == "tag_success"
      skip if existing genre tag and overwrite_existing == False
      prepare_genre_string(genres, max_genres, separator)
      set_genre_tag(audio_file, genre_string, file_extension)  <- format-specific dispatch
      update status column in Excel

[run_pipeline] always (finally block)
  write or refresh report.md from current stage state
```

---

## Key data structures

### Per-track JSON file

Written to `<output>/<input_folder_name>/json/<stem>__<sha1[:16]>.json`.

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

If inference fails, an `"error"` key is added at the top level with the exception string; `genres.labels` and `genres.confidences` remain empty lists. The pipeline continues to the next file.

Key-by-key reference:

| Key path | Type | Notes |
|---|---|---|
| `timestamp` | str (ISO 8601) | `datetime.now().isoformat()` — written first |
| `hash` | str | First 16 chars of SHA-1 of resolved POSIX path |
| `file.path` | str | OS-native absolute path (`original_audio_path.resolve()`) |
| `file.name` | str | Filename stem without extension |
| `file.extension` | str | File suffix, e.g. `".flac"` |
| `file.size.megabytes` | float | Truncated to 2 decimal places (floor, not round) |
| `file.size.bytes` | int | Raw `stat().st_size` |
| `file.duration.time` | str | `"HH:MM:SS"` — human-readable, no fractional seconds |
| `file.duration.seconds` | float | Truncated to 2 decimal places (floor, not round) |
| `genres.labels` | list[str] | Cleaned genre names; hierarchical prefix stripped |
| `genres.confidences` | list[float] | Rounded to 4 decimal places |
| `genres.model` | str | Model key, e.g. `"maest_519l_pytorch"` |
| `analysis_config.audio_segment_offset` | int/float | Start of analysis window in seconds |
| `analysis_config.audio_segment_duration` | int/float | Length of analysis window in seconds |
| `error` | str | Only present if inference raised an exception |

### Excel file: tracks_genres.xlsx

One row per audio file. Columns in exact order:

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `status` | str | Analysis / tagging status (see table below) |
| 2 | `path` | str | OS-native absolute path to the audio file |
| 3 | `name` | str | Filename stem without extension |
| 4 | `duration` | timedelta | Track length; formatted `[h]:mm:ss` in Excel for sorting |
| 5 | `genres` | str | Top genres as a comma-separated string |
| 6 | `is_broken_beat` | bool | `True` if any genre matches a broken-beat rhythm pattern |
| 7 | `model_key` | str | Key identifying which model produced the result |
| 8 | `confidences` | str | Confidence scores matching each genre, comma-separated |

Deduplication key: `path` column.

Excel formatting applied automatically:
- Header row: dark blue fill (`#1F4E78`), white bold text, centered, thin grey borders
- Column widths: auto-sized to content
- `duration` column: number format `[h]:mm:ss`
- Row 1 frozen (`freeze_panes = "A2"`)
- Auto-filter on all columns

### Status values

Set by `extractor.py` after analysis stage:

| Value | Meaning |
|---|---|
| `analysis_success` | Inference completed without errors |
| `analysis_error` | Inference raised an exception; `error` field present in JSON |

Overwritten by `tagger.py` after tagging stage:

| Value | Meaning |
|---|---|
| `tag_success` | Genre tag written and saved successfully |
| `tag_skipped_existing` | File already had a genre tag; `OVERWRITE_EXISTING = False` |
| `tag_error` | Any tagging failure (file not found, unsupported format, save error, etc.) |

Rows with `tag_success` are skipped on the next tagging run (incremental safety).

### Report file: report.md

`report.md` is cumulative for a given output folder and is refreshed after each completed stage plus once in the final `finally` block.

Key behavior:

- Timing fields are cumulative across previous runs for the same `report.md`:
  - `total_runtime_seconds`
  - `analyze_seconds`
  - `excel_seconds`
  - `tag_seconds`
- Previous timing values are loaded from the existing `report.md` at pipeline start.
- Current-stage counters such as `processed_now`, `rows_added_now`, and `tag_success_now` reflect only the current run.
- `top_genres` in the report is limited to the top 3 genres.
- `Broken Beat Summary` reports:
  - `broken_beat_tracks`
  - `non_broken_beat_tracks`
  - `broken_beat_share`
- `Tag` section is always present; if tagging did not run, it reports `status: not_run`.

---

## Behavior invariants

These behaviors are fixed. Do not change them.

- Default model result key is `maest_519l_pytorch`.
  - Custom `MODEL_KEY` in config applies only when `MODEL_FILE_PATH` is also set.
- Label cleanup: hierarchical Discogs labels are stripped to leaf only.
  - Pattern: `"Electronic---Techno"` becomes `"Techno"`.
  - Implementation: `label.split("---", 1)[-1]`
- Extractor output columns must remain exactly (in this order):
  - `status`, `path`, `name`, `duration`, `genres`, `is_broken_beat`, `model_key`, `confidences`
- Tag safety default: `OVERWRITE_EXISTING = False`.
  - Never change this default. The user must explicitly set it to `True`.
- Output directory structure derived from input path:
  - `<output_directory>/<input_folder_name>/json/`
  - `<output_directory>/<input_folder_name>/tracks_genres.xlsx`
  - `<output_directory>/<input_folder_name>/report.md`
- JSON stem format: `{file_stem}__{sha1_of_resolved_posix_path[:16]}`.
  - This hash provides collision resistance across folders with identical filenames.
- JSON `hash` field stores the same 16-char SHA-1 suffix used in the filename.
- `file.path` in JSON: single OS-native absolute string. No path triad (win/wsl/linux).
- Duration and size: `file.duration.seconds` and `file.size.megabytes` are always truncated (floor) to 2 decimal places, never rounded mathematically.

---

## Module contracts

### config.py

- Exposes `get_config() -> dict` that returns all constants as a flat dictionary.
- Add new configuration keys here and to `apply_cli_overrides()` in `pipeline.py` if they need CLI exposure.
- Tagger sub-config keys: `genre_source_field="genres"`, `file_path_field="path"`, `status_field="status"`.

### environment.py

- `run_environment_checks(project_dir: Path, models_dir: str, checkpoint_filename: str, checkpoint_path_value: str) -> bool`
  - Returns `True` only if all checks pass and no restart is required.
  - Hard failures: wrong Python version, missing packages, broken MAEST API, invalid checkpoint path.
  - Soft behavior: if NVIDIA GPU found and torch is CPU-only, reinstalls torch from CUDA index and sets `restart_required = True` (returns `False`).
  - FFmpeg absence is a warning only, not a hard failure.

### pipeline.py

- `run_pipeline(config: Dict, stage: str, script_dir: Path, non_interactive: bool) -> int` — main entry; returns 0 on success, 1 on failure.
- `apply_cli_overrides(base_config, args)` — merges argparse namespace into config dict; sets `config["tag_mode"]`.
- `apply_model_runtime_defaults(config, base_dir)` — builds `maest_models` dict and `maest_result_key` at runtime.
- `build_runtime_paths(script_dir, config, stage, non_interactive) -> RuntimePaths` — resolves input/output dirs and derives all output paths.
- `load_models(config, script_dir) -> Dict` — resolves checkpoint, loads MAEST, returns `{"maest": {name: {model, arch, checkpoint, device}}}`.
- `analyze_audio_file(original_audio_path, models, config, output_dir) -> Dict` — per-file inference; always writes a JSON (with `error` key on failure).
- `_resolve_audio_duration_seconds(audio_path) -> Optional[float]` — reads duration via mutagen; returns `None` on failure (fallback: `len(audio)/sample_rate`).
- `_truncate_two_decimals(value) -> float` — truncates float to 2 decimal places without rounding (`Decimal + ROUND_DOWN`).
- `_format_duration_hhmmss(total_seconds) -> str` — formats seconds as `HH:MM:SS`.

### report.py

- `load_cumulative_timings(report_path: Path) -> dict[str, float]` — reads timing totals from an existing `report.md`; missing values default to `0.0`.
- `summarize_json_library(json_dir: Path) -> dict` — aggregates report-level metrics directly from per-track JSON files.
- `summarize_excel_library(excel_path: Path) -> dict` — aggregates report-level metrics from `tracks_genres.xlsx`.
- `summarize_excel_dataframe(dataframe: pd.DataFrame) -> dict` — computes in-memory summary stats for report generation.
- `load_best_available_library_summary(excel_path: Path, json_dir: Path) -> dict | None` — prefers Excel summary, falls back to JSON summary.
- `write_markdown_report(...)` — writes the expanded `report.md` with cumulative timings and current library snapshot.

### extractor.py

- `create_excel_report(json_dir: Path, output_excel: Path) -> dict` — idempotent; safe to rerun; deduplicates by `path`.
- `process_json_file(json_path: Path) -> Optional[dict]` — reads JSON; returns row dict with `status` set to `analysis_success` or `analysis_error`.
- `check_broken_beat(genres: list[str]) -> bool` — returns `True` if any genre matches a broken-beat pattern.
- `auto_adjust_excel_columns(excel_path)` — applies all visual formatting: header style, column widths, `[h]:mm:ss` format for `duration`, freeze panes, autofilter.

### tagger.py

- `run_genre_tagging(excel_path: Path, config: Dict) -> dict` — reads Excel, tags files, updates status column in-place.
- `set_genre_tag(audio_file, genre_value: str, file_extension: str) -> bool` — dispatches to format-specific writer based on file suffix.
- `get_existing_genre(audio_file, file_extension: str) -> str | None` — reads current genre tag; returns `None` if tag absent.
- `convert_path_for_current_env(file_path: str) -> str` — handles Windows ↔ WSL path conversion at runtime.
- Format dispatch table:

| Extension(s) | Tag format | Tag key |
|---|---|---|
| `.m4a`, `.alac`, MP4 instances | MP4 | `©gen` |
| `.mp3`, `.wav`, `.aif`, `.aiff`, `.dsf`, `.dff`, MP3/WAVE/AIFF instances | ID3 | `TCON` |
| `.flac`, FLAC instances | Vorbis comment | `GENRE` |
| `.ape`, `.wv` | APE tag | `Genre` |

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
   `python -m py_compile src/config.py src/environment.py src/pipeline.py src/extractor.py src/tagger.py src/report.py src/main.py`

2. CLI help check:
   `python src/main.py --help`

3. If analysis logic changed: run a one-file smoke test:
   `python src/main.py --stage analyze --max-files 1 --input-directory "path/to/music" --output-directory "path/to/meta"`

4. If extractor changed: verify Excel output contains all required columns in correct order and deduplication by `path` works correctly.

5. If tagger changed: verify that status column is updated correctly (`tag_success` / `tag_error` / `tag_skipped_existing`) and that files with existing genre tags are not overwritten when `OVERWRITE_EXISTING = False`.

6. If environment logic changed: verify the torch detection flow, CUDA reinstall path, and the `restart_required` flag behavior.

7. Ensure no hardcoded machine-specific absolute paths appear in any source file or example.

8. If report logic changed: verify that cumulative timing fields continue to increase across reruns for the same output folder and that `Tag` / `Broken Beat Summary` sections render correctly.
