# AGENTS

Guidance for coding agents working in this repository.

## Project overview

Music Genre Tagger analyzes audio files with MAEST, stores track results in `SQLite`, rebuilds `genres.xlsx` from the database, optionally exports `tracks.json`, and can write detected genres back into file metadata.

Core properties:

- incremental analysis by track `hash`
- non-destructive tagging by default
- cross-platform path handling for Windows, Linux, and WSL
- single local database file as the source of truth

## Repository map

```text
src/
  main.py        - CLI entrypoint
  pipeline.py    - pipeline orchestration, inference, stage flow
  environment.py - dependency and environment checks
  extractor.py   - database -> genres.xlsx export
  report.py      - report generation and library summaries
  storage.py     - SQLite storage, rows/dataframe export, tracks.json export
  tagger.py      - writes genre tags into audio metadata
  config.py      - user-facing configuration
```

## Pipeline flow

```text
main.py
  -> apply_cli_overrides()
  -> run_environment_checks()
  -> run_pipeline()

run_pipeline(stage="analyze")
  -> build_runtime_paths()
  -> init_db(tracks.db)
  -> find_audio_files()
  -> get_existing_hashes()
  -> analyze_audio_file()
  -> upsert_track()
  -> optionally export_tracks_json()

run_pipeline(stage="excel")
  -> create_excel_report(tracks.db, genres.xlsx)

run_pipeline(stage="tag")
  -> run_genre_tagging(genres.xlsx)
  -> update_track_statuses(tracks.db, status_updates)
  -> optionally export_tracks_json()

all stages
  -> summarize_database_library()
  -> write_markdown_report(report.md)
```

## Data model

### SQLite file: `tracks.db`

Table: `tracks`

Columns:

- `timestamp`
- `hash`
- `path`
- `name`
- `extension`
- `megabytes`
- `bytes`
- `time`
- `seconds`
- `labels`
- `confidences`
- `model`
- `audio_segment_offset`
- `audio_segment_duration`
- `error`
- `status`
- `updated_at`

Storage rules:

- `hash` is the primary key (`UNIQUE`)
- `labels` and `confidences` are JSON strings in the database
- indexes on `path` and `status`
- `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=30000` set on every connection
- `status` is a single field with these values:
  - `analysis_success`
  - `analysis_error`
  - `tag_success`
  - `tag_skipped_existing`
  - `tag_error`

### Combined JSON file: `tracks.json`

- optional export
- enabled by default (`WRITE_TRACKS_JSON = True`)
- disabled with `--no-tracks-json`
- contains a root object with `timestamp` and `tracks`
- each item in `tracks` matches the familiar per-track payload shape
- refreshed after `analyze`, after `tag`, and in the `finally` block

### Excel file: `genres.xlsx`

Fully rebuilt from the database every time the `excel` stage runs.

Column order is fixed:

1. `status`
2. `path`
3. `name`
4. `duration`
5. `genres`
6. `is_broken_beat`
7. `model_key`
8. `confidences`

Excel formatting applied automatically:

- header row: dark blue fill (`#1F4E78`), white bold text, centered, thin grey borders
- column widths: auto-sized to content
- `duration` column: number format `[h]:mm:ss`
- row 1 frozen (`freeze_panes = "A2"`)
- auto-filter on all columns

## Behavior invariants

- Default model result key stays `maest_519l_pytorch`.
- Label cleanup stays `label.split("---", 1)[-1]`.
- `OVERWRITE_EXISTING = False` remains the safe default.
- Output structure is:
  - `<output_directory>/<input_folder_name>/tracks.db`
  - `<output_directory>/<input_folder_name>/genres.xlsx`
  - `<output_directory>/<input_folder_name>/report.md`
  - `<output_directory>/<input_folder_name>/tracks.json` when enabled
- Track `hash` is the first 16 chars of the SHA-1 of the resolved POSIX path.
- `file.path` stays a single OS-native absolute path string.
- `megabytes` and `seconds` are truncated to 2 decimals, not rounded mathematically.
- `genres.xlsx` is fully rebuilt from the database, not appended incrementally.
- Rows with `tag_success` are skipped on the next tagging run (incremental safety).

## Module contracts

### `config.py`

- `get_config()` returns the full runtime config dict.
- `write_tracks_json` defaults to `True` in the returned dict; it is not a user-facing constant — it can only be set to `False` via the `--no-tracks-json` CLI flag.
- Tagger sub-config keys live under `config["tagger"]`:
  - `genre_source_field` = `"genres"`
  - `file_path_field` = `"path"`
  - `status_field` = `"status"`
  - `genre_separator`, `max_genres`, `overwrite_existing`, `max_rows`

### `main.py`

- Parses CLI args with `argparse` and calls `apply_cli_overrides()`.
- Runs `run_environment_checks()` before the pipeline.
- Exposes `--no-tracks-json` to disable `tracks.json` export.
- Returns exit code `0` on success, `1` on failure.

### `environment.py`

- `run_environment_checks(project_dir, models_dir, checkpoint_filename, checkpoint_path_value) -> bool`
  - Returns `True` only if all checks pass and no restart is required.
  - Hard failures: wrong Python version, missing packages, broken MAEST API, failed SQLite runtime check, invalid checkpoint path, restart required after torch reinstall.
  - Soft behavior: if NVIDIA GPU found and torch is CPU-only, reinstalls torch from CUDA index and sets `restart_required = True` (returns `False`).
  - FFmpeg absence is a warning only, not a hard failure.
  - `_check_sqlite_runtime()` runs an in-memory smoke test and verifies `sqlite3` is functional.

### `pipeline.py`

- `run_pipeline(config, stage, script_dir, non_interactive) -> int` — main coordinator; returns `0` on success, `1` on failure.
- `apply_cli_overrides(base_config, args)` — merges argparse namespace into config dict; sets `config["tag_mode"]` and `config["write_tracks_json"]`.
- `apply_model_runtime_defaults(config, base_dir)` — builds `maest_models` dict and `maest_result_key` at runtime.
- `build_runtime_paths(script_dir, config, stage, non_interactive) -> RuntimePaths` — resolves input/output dirs and derives all output paths:
  - `db_path` = `meta_root / "tracks.db"`
  - `tracks_json_path` = `meta_root / "tracks.json"`
  - `excel_path` = `meta_root / "genres.xlsx"`
  - `report_path` = `meta_root / "report.md"`
- `load_models(config, script_dir) -> Dict` — resolves checkpoint, loads MAEST, returns `{"maest": {name: {model, arch, checkpoint, device}}}`.
- `analyze_audio_file(original_audio_path, models, config) -> Dict` — per-file inference; returns the track payload dict (with `error` key on failure); does NOT write to disk.
- `run_analysis_stage(config, script_dir, input_dir, db_path) -> Dict` — calls `init_db`, finds audio, skips existing hashes, runs analysis, calls `upsert_track` per file.
- `run_excel_stage(config, db_path, excel_path) -> Dict` — delegates to `create_excel_report`.
- `run_tag_stage(config, db_path, excel_path) -> Dict` — runs `run_genre_tagging`, then calls `update_track_statuses` to sync status back to DB.
- `build_audio_hash(audio_path) -> str` — first 16 chars of SHA-1 of resolved POSIX path.
- `_truncate_two_decimals(value) -> float` — truncates to 2 decimal places using `Decimal + ROUND_DOWN`.
- `_format_duration_hhmmss(total_seconds) -> str` — formats seconds as `HH:MM:SS`.
- `_resolve_audio_duration_seconds(audio_path) -> Optional[float]` — reads duration via mutagen; returns `None` on failure.

### `storage.py`

- `init_db(db_path)` — creates schema and indexes; sets WAL mode.
- `get_existing_hashes(db_path) -> set[str]` — returns all `hash` values from the DB.
- `upsert_track(db_path, track_data)` — flattens nested track payload and inserts or updates by `hash`.
- `build_excel_dataframe(db_path) -> pd.DataFrame` — converts DB rows into the fixed Excel column shape.
- `load_track_records(db_path) -> List[Dict]` — returns all rows ordered by `path COLLATE NOCASE`.
- `update_track_statuses(db_path, status_updates)` — bulk-updates `status` and `updated_at` by `path`.
- `export_tracks_json(db_path, output_path)` — writes combined JSON snapshot with `timestamp` and `tracks`.

### `extractor.py`

- `create_excel_report(db_path, output_excel) -> Dict` — always fully rebuilds the workbook from DB data; applies all Excel formatting via `auto_adjust_excel_columns`.
- `auto_adjust_excel_columns(excel_path)` — applies header style, column widths, `[h]:mm:ss` format for `duration`, freeze panes, autofilter.
- Returns `{"tracks_seen", "rows_written", "rows_total", "library_summary"}`.

### `report.py`

- `summarize_database_library(db_path) -> dict` — primary library summary path; builds DataFrame from DB then delegates to `summarize_excel_dataframe`.
- `summarize_excel_dataframe(dataframe) -> dict` — shared in-memory summary implementation.
- `summarize_excel_library(excel_path) -> dict` — fallback summary from `genres.xlsx`.
- `load_best_available_library_summary(db_path, excel_path) -> dict | None` — prefers DB, falls back to Excel.
- `load_cumulative_timings(report_path) -> dict` — reads timing totals from existing `report.md`; missing values default to `0.0`.
- `write_markdown_report(report_path, stage, input_dir, db_path, tracks_json_path, excel_path, ...)` — writes cumulative `report.md`; `tracks_json_path` is `None` when export is disabled.

### `tagger.py`

- `run_genre_tagging(excel_path, config) -> dict` — reads `genres.xlsx`, tags files, updates status column in-place, saves Excel.
- Returns `{"success", "skipped", "error", "already_processed", "status_updates", "status_breakdown", "library_summary"}`.
- `status_updates` is a list of `(path, status)` tuples consumed by `update_track_statuses` in `pipeline.py`.
- `set_genre_tag(audio_file, genre_value, file_extension) -> bool` — dispatches to format-specific writer.
- `get_existing_genre(audio_file, file_extension) -> str | None` — reads current genre tag.
- `convert_path_for_current_env(file_path) -> str` — handles Windows ↔ WSL path conversion at runtime.

Format dispatch table:

| Extension(s) | Tag format | Tag key |
|---|---|---|
| `.m4a`, `.alac`, MP4 instances | MP4 | `©gen` |
| `.mp3`, `.wav`, `.aif`, `.aiff`, `.dsf`, `.dff`, MP3/WAVE/AIFF instances | ID3 | `TCON` |
| `.flac`, FLAC instances | Vorbis comment | `GENRE` |
| `.ape`, `.wv` | APE tag | `Genre` |

## Model and checkpoint policy

- Default checkpoint: `discogs-maest-30s-pw-129e-519l-swa.ckpt`
- Default checkpoint location: `src/models/` (relative to project root)
- If the checkpoint file is missing and `MODEL_FILE_PATH` is empty, it is downloaded automatically from GitHub Releases on first run via `urllib.request.urlretrieve` to a `.tmp` file, then atomically renamed.
- Keep `src/models/` out of git (covered by `.gitignore`).
- Do not change the default model architecture (`maest_519l_pytorch`) or the default checkpoint filename without updating all references throughout the codebase.

## Dependency policy

- Do not upgrade dependency versions unless explicitly requested.
- `requirements.txt` pins baseline `torch`, `torchaudio`, `torchvision` without CUDA specifiers.
- At runtime, `environment.py` detects NVIDIA GPU via `nvidia-smi` and auto-upgrades torch to a CUDA build (`cu121` index) if needed.
- After a CUDA torch reinstall, one manual rerun is required. This is by design.
- Do not remove or bypass the torch auto-upgrade logic in `environment.py`.

## Supported audio formats

| Extension | Read method | Tag format |
|---|---|---|
| `.flac` | soundfile | Vorbis comment |
| `.wav` | soundfile | ID3 |
| `.aiff`, `.aif` | soundfile | ID3 |
| `.mp3` | soundfile | ID3 |
| `.m4a` | ffmpeg -> temp WAV | MP4 (`©gen`) |
| `.dsf` | ffmpeg -> temp WAV | ID3 |
| `.ape` | ffmpeg -> temp WAV | APE tag |
| `.wv` | ffmpeg -> temp WAV | APE tag |

Formats requiring ffmpeg produce a temporary WAV file in the system temp directory; it is deleted after inference completes.

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

## Verification checklist

Run before finishing changes:

1. `python -m py_compile src/config.py src/environment.py src/pipeline.py src/extractor.py src/tagger.py src/report.py src/main.py src/storage.py`
2. `python src/main.py --help`
3. if analysis logic changed, smoke test one track
4. verify `genres.xlsx` columns and ordering
5. verify tag status sync back into `tracks.db`
6. ensure no hardcoded machine-specific paths in source or docs
