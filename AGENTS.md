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

- `hash` is the primary key
- `labels` and `confidences` are JSON strings in the database
- `status` is a single field with these values:
  - `analysis_success`
  - `analysis_error`
  - `tag_success`
  - `tag_skipped_existing`
  - `tag_error`

### Combined JSON file: `tracks.json`

- optional export
- enabled by default
- disabled with `--no-tracks-json`
- contains a root object with `timestamp` and `tracks`
- each item in `tracks` matches the familiar per-track payload shape

### Excel file: `genres.xlsx`

Built from the database every time the `excel` stage runs.

Column order is fixed:

1. `status`
2. `path`
3. `name`
4. `duration`
5. `genres`
6. `is_broken_beat`
7. `model_key`
8. `confidences`

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

## Module contracts

### `config.py`

- `get_config()` returns the full runtime config dict.
- `write_tracks_json` is controlled here and can be disabled from CLI.

### `main.py`

- parses CLI args
- exposes `--no-tracks-json`

### `pipeline.py`

- `run_pipeline()` is the main coordinator
- `run_analysis_stage()` writes analysis results to `tracks.db`
- `run_excel_stage()` rebuilds `genres.xlsx` from `tracks.db`
- `run_tag_stage()` syncs updated status values back into `tracks.db`
- `analyze_audio_file()` returns the track payload in nested JSON-like form

### `storage.py`

- `init_db(db_path)` creates schema and indexes
- `get_existing_hashes(db_path)` returns processed hashes
- `upsert_track(db_path, track_data)` stores one analysis result
- `build_excel_dataframe(db_path)` converts DB rows into the fixed Excel shape
- `load_track_records(db_path)` returns ordered DB rows as dicts
- `update_track_statuses(db_path, status_updates)` writes tag statuses back to DB
- `export_tracks_json(db_path, output_path)` exports the combined JSON snapshot

### `extractor.py`

- `create_excel_report(db_path, output_excel)` always rebuilds the workbook from DB data
- preserves current Excel formatting and column order

### `report.py`

- `summarize_database_library(db_path)` is the primary library summary path
- `summarize_excel_dataframe(dataframe)` remains the shared summary implementation
- `write_markdown_report()` reports `db_path`, optional `tracks_json_path`, `excel_path`, and cumulative timings

### `tagger.py`

- `run_genre_tagging(excel_path, config)` updates the Excel status column
- returns `status_updates` so pipeline can sync them back to the database

## Verification checklist

Run before finishing changes:

1. `python -m py_compile src/config.py src/environment.py src/pipeline.py src/extractor.py src/tagger.py src/report.py src/main.py src/storage.py`
2. `python src/main.py --help`
3. if analysis logic changed, smoke test one track
4. verify `genres.xlsx` columns and ordering
5. verify tag status sync back into `tracks.db`
6. ensure no hardcoded machine-specific paths in source or docs
