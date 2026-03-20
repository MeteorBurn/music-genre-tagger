# AGENTS

Practical instructions for coding agents working in this repository.
This is a script-driven Python pipeline with a single entrypoint.

## Project structure

- Entrypoint: `src/main.py`
- Pipeline orchestration: `src/pipeline.py`
- Environment checks: `src/environment.py`
- JSON to Excel extraction: `src/extractor.py`
- Audio tagging: `src/tagger.py`
- User configuration: `src/config.py`
- Dependency source of truth: `requirements.txt`

## Flow overview

Default run (`--stage all`) executes:
1. Environment checks
2. Audio analysis to JSON
3. JSON aggregation to `tracks_genres.xlsx`
4. Optional tagging
5. Markdown report generation

## Environment and setup commands

- Create venv:
  - `python -m venv .venv`
- Activate venv:
  - PowerShell: `.\.venv\Scripts\Activate.ps1`
  - Bash: `source .venv/bin/activate`
- Install dependencies:
  - `pip install -r requirements.txt`

## Run commands

- Full flow:
  - `python src/main.py`
- Stage-only runs:
  - `python src/main.py --stage analyze`
  - `python src/main.py --stage excel`
  - `python src/main.py --stage tag`

## Validation commands

- Syntax check all scripts:
  - `python -m py_compile src/config.py src/environment.py src/pipeline.py src/extractor.py src/tagger.py src/main.py`
- CLI check:
  - `python src/main.py --help`

## Invariants (do not break)

- Preserve MAEST result key: `maest_519l_pytorch`
- Preserve label cleanup: `A---B -> B`
- Preserve extractor fields:
  - `file_path`, `file_name`, `genres_maest`, `confidences`, `is_broken_beat`, `model_key`
- Preserve default tag safety:
  - `overwrite_existing = False`
- Do not silently change MAEST architecture/checkpoint behavior.

## Path and model rules

- Use `pathlib.Path` and keep relative-path handling deterministic.
- Keep examples generic in docs and logs (no machine-specific absolute paths).
- Default model directory is `src/models`.
- If custom `checkpoint_path` is provided, validate and use it.
- If custom `checkpoint_path` is empty and checkpoint file is missing, download checkpoint into `src/models`.
- Generic example paths for docs/config:
  - `path/to/project`
  - `path/to/music_library_demo`
  - `path/to/output_meta_demo`

## Coding and safety

- Keep logs, comments, and user-facing text in English.
- Prefer explicit error messages with actionable fixes.
- Do not auto-install or upgrade dependencies unless requested.
- Avoid destructive operations on user media files.
