# AGENTS

Guidance for autonomous coding agents working in this repository.
This project is a script-based Python audio metadata pipeline.

## 1) Repository at a glance

- Entrypoint: `src/main.py`
- Core flow orchestration: `src/pipeline.py`
- Environment checks: `src/environment.py`
- JSON to Excel stage: `src/extractor.py`
- Tag writing stage: `src/tagger.py`
- User-facing configuration: `src/config.py`
- Dependency list: `requirements.txt`

Default flow (`--stage all`):
1. Environment checks
2. Audio analysis -> JSON
3. JSON aggregation -> `tracks_genres.xlsx`
4. Optional tag writing
5. `report.md` generation

## 2) Rule files from other tools

Checked for additional policy files:

- Cursor rules: `.cursor/rules/**` -> not found
- Cursor single file: `.cursorrules` -> not found
- Copilot rules: `.github/copilot-instructions.md` -> not found

If any of these files are added later, treat them as higher-priority guidance.

## 3) Setup commands

- Create virtual environment:
  - `python -m venv .venv`
- Activate environment:
  - PowerShell: `.\.venv\Scripts\Activate.ps1`
  - Bash: `source .venv/bin/activate`
- Install dependencies:
  - `pip install -r requirements.txt`

## 4) Run commands

- Full pipeline:
  - `python src/main.py`
- Stage-only runs:
  - `python src/main.py --stage analyze`
  - `python src/main.py --stage excel`
  - `python src/main.py --stage tag`

Useful overrides:

- `--input-directory <path>`
- `--output-directory <path>`
- `--file-pattern <text>`
- `--max-files <N>`
- `--convert-to-wav`
- `--tag-yes` / `--tag-no`
- `--non-interactive`

## 5) Build, lint, and test commands

There is no package build step and no committed linter/formatter config.
Use these reliability checks before finishing changes.

### Syntax / static safety

- Check all runtime modules:
  - `python -m py_compile src/config.py src/environment.py src/pipeline.py src/extractor.py src/tagger.py src/main.py`
- Check one file:
  - `python -m py_compile src/pipeline.py`
- CLI smoke check:
  - `python src/main.py --help`

### Runtime smoke checks

- Environment check path:
  - `python src/main.py --stage analyze --max-files 1`
- Excel-only path (after JSON exists):
  - `python src/main.py --stage excel --input-directory "path/to/music" --output-directory "path/to/meta"`
- Tag-only path (after Excel exists):
  - `python src/main.py --stage tag --input-directory "path/to/music" --output-directory "path/to/meta"`

### Single test guidance

There is currently no `tests/` directory.
Use one of these as the "single test" equivalent:

- Analyze one file only:
  - `python src/main.py --stage analyze --max-files 1 --input-directory "path/to/music" --output-directory "path/to/meta"`
- Analyze a narrow subset:
  - `python src/main.py --stage analyze --file-pattern "demo" --max-files 1 --input-directory "path/to/music" --output-directory "path/to/meta"`

If pytest tests are added later, run one test with:

- `python -m pytest tests/test_file.py::test_name -q`

## 6) Code style and conventions

### Imports

- Order imports as: stdlib -> third-party -> local modules.
- Prefer explicit imports; avoid wildcard imports.
- Remove unused imports when editing a file.

### Formatting

- Follow PEP 8 defaults (4 spaces, readable line lengths).
- Keep code straightforward; avoid unnecessary abstraction.
- Prefer short helper functions over deeply nested logic.

### Types

- Add type hints to new/edited function signatures.
- Keep return types stable for pipeline-facing functions.
- Use `Optional[...]` and concrete container types where useful.

### Naming

- Functions and variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Keep names aligned with domain terms used in the pipeline.

### Error handling

- Fail fast on startup/config problems.
- In per-file loops, log the error and continue when safe.
- Include actionable context in messages (path, stage, key).
- Avoid bare `except`; catch `Exception` explicitly.

### Logging

- Use `logging`, not `print`, for operational output.
- Use levels correctly: `info` for progress, `warning` for degraded mode, `error` for failures.
- Keep user-facing logs in English.

### Paths and I/O

- Use `pathlib.Path` consistently.
- Resolve relative paths from project context, not shell assumptions.
- Read/write text using UTF-8.
- Do not hardcode machine-specific absolute paths.

## 7) Behavior invariants (do not break)

- Preserve default result key behavior:
  - Default model mode uses `maest_519l_pytorch`.
  - Custom `MODEL_KEY` applies only when `MODEL_FILE_PATH` is set.
- Preserve label cleanup behavior: `A---B -> B`.
- Preserve extractor output fields:
  - `file_path`, `file_name`, `genres_maest`, `confidences`, `is_broken_beat`, `model_key`
- Preserve tag safety default:
  - `overwrite_existing = False`
- Keep metadata structure derived from input/output directories:
  - `<output>/<input_name>/json/`
  - `<output>/<input_name>/tracks_genres.xlsx`
  - `<output>/<input_name>/report.md`

## 8) Dependency and model policy

- Do not auto-upgrade dependency versions unless requested.
- Default checkpoint behavior must remain internal:
  - if `MODEL_FILE_PATH` is empty, use built-in default model logic,
  - download checkpoint into `src/models` when needed.
- Keep `models/` artifacts out of git.

## 9) Change safety checklist

Before handing off:

1. Run `py_compile` for edited files (or all `src/*.py`).
2. Run `python src/main.py --help`.
3. If analysis logic changed, run a one-file smoke (`--max-files 1`).
4. If extractor changed, verify expected Excel columns and dedupe by `file_path`.
5. If tagger changed, verify status updates and no overwrite when existing genre is present.
6. Ensure docs/examples remain generic (no local machine paths).
