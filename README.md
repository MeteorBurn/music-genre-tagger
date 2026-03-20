# Python Audio Pipeline

One-flow cross-platform pipeline for genre extraction with MAEST (PyTorch), Excel export, optional metadata tagging, and final markdown report.

## What this project is for

This project helps you analyze a music library and assign genre labels to tracks automatically. It provides a practical workflow for music cataloging with reproducible results.

Why this is useful:

- It saves time compared to manual genre sorting for large libraries.
- It provides consistent genre suggestions from the same model across all tracks.
- It creates structured metadata (`json`, `tracks_genres.xlsx`, `report.md`) that is easy to review.
- It can write selected genres directly into audio file tags, so music players and DJ software can filter and search by genre.
- It supports incremental reruns, so you can process only new tracks later without rebuilding everything from scratch.

## New entry point

- `src/main.py` is the only required script for running the full flow.
- Internal modules:
  - `src/environment.py`
  - `src/pipeline.py`
  - `src/extractor.py`
  - `src/tagger.py`
  - `src/config.py`

## What the flow does

1. Environment diagnostics with actionable fix instructions.
2. Audio analysis to per-track JSON (`maest_519l_pytorch`).
3. JSON aggregation to Excel (`tracks_genres.xlsx`) with dedupe by `file_path`.
4. Optional genre tag writing into audio metadata.
5. Final markdown report (`report.md`).

## Setup (PowerShell)

```powershell
cd path\to\project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Torch note:

- `requirements.txt` includes `torch` and `torchaudio` baseline dependencies.
- At startup, environment checks detect NVIDIA GPU and install a matching torch stack automatically.
- On NVIDIA systems, CUDA build install command uses `--index-url https://download.pytorch.org/whl/cu121`.
- Startup uses force-reinstall/no-cache for torch stack updates to avoid stale CPU wheel reuse.
- If torch stack is installed/updated, rerun `python .\src\main.py` once to continue with updated packages.

## CLI usage

Run all stages (default):

```powershell
python .\src\main.py
```

What this command does:

- Runs the full pipeline in order: environment check -> analyze -> excel -> optional tag -> report.
- If `input_directory` or `output_directory` is not configured, CLI asks for values interactively.
- Best choice for normal day-to-day usage.

Run a specific stage:

```powershell
python .\src\main.py --stage analyze
python .\src\main.py --stage excel
python .\src\main.py --stage tag
```

What each stage command does:

- `--stage analyze` scans audio files, runs MAEST inference, and writes JSON files.
- `--stage excel` reads JSON files and updates `tracks_genres.xlsx` with new records.
- `--stage tag` reads `tracks_genres.xlsx` and writes genres into audio tags.
- Stage-only commands are mainly useful for testing, debugging, or partial reruns.

## Command reference

- `python .\src\main.py --stage all --input-directory "path/to/music_library_demo" --output-directory "path/to/output_meta_demo"`
  - Full non-default path run. Creates/uses metadata project folder and executes all stages.
- `python .\src\main.py --stage analyze --input-directory "path/to/music_library_demo" --max-files 20`
  - Analyze only first 20 tracks (smoke or incremental test).
- `python .\src\main.py --stage excel --input-directory "path/to/music_library_demo" --output-directory "path/to/output_meta_demo"`
  - Build/update Excel for the metadata project derived from input/output paths.
- `python .\src\main.py --stage tag --input-directory "path/to/music_library_demo" --output-directory "path/to/output_meta_demo"`
  - Run tagging for the metadata project derived from input/output paths.
- `python .\src\main.py --stage all --non-interactive --tag-no`
  - CI/script mode. Disables prompts and skips tagging.

Stage options:

- `--stage all|analyze|excel|tag` (default `all`) - selects execution scope. `all` is the primary mode.
- `--input-directory <path>` - source audio library directory used by analyze stage.
- `--output-directory <path>` - base directory for generated metadata projects (one subfolder per input library).
- `--file-pattern <text>` - analyzes only files whose names contain this substring.
- `--max-files <N>` - hard cap on files processed in current analyze run.
- `--convert-to-wav` - enables temporary WAV conversion before inference (for incompatible formats).
- `--tag-yes` - for `--stage all`, always run tagging without interactive prompt.
- `--tag-no` - for `--stage all`, always skip tagging without interactive prompt.
- `--non-interactive` - disables input prompts; missing required values will cause explicit errors.
- `--loglevel <level>` - logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

Example test values:

- `--input-directory "path/to/music_library_demo"`
- `--output-directory "path/to/output_meta_demo"`

## Paths and meta structure

- If `input_directory` is empty and the selected stage needs analysis, the CLI asks for it.
- If `output_directory` is empty and the selected stage uses analyze flow, the CLI asks for output base path.
- If user skips output base path, project root is used as output base.
- With `--non-interactive` and empty `output_directory`, project root is used as output base.
- If no custom checkpoint path is provided, MAEST pretrained mode is used (managed by `maest_infer`).
- Meta root name is based on input directory name.
- Required structure inside meta root:
  - `json/`
  - `tracks_genres.xlsx`
  - `report.md`

## Model config rules

- `MODEL_FILE_PATH` in `src/config.py`:
  - empty -> default model is used through `maest_infer` pretrained mode.
  - set -> pipeline loads your checkpoint file.
- `MODEL_KEY` in `src/config.py`:
  - used only when `MODEL_FILE_PATH` is set.
  - ignored for default model mode (default key remains `maest_519l_pytorch`).

## Config notes

- `AUDIO_EXTENSIONS` in `src/config.py` controls which file extensions are included in analysis.
- Removing an extension from `AUDIO_EXTENSIONS` excludes that format from analysis.
- Some formats require ffmpeg conversion before inference; if ffmpeg is unavailable, those files are skipped with per-file errors while pipeline continues.
- Analyze-stage logs include per-track genres and elapsed time: `Analyzed: <file> [genre1, genre2, ...] (time: X.XXs)`.

## Invariants preserved

- MAEST key in JSON: `maest_519l_pytorch`
- Label cleanup: `A---B -> B`
- Extractor fields:
  - `file_path`
  - `file_name`
  - `genres_maest`
  - `confidences`
  - `is_broken_beat`
  - `model_key`
- Tagger safety default: `overwrite_existing = False`
