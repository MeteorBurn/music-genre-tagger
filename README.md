# MusicTagger

One-flow cross-platform pipeline for genre extraction with MAEST (PyTorch), Excel export, optional metadata tagging, and final markdown report.

## New entry point

- `main.py` is the only required script for running the full flow.
- Internal modules:
  - `environment.py`
  - `pipeline.py`
  - `extractor.py`
  - `tagger.py`
  - `config.py`

## What the flow does

1. Environment diagnostics with actionable fix instructions.
2. Audio analysis to per-track JSON (`maest_519l_pytorch`).
3. JSON aggregation to Excel (`tracks_genres.xlsx`) with dedupe by `file_path`.
4. Optional genre tag writing into audio metadata.
5. Final markdown report (`report.md`).

## Setup (PowerShell)

```powershell
cd E:\Projects\MusicTagger
python -m venv .venv-maest
.\.venv-maest\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## CLI usage

Run all stages (default):

```powershell
python .\main.py
```

Run a specific stage:

```powershell
python .\main.py --stage analyze
python .\main.py --stage excel
python .\main.py --stage tag
```

Stage options:

- `--stage all|analyze|excel|tag` (default `all`)
- `--input-directory <path>`
- `--output-directory <path>`
- `--json-directory <path>`
- `--excel-path <path>`
- `--report-path <path>`
- `--file-pattern <text>`
- `--max-files <N>`
- `--convert-to-wav`
- `--tag-yes` or `--tag-no` (for `--stage all`)
- `--non-interactive`

## Paths and meta structure

- If `input_directory` is empty and the selected stage needs analysis, the CLI asks for it.
- If `output_directory` is empty and the selected stage uses analyze flow, the CLI asks for output base path.
- If user skips output base path, project-relative `meta/` is used.
- Meta root name is based on input directory name.
- Required structure inside meta root:
  - `json/`
  - `tracks_genres.xlsx`
  - `report.md`

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
