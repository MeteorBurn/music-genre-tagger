# AGENTS

This file defines practical guidance for coding agents working in this repository.

## Scope

- Project root: `E:/Projects/MusicTagger`
- Main workflow:
  1. `04_universal_audio_analyzer.py` (audio -> JSON)
  2. `05_json_to_excel_extractor.py` (JSON -> Excel)
  3. `06_audio_genres_tagger.py` (Excel -> audio tags)

## Environment

- Use virtual environment: `.venv-maest`
- Python entrypoint for checks: `00_check_environment.py`
- Dependencies source of truth: `requirements.txt`

## Key conventions

- Keep paths relative where possible (`models`, `json`, `analysis.xlsx`).
- Resolve relative paths from script directory (not current shell directory).
- MAEST model key in JSON must remain: `maest_519l_pytorch`.
- Genre labels should be stored without top-level prefix (`A---B` -> `B`).
- In tagger, keep `overwrite_existing = False` unless explicitly requested.

## Safety rules

- Do not mass-edit audio metadata unless explicitly requested.
- Do not change checkpoint filename or model arch silently.
- Do not install packages automatically unless user asked.

## Verification checklist

Before considering changes complete:

1. Run syntax check for edited Python files.
2. Run `00_check_environment.py` in `.venv-maest`.
3. If analyzer changed, run a small smoke test (`max_files=1`).
4. If Excel scripts changed, verify `analysis.xlsx` columns and autosize still work.
