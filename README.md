# MusicTagger

Windows pipeline for genre extraction with MAEST (PyTorch) and optional writing genres into audio tags.

## What this project does

1. Analyze audio files with MAEST and save one JSON per track.
2. Collect JSON results into an Excel table.
3. Write predicted genres from Excel back into file metadata tags.

## Scripts

- `00_check_environment.py` - checks Python/modules from `requirements.txt`, MAEST API availability, FFmpeg, model file, and CPU/CUDA runtime.
- `04_universal_audio_analyzer.py` - batch audio analysis with MAEST (`maest_519l_pytorch`), outputs JSON files.
- `05_json_to_excel_extractor.py` - builds/updates `analysis.xlsx` from JSON outputs, skips duplicates by `file_path`, auto-adjusts column widths.
- `06_audio_genres_tagger.py` - writes genres from Excel into audio tags using `mutagen`, updates `status`, auto-adjusts column widths.

## Requirements

Defined in `requirements.txt`:

- `numpy`, `soundfile`
- `torch`, `torchaudio`
- `maest-infer`
- `pandas`, `openpyxl`
- `mutagen`

## Setup (PowerShell)

```powershell
cd E:\Projects\MusicTagger
python -m venv .venv-maest
.\.venv-maest\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional CUDA 11.8 wheels (if needed explicitly):

```powershell
pip install --upgrade --force-reinstall torch==2.6.0+cu118 torchaudio==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Validate environment:

```powershell
python .\00_check_environment.py
```

## Required files and folders

- `models/discogs-maest-30s-pw-129e-519l-swa.ckpt`
- input music folder path in `04_universal_audio_analyzer.py` (`CONFIG["input_directory"]`)

Notes:

- `04_universal_audio_analyzer.py` resolves relative paths from the script directory, so `models` and `json` work from any current shell folder.
- Genres in JSON are saved without top-level prefix (`Electronic---Ambient` becomes `Ambient`).

## Run order

1) Analyze audio -> JSON:

```powershell
python .\04_universal_audio_analyzer.py
```

2) JSON -> Excel:

```powershell
python .\05_json_to_excel_extractor.py
```

3) Excel -> audio tags:

```powershell
python .\06_audio_genres_tagger.py
```

## Default behavior highlights

- Analyzer model key in JSON: `maest_519l_pytorch`
- Excel source key for genres: `genres_maest`
- Tagging overwrite policy: `overwrite_existing = False` (existing genre tags are preserved)
- Status tracking in Excel: `status` column (`success`, `skipped`, `error`)
