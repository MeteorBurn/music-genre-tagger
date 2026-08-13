# MAEST 522 Data and Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local, auditable data foundation and lightweight browser UI used to label Microhouse, RoMinimal, and DeepTech-Minimal in rounds of 100 candidates per genre, up to 1,000 candidates per genre.

**Architecture:** A small FastAPI service owns an append-only SQLite annotation database. Importers normalize folders and M3U/M3U8 playlists into one track catalog, grouping duplicates before a frozen group split is assigned. Queue construction is split-aware: train candidates may later be model-ranked, while validation and test candidates remain blind. A vanilla HTML/CSS/JavaScript UI reviews every queued track against all three labels.

**Tech Stack:** Python 3.10, FastAPI, Uvicorn, SQLite, stdlib `wave`/`subprocess`, existing SoundFile/Mutagen, iterative-stratification, vanilla browser APIs.

## Global Constraints

- Do not edit `src/config.py`; this workflow is isolated under `tools/maest522`.
- Treat audio files as read-only. Store absolute paths locally, but never publish them in dataset exports.
- Store four states per target label: `positive`, `negative`, `uncertain`, `unreviewed`.
- A review is complete only after all three labels have a non-`unreviewed` state.
- Freeze `train`/`val`/`test` assignments before model-guided acquisition begins.
- Exact hashes, acoustic fingerprints, release groups, and artist groups may not cross splits.
- Queue rounds contain at most 100 unique tracks per source label; one track can satisfy several source-label quotas without duplicate review.
- Never rank blind validation/test queues with the student model.
- All schema changes use explicit SQLite migrations and preserve the annotation event history.

---

### Task 1: Add the isolated package, dependencies, constants, and annotation schema

**Files:**

- Modify: `.gitignore`
- Create: `requirements-annotation.txt`
- Create: `tools/__init__.py`
- Create: `tools/maest522/__init__.py`
- Create: `tools/maest522/constants.py`
- Create: `tools/maest522/annotation_db.py`
- Create: `tests/maest522/__init__.py`
- Create: `tests/maest522/test_annotation_db.py`

- [ ] Add a failing schema and constants test.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.maest522.annotation_db import AnnotationStore
from tools.maest522.constants import NEW_LABELS, REVIEW_STATES


class AnnotationStoreTest(TestCase):
    def test_initializes_versioned_schema_and_three_labels(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = AnnotationStore(Path(temp_dir) / "annotations.db")
            store.initialize()
            self.assertEqual(NEW_LABELS, (
                "Electronic---Microhouse",
                "Electronic---RoMinimal",
                "Electronic---DeepTech-Minimal",
            ))
            self.assertEqual(REVIEW_STATES, {
                "positive", "negative", "uncertain", "unreviewed"
            })
            self.assertEqual(store.schema_version(), 1)
```

- [ ] Run the test and confirm it fails because the package does not exist.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_annotation_db -v
```

- [ ] Expose only the annotation tool directories through `.gitignore`.

```gitignore
!tools/
!tools/__init__.py
!tools/annotation_ui.py
!tools/maest522/
!tools/maest522/**
```

- [ ] Add bounded annotation-only dependencies.

```text
fastapi>=0.115,<1
uvicorn>=0.34,<1
iterative-stratification>=0.1.9,<1
```

- [ ] Implement constants and a version-1 schema with these tables.

```python
NEW_LABELS = (
    "Electronic---Microhouse",
    "Electronic---RoMinimal",
    "Electronic---DeepTech-Minimal",
)
REVIEW_STATES = {"positive", "negative", "uncertain", "unreviewed"}
ROUND_SIZE_PER_LABEL = 100
MAX_CANDIDATES_PER_LABEL = 1_000
SPLITS = ("train", "val", "test")
```

```sql
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    path TEXT NOT NULL,
    exact_sha256 TEXT NOT NULL,
    acoustic_fingerprint TEXT,
    duration_seconds REAL NOT NULL,
    artist TEXT,
    release_id TEXT,
    group_id TEXT,
    split TEXT CHECK(split IN ('train','val','test')),
    UNIQUE(project_id, exact_sha256)
);
CREATE TABLE sources (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    kind TEXT NOT NULL CHECK(kind IN ('folder','m3u','m3u8')),
    source_path TEXT NOT NULL,
    candidate_role TEXT NOT NULL CHECK(candidate_role IN ('positive_candidate','hard_negative_candidate','unlabeled_pool')),
    imported_at TEXT NOT NULL
);
CREATE TABLE track_sources (
    track_id INTEGER NOT NULL REFERENCES tracks(id),
    source_id INTEGER NOT NULL REFERENCES sources(id),
    suggested_label TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(track_id, source_id, suggested_label)
);
CREATE TABLE queue_items (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    track_id INTEGER NOT NULL REFERENCES tracks(id),
    round_number INTEGER NOT NULL,
    acquisition_kind TEXT NOT NULL,
    acquisition_score REAL,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, track_id)
);
CREATE TABLE annotation_events (
    id INTEGER PRIMARY KEY,
    queue_item_id INTEGER NOT NULL REFERENCES queue_items(id),
    label TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('positive','negative','uncertain','unreviewed')),
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE schema_meta (version INTEGER NOT NULL);
```

Use WAL, foreign keys, a 30-second busy timeout, parameterized SQL, and transactional migrations. Current annotation state is the latest event per `(queue_item_id, label)`; do not overwrite prior events.

- [ ] Run the focused test and inspect the schema with `PRAGMA foreign_key_check`.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_annotation_db -v
```

- [ ] Commit the package foundation.

```powershell
git add .gitignore requirements-annotation.txt tools/__init__.py tools/maest522/__init__.py tools/maest522/constants.py tools/maest522/annotation_db.py tests/maest522
git commit -m "feat: add MAEST 522 annotation store"
```

### Task 2: Import folders and M3U/M3U8 playlists into one deduplicated catalog

**Files:**

- Create: `tools/maest522/library.py`
- Create: `tools/maest522/playlists.py`
- Create: `tests/maest522/test_library_import.py`
- Create: `tests/maest522/test_playlists.py`

- [ ] Add tests for UTF-8/UTF-8-BOM playlists, relative paths, `file:///` entries, comments, missing files, and overlapping folder/playlist sources.

```python
entries = parse_playlist(playlist_path)
self.assertEqual(entries, [
    (playlist_path.parent / "relative" / "one.flac").resolve(),
    Path("D:/Music/two.mp3"),
])
```

The catalog test must assert that importing the same audio through two sources produces one `tracks` row and two `track_sources` rows.

- [ ] Run both tests and confirm the missing modules fail.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_playlists tests.maest522.test_library_import -v
```

- [ ] Implement the parser and catalog importer with these public interfaces.

```python
def parse_playlist(playlist_path: Path) -> list[Path]: ...

def discover_audio_files(folder: Path) -> list[Path]: ...

def import_source(
    store: AnnotationStore,
    project_id: int,
    source_path: Path,
    suggested_label: str | None,
    candidate_role: str,
) -> ImportSummary: ...
```

Supported extensions must match the project pipeline: `.flac`, `.wav`, `.aif`, `.aiff`, `.mp3`, `.m4a`, `.dsf`, `.ape`, `.wv`. Resolve a playlist-relative entry against the playlist directory. Record missing/unreadable entries in `ImportSummary.errors` and continue.

Reject new imports after split freeze so the candidate universe cannot change underneath fixed holdouts. A separately created project is the recovery path when more source material must be added.

- [ ] Hash each file by content with streaming SHA-256 and read duration/artist/release metadata with Mutagen. Do not reuse the main pipeline's path-derived 16-character hash because it does not detect copied audio.

- [ ] Run the focused tests.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_playlists tests.maest522.test_library_import -v
```

- [ ] Commit import support.

```powershell
git add tools/maest522/library.py tools/maest522/playlists.py tests/maest522/test_library_import.py tests/maest522/test_playlists.py
git commit -m "feat: import annotation folders and playlists"
```

### Task 3: Compute duplicate groups and freeze leakage-safe splits

**Files:**

- Create: `tools/maest522/fingerprints.py`
- Create: `tools/maest522/splits.py`
- Create: `tests/maest522/test_fingerprints.py`
- Create: `tests/maest522/test_splits.py`

- [ ] Add a test that mocks `fpcalc -json` and a split test containing exact duplicates, matching acoustic fingerprints, the same release, and the same normalized artist.

```python
assignments = assign_group_splits(tracks, seed=522)
self.assertEqual(assignments["original"], assignments["copied-file"])
self.assertEqual(assignments["radio-edit"], assignments["full-mix"])
assert_no_group_leakage(tracks, assignments)
```

- [ ] Run the tests and confirm failure.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_fingerprints tests.maest522.test_splits -v
```

- [ ] Implement `fpcalc` invocation without shell interpolation.

```python
def calculate_fingerprint(audio_path: Path, fpcalc_path: Path) -> FingerprintResult:
    completed = subprocess.run(
        [str(fpcalc_path), "-json", str(audio_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    ...
```

Missing `fpcalc` is a visible project warning, not silent success. A split cannot be finalized until every usable track has either a fingerprint or an explicit `fingerprint_unavailable` audit record.

- [ ] Build connected components with union-find. Union tracks sharing an exact SHA-256, acoustic fingerprint, normalized release ID, or normalized artist. Set `group_id` to SHA-256 of the sorted member exact hashes so it is deterministic.

- [ ] Assign whole groups 70/15/15 using multilabel iterative stratification over the imported source-label hints. Resolve rounding deterministically with seed `522`, persist assignments once, and reject later reassignment unless the caller uses a separately named destructive rebuild command.

```python
def freeze_group_splits(
    store: AnnotationStore,
    project_id: int,
    seed: int = 522,
    proportions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> SplitSummary: ...

def audit_split_leakage(store: AnnotationStore, project_id: int) -> LeakageAudit: ...
```

- [ ] Run tests and verify deterministic output across two fresh databases.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_fingerprints tests.maest522.test_splits -v
```

- [ ] Commit split logic.

```powershell
git add tools/maest522/fingerprints.py tools/maest522/splits.py tests/maest522/test_fingerprints.py tests/maest522/test_splits.py
git commit -m "feat: add leakage-safe annotation splits"
```

### Task 4: Generate 100-track rounds and export a training manifest

**Files:**

- Create: `tools/maest522/queues.py`
- Create: `tools/maest522/manifests.py`
- Create: `tests/maest522/test_queues.py`
- Create: `tests/maest522/test_manifests.py`

- [ ] Add tests proving per-label quotas, overlap deduplication, the 1,000-per-label cap, immutable split assignments, and rejection of student scores for `val`/`test`.

```python
round_summary = create_round(store, project_id, round_number=1)
self.assertLessEqual(round_summary.source_counts[NEW_LABELS[0]], 100)
self.assertEqual(round_summary.unique_tracks, 230)
with self.assertRaises(ValueError):
    create_round(store, project_id, round_number=2, split="val", student_scores={1: 0.9})
```

- [ ] Run the tests and confirm failure.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_queues tests.maest522.test_manifests -v
```

- [ ] Implement deterministic round construction. Round 1 samples by source hint and seed, reserving 25 of each 100 source-label slots for `hard_negative_candidate` tracks when that pool is large enough. Later train rounds accept acquisition scores but still enforce source quotas, group uniqueness, hard-negative coverage, and global caps. Validation/test queues are sampled once from their frozen pools and never re-ranked.

```python
def create_round(
    store: AnnotationStore,
    project_id: int,
    round_number: int,
    split: str = "train",
    student_scores: Mapping[int, float] | None = None,
) -> QueueSummary: ...
```

- [ ] Export UTF-8 JSONL with one row per reviewed track and no local absolute path when `portable=True`.

```json
{"track_id":"sha256:...","group_id":"group:...","audio_ref":"audio/sha256...flac","split":"train","duration_seconds":384.2,"window_offsets_seconds":[61.84,177.1,292.36],"source_labels":["Electronic---Microhouse"],"labels":{"Electronic---Microhouse":"positive","Electronic---RoMinimal":"negative","Electronic---DeepTech-Minimal":"uncertain"}}
```

`uncertain` is preserved and later masked; `unreviewed` rows are excluded. Also export `dataset_summary.json` containing counts by split/state/source and the split-audit hash.

- [ ] Run focused tests.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_queues tests.maest522.test_manifests -v
```

- [ ] Commit queues and exports.

```powershell
git add tools/maest522/queues.py tools/maest522/manifests.py tests/maest522/test_queues.py tests/maest522/test_manifests.py
git commit -m "feat: build MAEST 522 annotation rounds"
```

### Task 5: Serve safe local audio previews

**Files:**

- Create: `tools/maest522/audio_preview.py`
- Create: `tests/maest522/test_audio_preview.py`

- [ ] Add tests for project-bound path lookup, byte ranges, seek offsets, ffmpeg conversion arguments, and rejection of arbitrary filesystem paths.

- [ ] Implement lookup by integer track ID only. Resolve the stored path and verify it belongs to the selected project before opening it. Serve browser-compatible files with HTTP range support; convert unsupported formats into a task-specific cache below the annotation project directory.

```python
def build_preview_response(
    store: AnnotationStore,
    project_id: int,
    track_id: int,
    range_header: str | None,
) -> Response: ...
```

Invoke ffmpeg as an argument list with `-nostdin -vn -ac 2 -ar 44100 -c:a libmp3lame`; never interpolate paths into a command string. Cache names use exact SHA-256, not user filenames.

- [ ] Run the focused test.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_audio_preview -v
```

- [ ] Commit preview serving.

```powershell
git add tools/maest522/audio_preview.py tests/maest522/test_audio_preview.py
git commit -m "feat: serve local annotation previews"
```

### Task 6: Implement the FastAPI contract and keyboard-first review UI

**Files:**

- Create: `tools/maest522/annotation_api.py`
- Create: `tools/maest522/static/index.html`
- Create: `tools/maest522/static/app.js`
- Create: `tools/maest522/static/styles.css`
- Create: `tools/annotation_ui.py`
- Create: `tests/maest522/test_annotation_api.py`
- Create: `tests/maest522/test_annotation_ui_assets.py`

- [ ] Add API tests for project creation, source import, split freeze, round creation, next item, annotation append, undo-by-new-event, progress, and manifest export.

```text
POST /api/projects
POST /api/projects/{project_id}/sources
POST /api/projects/{project_id}/fingerprints
POST /api/projects/{project_id}/splits/freeze
POST /api/projects/{project_id}/rounds
GET  /api/projects/{project_id}/queue/next
POST /api/projects/{project_id}/queue/{queue_item_id}/annotations
GET  /api/projects/{project_id}/progress
GET  /api/projects/{project_id}/export
GET  /api/projects/{project_id}/audio/{track_id}
```

- [ ] Run the tests and confirm failure.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_annotation_api tests.maest522.test_annotation_ui_assets -v
```

- [ ] Implement Pydantic request/response models with explicit enums and actionable 4xx errors. Bind to `127.0.0.1` by default. Do not enable permissive CORS.

- [ ] Implement one review screen containing full-track playback, waveform-free seek controls, 20%/50%/80% jump buttons, three four-state label rows, notes, save/skip/back, queue progress, and visible source hints. Do not display teacher/student scores during review.

Keyboard contract:

```text
Space       play/pause
1/2/3       focus Microhouse/RoMinimal/DeepTech-Minimal
P/N/U/X     positive/negative/uncertain/unreviewed
J/K/L       jump to 20%/50%/80%
Enter       save and advance
Backspace   previous item
```

- [ ] Add an M3U/M3U8 upload control that sends playlist text plus the playlist base directory. Also retain a server-path field for large playlists and folder imports.

- [ ] Implement the entry point.

```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8522)
    ...
```

- [ ] Run focused API and asset-contract tests.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_annotation_api tests.maest522.test_annotation_ui_assets -v
```

- [ ] Commit the UI.

```powershell
git add tools/maest522/annotation_api.py tools/maest522/static tools/annotation_ui.py tests/maest522/test_annotation_api.py tests/maest522/test_annotation_ui_assets.py
git commit -m "feat: add MAEST 522 annotation UI"
```

### Task 7: Verify a complete local annotation round and document operator commands

**Files:**

- Create: `docs/maest522-annotation-guide.md`
- Create: `tests/maest522/test_annotation_workflow.py`

- [ ] Add an end-to-end test using short generated WAV fixtures: import overlapping folder and M3U sources, mock fingerprints, freeze splits, build round 1, annotate all labels, reopen the database, and export JSONL.

- [ ] Run the complete annotation test group.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/maest522 -p "test_*annotation*.py" -v
```

- [ ] Document the exact PowerShell workflow.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-annotation.txt
.\.venv\Scripts\python.exe tools\annotation_ui.py --db .\annotation-projects\maest522\annotations.db
```

The guide must cover: create project; import one folder or uploaded M3U; review import errors; run fingerprints; freeze and audit splits; create the first 100-per-label round; resume after interruption; export manifests; back up the SQLite database. It must state that later train rounds receive scores from the active-learning command in the training plan.

- [ ] Run syntax and CLI checks.

```powershell
.\.venv\Scripts\python.exe -m py_compile tools\annotation_ui.py
.\.venv\Scripts\python.exe -m compileall -q tools\maest522
.\.venv\Scripts\python.exe tools\annotation_ui.py --help
```

- [ ] Inspect the diff for machine-specific paths and commit.

```powershell
rg -n "[A-Z]:\\\\Users\\\\|E:\\\\Projects" tools tests docs/maest522-annotation-guide.md
git diff --check
git add docs/maest522-annotation-guide.md tests/maest522/test_annotation_workflow.py
git commit -m "docs: add MAEST 522 annotation workflow"
```

## Completion Gate

- A 300-source-slot round can contain fewer than 300 unique tracks when source labels overlap, but never more than 100 selections credited to one label.
- Restarting the UI preserves queue position and all event history.
- Every completed item has three explicit review states.
- The leakage audit is clean and its digest is embedded in exports.
- No student score has influenced validation or test membership/order.
- The export is sufficient for the training plan without querying UI internals.
