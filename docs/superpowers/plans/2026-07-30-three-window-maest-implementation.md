# Three-window MAEST Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Music Genre Tagger to `maest-infer==0.2.0` and derive each track's top-three genres from the mean sigmoid scores of up to three duration-aware 30-second windows.

**Architecture:** Keep orchestration in `pipeline.py`, but extract window selection and score aggregation into small pure functions that can be tested without loading MAEST. Replace the legacy analysis columns with a version-2 schema guarded by SQLite `user_version`, and carry the new metadata through the existing database-to-JSON path.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, maest-infer 0.2.0, SQLite, unittest.

## Global Constraints

- Require exactly `maest-infer==0.2.0`.
- Candidate window centers are 20%, 50%, and 80% of decoded duration.
- Target window duration is exactly 30 seconds.
- Tracks no longer than 30 seconds are analyzed once without padding.
- Deduplicate clamped starts while preserving chronological order.
- Average complete sigmoid score vectors before selecting the top three labels.
- No migration or backward compatibility for pre-2.0 databases.
- Preserve label cleanup as `label.split("---", 1)[-1]`.
- Preserve per-file failure isolation and the non-destructive tagging default.

---

### Task 1: Duration-aware window selection

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_pipeline_windows.py`
- Modify: `.gitignore`
- Modify: `src/config.py:29-33`
- Modify: `src/pipeline.py:282-297`

**Interfaces:**
- Produces: `select_audio_windows(audio: np.ndarray, sample_rate: int, window_duration_sec: float, positions: Sequence[float]) -> List[Tuple[float, np.ndarray]]`
- Each tuple contains the actual start offset in seconds and a non-empty view/copy of the decoded mono audio.

- [ ] **Step 1: Make the test package trackable**

Add these exceptions after the default directory ignore in `.gitignore`:

```gitignore
!tests/
!tests/**
```

Create an empty `tests/__init__.py`.

- [ ] **Step 2: Write failing table-driven tests**

Add `tests/test_pipeline_windows.py` with literal expected starts for decoded
audio lasting 20, 30, 40, 60, 180, and 300 seconds:

```python
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline import select_audio_windows


class SelectAudioWindowsTests(unittest.TestCase):
    def test_selects_duration_aware_window_starts(self):
        cases = [
            (20, [0.0]),
            (30, [0.0]),
            (40, [0.0, 5.0, 10.0]),
            (60, [0.0, 15.0, 30.0]),
            (180, [21.0, 75.0, 129.0]),
            (300, [45.0, 135.0, 225.0]),
        ]
        sample_rate = 10
        for duration, expected_starts in cases:
            with self.subTest(duration=duration):
                audio = np.arange(duration * sample_rate, dtype=np.float32)
                windows = select_audio_windows(
                    audio, sample_rate, 30.0, (0.2, 0.5, 0.8)
                )
                self.assertEqual([offset for offset, _ in windows], expected_starts)

    def test_slices_using_sample_boundaries(self):
        sample_rate = 10
        audio = np.arange(60 * sample_rate, dtype=np.float32)
        windows = select_audio_windows(audio, sample_rate, 30.0, (0.2, 0.5, 0.8))
        self.assertEqual([window[0] for _, window in windows], [0.0, 150.0, 300.0])
        self.assertTrue(all(len(window) == 300 for _, window in windows))

    def test_rejects_empty_audio(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            select_audio_windows(np.array([], dtype=np.float32), 16000, 30.0, (0.2, 0.5, 0.8))
```

These tests catch incorrect center arithmetic, failure to clamp, duplicate
starts, padding of short tracks, and slicing in seconds rather than samples.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_windows -v
```

Expected: import failure because `select_audio_windows` does not exist.

- [ ] **Step 4: Implement the smallest window selector**

Replace `AUDIO_OFFSET` and `AUDIO_DURATION` with:

```python
AUDIO_WINDOW_DURATION = 30
AUDIO_WINDOW_POSITIONS = (0.2, 0.5, 0.8)
```

Return them from `get_config()` as `audio_window_duration` and
`audio_window_positions`. Replace `trim_audio_segment()` with a selector that:

1. validates non-empty audio, positive sample rate, positive duration, and
   positions in `[0, 1]`;
2. returns the complete audio at offset `0.0` when its duration is no greater
   than the target;
3. converts the clamped candidate starts to integer sample indices;
4. deduplicates sample indices;
5. returns chronological `(start_sample / sample_rate, slice)` tuples.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the same unittest command. Expected: all window tests pass.

- [ ] **Step 6: Commit the slice**

```powershell
git add .gitignore src/config.py src/pipeline.py tests/__init__.py tests/test_pipeline_windows.py
git commit -m "feat: select duration-aware MAEST windows"
```

---

### Task 2: Mean score aggregation and three-window inference

**Files:**
- Create: `tests/test_pipeline_predictions.py`
- Modify: `src/pipeline.py:466-600`

**Interfaces:**
- Consumes: `select_audio_windows(...)` from Task 1.
- Produces: `aggregate_window_predictions(window_predictions: Sequence[Tuple[np.ndarray, Sequence[str]]]) -> Tuple[np.ndarray, List[str]]`
- `analyze_audio_file(...)` stores top-N labels selected from the aggregated vector.

- [ ] **Step 1: Write failing aggregation tests**

Create tests using three complete literal score vectors:

```python
class AggregateWindowPredictionsTests(unittest.TestCase):
    def test_averages_scores_before_top_n_selection(self):
        predictions = [
            (np.array([0.9, 0.2, 0.2, 0.0]), ["A", "B", "C", "D"]),
            (np.array([0.0, 0.8, 0.2, 0.1]), ["A", "B", "C", "D"]),
            (np.array([0.0, 0.2, 0.9, 0.2]), ["A", "B", "C", "D"]),
        ]
        scores, labels = aggregate_window_predictions(predictions)
        np.testing.assert_allclose(scores, [0.3, 0.4, 0.43333333, 0.1])
        self.assertEqual(labels, ["A", "B", "C", "D"])
        self.assertEqual(
            process_predictions(scores, labels, 3),
            [("C", scores[2]), ("B", scores[1]), ("A", scores[0])],
        )

    def test_rejects_changed_label_order(self):
        predictions = [
            (np.array([0.2, 0.8]), ["A", "B"]),
            (np.array([0.8, 0.2]), ["B", "A"]),
        ]
        with self.assertRaisesRegex(ValueError, "label vocabulary"):
            aggregate_window_predictions(predictions)

    def test_rejects_non_vector_scores(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            aggregate_window_predictions([(np.array([[0.2, 0.8]]), ["A", "B"])])
```

Add a controlled `analyze_audio_file` test that patches only audio decoding,
uses a fake MAEST object returning distinct real score vectors for three
calls, and asserts the returned payload contains the mean-derived top three
and three offsets. Do not assert that the fake was called merely for its own
sake; assert the real payload produced by `analyze_audio_file`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_predictions -v
```

Expected: import failure because `aggregate_window_predictions` does not exist.

- [ ] **Step 3: Implement aggregation**

Implement `aggregate_window_predictions()` to:

1. reject an empty prediction list;
2. normalize each score output with `np.asarray(scores)`;
3. require `ndim == 1`;
4. require score count to equal label count;
5. require identical ordered labels for every window;
6. return `np.mean(np.stack(score_vectors), axis=0)` and the common labels.

- [ ] **Step 4: Integrate the selector into `analyze_audio_file`**

Decode once, call `select_audio_windows()`, run `model.predict_labels()` once
per unique window, aggregate the raw returned scores, clean labels once, and
call `process_predictions()` only on the mean vector.

Set payload metadata to:

```python
"analysis_config": {
    "audio_segment_offsets": offsets,
    "audio_segment_duration": config["audio_window_duration"],
    "audio_segment_count": len(windows),
    "aggregation": "mean",
}
```

Any inference exception must flow to the existing per-file `error` handling;
do not retain partial predictions.

- [ ] **Step 5: Run focused and window regression tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_predictions tests.test_pipeline_windows -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the slice**

```powershell
git add src/pipeline.py tests/test_pipeline_predictions.py
git commit -m "feat: average predictions across MAEST windows"
```

---

### Task 3: Version-2 SQLite and JSON metadata

**Files:**
- Create: `tests/test_storage_v2.py`
- Modify: `src/storage.py:36-338`

**Interfaces:**
- Consumes: the new `analysis_config` payload from Task 2.
- Produces: schema version `2`, `audio_segment_offsets` JSON text,
  `audio_segment_duration` REAL, `audio_segment_count` INTEGER, and
  `aggregation` TEXT.

- [ ] **Step 1: Write failing storage tests**

Use `tempfile.TemporaryDirectory()` and real SQLite:

```python
class StorageV2Tests(unittest.TestCase):
    def test_round_trips_three_window_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tracks.db"
            init_db(db_path)
            upsert_track(db_path, build_track_payload())
            [record] = load_track_records(db_path)
            self.assertEqual(json.loads(record["audio_segment_offsets"]), [21.0, 75.0, 129.0])
            self.assertEqual(record["audio_segment_duration"], 30.0)
            self.assertEqual(record["audio_segment_count"], 3)
            self.assertEqual(record["aggregation"], "mean")

            [exported] = export_tracks_json(db_path, Path(directory) / "tracks.json")
            payload = json.loads(exported.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["tracks"][0]["analysis_config"]["audio_segment_offsets"],
                [21.0, 75.0, 129.0],
            )

    def test_rejects_legacy_database(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "tracks.db"
            with sqlite3.connect(db_path) as connection:
                connection.execute("CREATE TABLE tracks (hash TEXT PRIMARY KEY)")
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                init_db(db_path)
```

The fixture must include all real fields consumed by `_flatten_track`; do not
use a partial payload that bypasses production behavior.

- [ ] **Step 2: Run tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_storage_v2 -v
```

Expected: missing new columns/metadata and no legacy-schema rejection.

- [ ] **Step 3: Implement schema version 2**

Before creating tables, query `sqlite_master` for `tracks`.

- Existing `tracks` plus `PRAGMA user_version != 2`: raise `RuntimeError` with
  instructions to move or delete the incompatible database.
- New database: create the new columns, then set `PRAGMA user_version = 2`.
- Existing `tracks` plus version `2`: continue normally.

Replace the old offset column in `TRACK_COLUMNS`, flatten the new metadata with
JSON serialization, and reconstruct the same metadata in
`_build_track_payload()`.

- [ ] **Step 4: Run storage and pipeline tests**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the slice**

```powershell
git add src/storage.py tests/test_storage_v2.py
git commit -m "feat: store version 2 window metadata"
```

---

### Task 4: Dependency, documentation, and integrated verification

**Files:**
- Create: `tests/test_environment.py`
- Modify: `requirements.txt`
- Modify: `src/environment.py:417-520`
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: final runtime and storage behavior from Tasks 1-3.
- Produces: exact dependency pin and current operator/developer documentation.

- [ ] **Step 1: Write a failing MAEST version check**

Add a focused environment test that patches
`importlib.metadata.version("maest-infer")` to return `0.1.0` and asserts that
the new `_check_maest_version()` reports failure with required version
`0.2.0`. Add the corresponding success case for `0.2.0`.

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_environment -v
```

Expected: import failure because `_check_maest_version` does not exist.

- [ ] **Step 2: Pin and enforce the dependency**

Change:

```text
maest-infer>=0.1.0
```

to:

```text
maest-infer==0.2.0
```

Do not upgrade unrelated dependencies.

Implement `_check_maest_version(required_version: str = "0.2.0")` using
`importlib.metadata.version("maest-infer")`. Call it from
`run_environment_checks()` and include its result in the final success
condition so an existing 0.1.x environment cannot pass merely because the
module imports.

- [ ] **Step 3: Run the version tests and verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_environment -v
```

Expected: both mismatch and exact-match cases pass.

- [ ] **Step 4: Update user documentation**

Update README configuration, pipeline flow, track JSON example, and behavior
notes to describe:

- window centers at 20%, 50%, and 80%;
- clamping/deduplication and one-window short-track behavior;
- mean score aggregation before top-three selection;
- version-2 database incompatibility;
- removal of `AUDIO_OFFSET`;
- exact MAEST 0.2.0 requirement.

- [ ] **Step 5: Update repository instructions**

Update `AGENTS.md` repository map, pipeline flow, data model, invariants,
configuration contract, and verification checklist. Remove every statement
that says analysis uses one scalar offset.

- [ ] **Step 6: Run the full focused verification**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m py_compile src/config.py src/environment.py src/pipeline.py src/extractor.py src/tagger.py src/report.py src/main.py src/storage.py
.\.venv\Scripts\python.exe src/main.py --help
```

Expected: tests pass, compilation exits `0`, and help exits `0`.

- [ ] **Step 7: Inspect final repository changes**

```powershell
git status --short
git diff --check
git diff --stat
git diff
```

Confirm no model files, user audio, databases, spreadsheets, or unrelated
changes are included.

- [ ] **Step 8: Run the real one-track smoke test when available**

Use a temporary output directory and one explicitly selected local test track.
Do not tag source files. Verify the stored record has three offsets for a
sufficiently long track and that its top-three scores came from aggregation.
If the checkpoint or suitable test audio is unavailable, record that exact
gap instead of downloading or modifying user data without confirmation.

- [ ] **Step 9: Commit documentation and dependency changes**

```powershell
git add requirements.txt src/environment.py tests/test_environment.py README.md AGENTS.md
git commit -m "docs: describe version 2 MAEST analysis"
```
