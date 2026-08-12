# Sequential Single-Label Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three-label-at-once annotation workflow with an auditable, sequential, single-label workflow that atomically imports trusted positive/negative M3U batches, preserves partial labels as masked supervision, and safely migrates the current 100/100 Minimal-Deep-Tech seed project.

**Architecture:** Add a schema-v2 confirmed-label event ledger alongside the existing candidate-source tables. Trusted playlist imports use a read-only preflight followed by a digest-bound atomic commit; the UI persists one active label and reports editable soft goals. All confirmed tracks are fingerprinted and assigned to one final group split only after all three labels meet their goals, and manifest loading masks absent or uncertain labels.

**Tech Stack:** Python 3.10, SQLite/WAL, FastAPI, Pydantic v2, vanilla JavaScript/CSS, `unittest`, PyTorch manifest tensors, PowerShell 7.

## Global Constraints

- The exact appended label order is index 519 `Electronic---Minimal-Deep-Tech`, index 520 `Electronic---Microhouse`, index 521 `Electronic---RoMinimal`.
- Preserve all original 519 label strings, classifier rows, biases, and ordering exactly.
- Accept legacy `Electronic---DeepTech-Minimal` only inside the explicit v1-to-v2 migration; never emit or accept it in v2 runtime contracts.
- Work on branch `codex/maest-522-extension` in `E:\Projects\music-genre-tagger\.worktrees\maest522`.
- Use `E:\Projects\music-genre-tagger\.venv\Scripts\python.exe` for Python commands.
- Write tests before production code and observe every new test fail for the intended missing behavior.
- Do not modify source audio.
- Do not silently migrate a v1 database at application startup.
- Trusted M3U commits are atomic; opposite-state conflicts reject the entire batch.
- Positive and negative goals are independent soft targets, default `1000/1000`, and never delete data when lowered.
- Missing and uncertain labels are mask `0`, never negative supervision.
- Candidate selection remains outside this application; do not scan `M:\Volumes` as part of this feature.
- Do not train, export a final dataset, or run test evaluation until all three labels meet their goals and the final split audit passes.
- Preserve unrelated user changes and never commit local databases, playlists, logs, checkpoints, or audio.

---

## File Responsibility Map

- `tools/maest522/constants.py`: canonical public label order, schema version, and default goals.
- `tools/maest522/annotation_db.py`: schema-v2 DDL, configured SQLite connections, and low-level project initialization.
- `tools/maest522/migrate_annotation_db.py`: explicit backup-first v1-to-v2 migration and trusted seed conversion CLI.
- `tools/maest522/confirmed_labels.py`: current-event queries, goals, progress, corrections, and batch history.
- `tools/maest522/trusted_import.py`: trusted M3U preflight and digest-bound atomic commit.
- `tools/maest522/library.py`: reusable audio identity inspection without database side effects.
- `tools/maest522/queues.py`: one-label queue creation and uncertain replacement semantics.
- `tools/maest522/annotation_api.py`: strict request/response models and project-scoped endpoints.
- `tools/maest522/static/index.html`: sequential workflow layout and Russian operator copy.
- `tools/maest522/static/app.js`: active-label state, preflight/commit flow, goals, progress, and one-label review.
- `tools/maest522/static/styles.css`: progress and preflight result presentation.
- `tools/maest522/splits.py`: split only the confirmed-track union and audit per-label coverage.
- `tools/maest522/manifests.py`: export current confirmed states and masks for all three labels.
- `tools/maest522/training_data.py`: decode `unreviewed` and `uncertain` as masked targets.
- `docs/maest522-annotation-guide.md`: Russian sequential operator guide.
- `docs/maest522-training-guide.md`: final label order and partial-label training contract.
- `docs/maest522-publication-guide.md`: final public label order and provenance wording.

---

### Task 1: Finalize the Global 522 Label Contract

**Files:**
- Modify: `tools/maest522/constants.py`
- Modify: `tests/maest522/test_model_labels.py`
- Modify: `tests/maest522/test_checkpoint.py`
- Modify: `tests/maest522/test_hf_mapping.py`
- Modify: `tests/maest522/test_cards.py`
- Modify: `docs/maest522-annotation-guide.md`
- Modify: `docs/maest522-training-guide.md`
- Modify: `docs/maest522-publication-guide.md`

**Interfaces:**
- Produces: `NEW_LABELS == ("Electronic---Minimal-Deep-Tech", "Electronic---Microhouse", "Electronic---RoMinimal")`.
- Produces: `DEFAULT_POSITIVE_TARGET = 1000`, `DEFAULT_NEGATIVE_TARGET = 1000`, and `SCHEMA_VERSION = 2` for later tasks.
- Preserves: `build_522_labels(labels_519)` appends those three values without changing `labels_519`.

- [ ] **Step 1: Write the failing label-order tests**

Update `test_appends_three_labels_without_reordering_official_labels` to assert:

```python
self.assertEqual(
    labels_522[519:],
    (
        "Electronic---Minimal-Deep-Tech",
        "Electronic---Microhouse",
        "Electronic---RoMinimal",
    ),
)
self.assertEqual(labels_522[:519], tuple(legacy_labels))
```

Add assertions in checkpoint/HF/card tests that row 519 and `id2label["519"]` are Minimal-Deep-Tech, row 520 is Microhouse, and row 521 is RoMinimal.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' -m unittest `
  tests.maest522.test_model_labels `
  tests.maest522.test_checkpoint `
  tests.maest522.test_hf_mapping `
  tests.maest522.test_cards -v
```

Expected: failures show the old `Microhouse, RoMinimal, DeepTech-Minimal` order or old label string.

- [ ] **Step 3: Implement the exact constants**

Set:

```python
NEW_LABELS = (
    "Electronic---Minimal-Deep-Tech",
    "Electronic---Microhouse",
    "Electronic---RoMinimal",
)
DEFAULT_POSITIVE_TARGET = 1_000
DEFAULT_NEGATIVE_TARGET = 1_000
SCHEMA_VERSION = 2
```

Do not add the legacy name to `NEW_LABELS` or any runtime alias map.

- [ ] **Step 4: Update user-facing label tables and verify GREEN**

Replace the old appended label name and order in the three MAEST 522 guides. Re-run the four focused test modules and require exit code 0.

- [ ] **Step 5: Run global old-label sentinels**

Run:

```powershell
rg -n 'Electronic---DeepTech-Minimal|519.*Microhouse|521.*DeepTech' `
  tools tests docs --glob '!docs/superpowers/specs/**' --glob '!docs/superpowers/plans/**'
```

Expected: no runtime/documentation matches. The only allowed old string after Task 2 is inside the explicit migration module and its tests.

- [ ] **Step 6: Commit**

```powershell
git add tools/maest522/constants.py tests/maest522/test_model_labels.py `
  tests/maest522/test_checkpoint.py tests/maest522/test_hf_mapping.py `
  tests/maest522/test_cards.py docs/maest522-annotation-guide.md `
  docs/maest522-training-guide.md docs/maest522-publication-guide.md
git commit -m 'feat: finalize MAEST 522 label contract'
```

---

### Task 2: Add Schema v2 and the Explicit Backup-First Migration

**Files:**
- Modify: `tools/maest522/annotation_db.py`
- Create: `tools/maest522/migrate_annotation_db.py`
- Create: `tests/maest522/test_annotation_migration.py`
- Modify: `tests/maest522/test_annotation_db.py`

**Interfaces:**
- Produces: `AnnotationStore.initialize()` creates fresh schema v2 but rejects schema v1.
- Produces: `migrate_v1_to_v2(database_path: Path, backup_path: Path, project_id: int, positive_source_id: int, negative_source_id: int) -> MigrationReport`.
- Produces: CLI `python -m tools.maest522.migrate_annotation_db --db ... --backup ... --project-id 1 --positive-source-id 1 --negative-source-id 2`.
- `MigrationReport` contains source/backup SHA-256, converted positive/negative counts, schema version, integrity result, and foreign-key violation count.

- [ ] **Step 1: Write failing fresh-schema tests**

Assert fresh projects receive three `label_goals` rows with `1000/1000`, and assert these tables/columns exist:

```text
label_goals(project_id, label, positive_target, negative_target, updated_at)
confirmed_label_batches(id, project_id, label, state, source_kind,
  source_path, playlist_sha256, discovered_count, new_count,
  existing_count, created_at)
confirmed_label_events(id, project_id, track_id, label, state,
  event_kind, batch_id, note, created_at)
```

Also assert `queue_items.label` and `queue_rounds.label` exist and their uniqueness is label-scoped.

- [ ] **Step 2: Write failing migration tests**

Build a disposable v1 database with source IDs 1 and 2, 100 disjoint files each, old suggested label `Electronic---DeepTech-Minimal`, zero queues/events, and valid track metadata. Assert migration:

```python
self.assertEqual(report.converted_positive, 100)
self.assertEqual(report.converted_negative, 100)
self.assertEqual(store.schema_version(), 2)
self.assertEqual(current_counts["Electronic---Minimal-Deep-Tech"], (100, 100))
```

Add failure cases for missing backup path, backup inside the live path, wrong source label/role/project/count, overlap, missing audio, non-empty legacy queue tables, integrity error, and source DB changing during backup.

- [ ] **Step 3: Run migration tests and verify RED**

```powershell
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' -m unittest `
  tests.maest522.test_annotation_db `
  tests.maest522.test_annotation_migration -v
```

Expected: missing schema-v2 tables and migration module failures.

- [ ] **Step 4: Implement schema-v2 DDL**

Extend `SCHEMA_SQL` with the approved tables and indexes. Insert three goal rows in `create_project()`. Add exact label `CHECK` constraints from `NEW_LABELS`; do not accept the old label.

Rebuild fresh `queue_items` and `queue_rounds` definitions with a required `label` and label-scoped uniqueness:

```sql
UNIQUE(project_id, track_id, label)
UNIQUE(project_id, label, round_number, split)
```

- [ ] **Step 5: Implement the explicit migration**

The migration must:

1. Refuse an active writer by obtaining `BEGIN EXCLUSIVE`.
2. Require zero v1 queue rounds/items/events because their all-label semantics cannot be mapped safely.
3. Check exact source IDs, roles, project, old suggested label, 100/100 counts, disjoint SHA-256 identities, and existing files.
4. Checkpoint WAL, copy the database to the explicit backup path, and verify source/backup hashes.
5. Create v2 tables and rebuild empty queue tables with label columns.
6. Update selected v1 source links from the old label to `Electronic---Minimal-Deep-Tech`.
7. Insert one trusted batch and 100 confirmed events for each source.
8. Set schema version 2, commit, reopen, and verify integrity/FKs/counts.

No migration helper is called by `AnnotationStore.initialize()`.

- [ ] **Step 6: Verify GREEN and CLI help**

Run the two test modules, then:

```powershell
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' `
  -m tools.maest522.migrate_annotation_db --help
```

Require all tests and help exit 0.

- [ ] **Step 7: Commit**

```powershell
git add tools/maest522/annotation_db.py `
  tools/maest522/migrate_annotation_db.py `
  tests/maest522/test_annotation_db.py `
  tests/maest522/test_annotation_migration.py
git commit -m 'feat: add confirmed-label schema migration'
```

---

### Task 3: Implement the Confirmed-Label Ledger, Goals, and Progress

**Files:**
- Create: `tools/maest522/confirmed_labels.py`
- Create: `tests/maest522/test_confirmed_labels.py`

**Interfaces:**
- Produces: `LabelGoal`, `LabelProgress`, `ConfirmedBatchSummary`, and `CorrectionResult` frozen dataclasses.
- Produces: `get_label_goals(store, project_id) -> tuple[LabelGoal, ...]`.
- Produces: `update_label_goal(store, project_id, label, positive_target, negative_target) -> LabelGoal`.
- Produces: `get_label_progress(store, project_id) -> tuple[LabelProgress, ...]`.
- Produces: `list_confirmed_batches(store, project_id, label, limit=20) -> tuple[ConfirmedBatchSummary, ...]`.
- Produces: `append_correction(store, project_id, track_id, label, state, reason) -> CorrectionResult`.
- Produces: `current_confirmed_states(connection, project_id) -> dict[tuple[int, str], str]` for import/export tasks.

- [ ] **Step 1: Write failing goal and progress tests**

Cover default goals, independent target edits, rejection of zero/negative/unknown values, progress at/above/below target, uncertain reported separately, and latest-event-wins semantics.

Expected progress shape:

```python
LabelProgress(
    label="Electronic---Minimal-Deep-Tech",
    positive_count=100,
    positive_target=1000,
    negative_count=100,
    negative_target=1000,
    uncertain_count=0,
    complete=False,
)
```

- [ ] **Step 2: Write failing correction tests**

Assert a correction requires a non-empty reason, appends rather than updates, changes current progress, preserves prior event rows, rejects a split-frozen project, and rejects unknown project/track/label/state.

- [ ] **Step 3: Run and verify RED**

```powershell
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' -m unittest `
  tests.maest522.test_confirmed_labels -v
```

Expected: import/module failures because `confirmed_labels.py` does not exist.

- [ ] **Step 4: Implement current-state SQL and domain validation**

Use a `MAX(id)` subquery per `(project_id, track_id, label)`. Keep all mutation functions project-scoped, reject split-frozen projects, and use one transaction per goal update or correction.

- [ ] **Step 5: Verify GREEN**

Re-run `tests.maest522.test_confirmed_labels -v` and require exit 0.

- [ ] **Step 6: Commit**

```powershell
git add tools/maest522/confirmed_labels.py `
  tests/maest522/test_confirmed_labels.py
git commit -m 'feat: add confirmed label ledger'
```

---

### Task 4: Add Read-Only Preflight and Atomic Trusted M3U Commit

**Files:**
- Modify: `tools/maest522/library.py`
- Create: `tools/maest522/trusted_import.py`
- Create: `tests/maest522/test_trusted_import.py`

**Interfaces:**
- Produces: `AudioIdentity(path, exact_sha256, duration_seconds, artist, release_id)`.
- Produces: `inspect_audio_identity(audio_path: Path) -> AudioIdentity` without database writes.
- Produces: `TrustedImportPreflight` with `playlist_sha256`, `discovered`, `new`, `existing`, `missing_paths`, `duplicate_paths`, `conflict_paths`, and normalized entries.
- Produces: `preflight_trusted_playlist(store, project_id, playlist_path, label, state) -> TrustedImportPreflight`.
- Produces: `commit_trusted_playlist(store, project_id, playlist_path, label, state, expected_playlist_sha256) -> ConfirmedBatchSummary`.

- [ ] **Step 1: Write failing clean-preflight tests**

Use temporary WAV fixtures and M3U files. Snapshot database row counts before and after preflight and assert they are identical. Assert 100 valid entries report 100 new and zero errors.

- [ ] **Step 2: Write failing validation tests**

Cover missing audio, unsupported extension, duplicate resolved path, remote URL, opposite current state, unknown label/state, split-frozen project, and two different paths with identical content SHA-256.

Exact duplicate paths reject preflight. Different paths with the same audio SHA-256 collapse to one track identity and are reported as one new identity plus one physical-copy duplicate detail.

- [ ] **Step 3: Write failing commit tests**

Assert:

- valid positive and negative batches create one batch row and one event per new identity;
- same-state reimport is accepted with `new_count=0` and `existing_count=N`;
- changed playlist digest rejects with no writes;
- opposite-state conflict rejects the complete batch with no track/batch/event rows added;
- a forced exception after track insertion rolls back all rows;
- cross-label overlap is accepted.

- [ ] **Step 4: Run and verify RED**

```powershell
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' -m unittest `
  tests.maest522.test_trusted_import -v
```

Expected: missing `trusted_import` module/API failures.

- [ ] **Step 5: Refactor reusable audio inspection**

Extract file hashing and metadata inspection from `import_source()` into the side-effect-free `inspect_audio_identity()`. Make the legacy candidate importer call the same helper so identity rules cannot drift.

- [ ] **Step 6: Implement preflight and atomic commit**

Preflight parses and inspects outside a write transaction. Commit recomputes playlist byte digest and preflight results, refuses any error/conflict, then inserts tracks, batch, and events inside one `store.connection()` transaction.

Use `playlist_path.read_bytes()` for the playlist digest and the existing chunked SHA-256 implementation for audio identities.

- [ ] **Step 7: Verify GREEN**

Run `test_trusted_import` and existing `test_library_import`/`test_playlists` modules. Require exit 0.

- [ ] **Step 8: Commit**

```powershell
git add tools/maest522/library.py tools/maest522/trusted_import.py `
  tests/maest522/test_trusted_import.py
git commit -m 'feat: import trusted label playlists atomically'
```

---

### Task 5: Expose Strict Single-Label API and Queue Contracts

**Files:**
- Modify: `tools/maest522/queues.py`
- Modify: `tools/maest522/annotation_api.py`
- Modify: `tools/maest522/annotation_db.py`
- Modify: `tests/maest522/test_queues.py`
- Modify: `tests/maest522/test_annotation_api.py`
- Modify: `tests/maest522/test_annotation_workflow.py`

**Interfaces:**
- Changes: `create_round(store, project_id, label, round_number, split="train", student_scores=None, seed=522) -> QueueSummary`.
- Changes: queue payload contains one `active_label` and one current `state`.
- Adds: `GET/PATCH /api/projects/{project_id}/goals`.
- Adds: `GET /api/projects/{project_id}/confirmed-progress`.
- Adds: `POST /api/projects/{project_id}/trusted-playlists/preflight`.
- Adds: `POST /api/projects/{project_id}/trusted-playlists/commit`.
- Adds: `POST /api/projects/{project_id}/confirmed-labels/correct`.
- Changes: round request requires one exact `label`; annotation request requires `label`, `state`, and optional `note`, not a three-label dictionary.

- [ ] **Step 1: Write failing one-label queue tests**

Assert a Deep-Tech round selects only Deep-Tech credits, allows the same track in a later Microhouse round through `(project, track, label)` uniqueness, calculates the 1,000 cap per label, and treats latest uncertain as incomplete for goal progress but complete for that exact queue item so the next candidate is returned.

- [ ] **Step 2: Write failing API contract tests**

Exercise each endpoint with ASGI transport. Assert strict unknown-field rejection, exact label validation, clean preflight without writes, digest-bound commit, progress response, goal update, correction reason, and split-frozen mutation rejection.

Expected annotation request:

```json
{
  "label": "Electronic---Minimal-Deep-Tech",
  "state": "positive",
  "note": ""
}
```

- [ ] **Step 3: Run and verify RED**

```powershell
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' -m unittest `
  tests.maest522.test_queues `
  tests.maest522.test_annotation_api `
  tests.maest522.test_annotation_workflow -v
```

Expected: old all-label queue and request behavior causes assertion failures.

- [ ] **Step 4: Implement label-scoped queue selection**

Filter candidate rows by the requested label, write `queue_items.label` and `queue_rounds.label`, and count prior credits only for that label. `_next_incomplete_queue_item()` must query the queue item's exact label and consider positive, negative, or uncertain an answered item.

- [ ] **Step 5: Implement API models and endpoints**

Add explicit Pydantic models for goals, preflight, commit, correction, round, and single review. Reuse domain functions from Tasks 3 and 4; keep HTTP 400 for invalid input and 409 for state conflicts.

- [ ] **Step 6: Verify GREEN**

Re-run the three focused modules and require exit 0.

- [ ] **Step 7: Commit**

```powershell
git add tools/maest522/queues.py tools/maest522/annotation_api.py `
  tools/maest522/annotation_db.py tests/maest522/test_queues.py `
  tests/maest522/test_annotation_api.py `
  tests/maest522/test_annotation_workflow.py
git commit -m 'feat: expose single label annotation api'
```

---

### Task 6: Replace the UI with the Sequential Active-Label Workflow

**Files:**
- Modify: `tools/maest522/static/index.html`
- Modify: `tools/maest522/static/app.js`
- Modify: `tools/maest522/static/styles.css`
- Modify: `tests/maest522/test_annotation_ui_assets.py`

**Interfaces:**
- Persists: `maest522.activeLabel.<projectId>` in browser localStorage.
- Displays: one active-label selector, positive/negative goals and counts, uncertain count, completion state, trusted batch history, preflight details, and one-label review controls.
- Removes: three simultaneous label rows and number-key label switching.

- [ ] **Step 1: Write failing static UI tests**

Assert assets contain:

```text
Активный жанр
Плейлист +
Плейлист −
Проверить
Импортировать batch
Цель да
Цель нет
не уверен / пропустить
```

Assert the old copy `Для каждого трека укажите состояние всех трёх стилей` and `Digit1/Digit2/Digit3` handlers are absent.

- [ ] **Step 2: Run and verify RED**

```powershell
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' -m unittest `
  tests.maest522.test_annotation_ui_assets -v
```

Expected: missing new controls and old three-label UI still present.

- [ ] **Step 3: Implement the Russian layout**

Keep project selection, audio preview, jump controls, notes, and back navigation. Replace source/round setup with active-label goals and trusted playlist preflight/commit. Put fingerprints/split/export in a visibly separate finalization section disabled until API progress reports all goals met.

- [ ] **Step 4: Implement client state and events**

Use exact API payloads from Task 5. A successful preflight stores the returned digest and enables commit; changing label, state, or path invalidates it. Commit refreshes progress and batch history without changing the active label. `U` saves uncertain/skip for manual queue items.

- [ ] **Step 5: Verify GREEN and HTTP assets**

Run the UI asset test, then start a temporary server on port 8523 with a fresh temporary schema-v2 database and verify:

```powershell
Invoke-WebRequest -Uri 'http://127.0.0.1:8523/' -UseBasicParsing
Invoke-WebRequest -Uri 'http://127.0.0.1:8523/static/app.js' -UseBasicParsing
```

Require HTTP 200 for both.

- [ ] **Step 6: Commit**

```powershell
git add tools/maest522/static/index.html tools/maest522/static/app.js `
  tools/maest522/static/styles.css `
  tests/maest522/test_annotation_ui_assets.py
git commit -m 'feat: add sequential label import ui'
```

---

### Task 7: Export Partial Labels and Enforce Final Split Coverage

**Files:**
- Modify: `tools/maest522/splits.py`
- Modify: `tools/maest522/manifests.py`
- Modify: `tools/maest522/training_data.py`
- Modify: `tests/maest522/test_splits.py`
- Modify: `tests/maest522/test_manifests.py`
- Modify: `tests/maest522/test_training_data.py`

**Interfaces:**
- Produces: final split assignment over tracks with at least one confirmed event.
- Produces: split audit summary with positive/negative counts per label and split.
- Produces: manifest `labels` object containing all three exact keys with values `positive`, `negative`, `uncertain`, or `unreviewed`.
- Produces: manifest `label_mask` object with integer `1/0` values matching supervision.
- Changes: `decode_manifest_row()` accepts `unreviewed` and `uncertain` as target `0`, mask `0`.

- [ ] **Step 1: Write failing split-coverage tests**

Create grouped confirmed events for all labels. Assert one group never crosses splits and export is blocked when any split contains zero positive or zero negative supervision for a label. Assert unconfirmed project tracks are not part of final dataset counts.

- [ ] **Step 2: Write failing manifest tests**

For a track labeled positive Minimal-Deep-Tech only, assert export contains:

```json
"labels": {
  "Electronic---Minimal-Deep-Tech": "positive",
  "Electronic---Microhouse": "unreviewed",
  "Electronic---RoMinimal": "unreviewed"
},
"label_mask": {
  "Electronic---Minimal-Deep-Tech": 1,
  "Electronic---Microhouse": 0,
  "Electronic---RoMinimal": 0
}
```

Assert provenance links the row to trusted batch IDs/states without exposing local M3U or audio paths in portable output.

- [ ] **Step 3: Write failing training decoder tests**

Assert target tensor order `[Minimal-Deep-Tech, Microhouse, RoMinimal]`, positive/negative masks of 1, uncertain/unreviewed masks of 0, and rejection of a missing label key or unknown state.

- [ ] **Step 4: Run and verify RED**

```powershell
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' -m unittest `
  tests.maest522.test_splits `
  tests.maest522.test_manifests `
  tests.maest522.test_training_data -v
```

Expected: current queue-only/full-review export and decoder requirements fail.

- [ ] **Step 5: Implement confirmed-track split and coverage audit**

Change split candidate SQL to the distinct track IDs present in `confirmed_label_events`. Preserve current duplicate/release/artist grouping. Add per-label/per-state split counts to the audit digest input so changed label coverage invalidates the audit SHA-256.

- [ ] **Step 6: Implement ledger-based manifest export**

Read current confirmed events by latest event ID, emit all three label keys and masks, and export only tracks with at least one non-unreviewed current event. Candidate role provenance maps trusted positive batches to `positive_candidate`, trusted negative batches to `hard_negative_candidate`, and manual uncertain items to `unlabeled_pool`.

- [ ] **Step 7: Implement masked training decode**

Require all exact label keys. Map states using:

```python
if state == "positive":
    targets[index] = 1.0
    target_mask[index] = 1.0
elif state == "negative":
    target_mask[index] = 1.0
elif state in {"uncertain", "unreviewed"}:
    pass
else:
    raise ValueError(...)
```

Validate provided `label_mask` matches the state-derived mask.

- [ ] **Step 8: Verify GREEN**

Re-run the three focused modules and require exit 0.

- [ ] **Step 9: Commit**

```powershell
git add tools/maest522/splits.py tools/maest522/manifests.py `
  tools/maest522/training_data.py tests/maest522/test_splits.py `
  tests/maest522/test_manifests.py `
  tests/maest522/test_training_data.py
git commit -m 'feat: export masked partial label manifests'
```

---

### Task 8: Add End-to-End Workflow Coverage and Update Operator Documentation

**Files:**
- Modify: `tests/maest522/test_annotation_workflow.py`
- Create: `tests/maest522/test_sequential_workflow.py`
- Modify: `docs/maest522-annotation-guide.md`
- Modify: `docs/maest522-training-guide.md`
- Modify: `docs/maest522-publication-guide.md`

**Interfaces:**
- Verifies: trusted batches -> editable goals -> three-label completion -> fingerprints -> one split -> masked manifest.
- Documents: external candidate app creates `+`/`-` M3Us; the annotation UI never scans the full library.

- [ ] **Step 1: Write the failing end-to-end test**

Build three labels with small temporary positive/negative batches, including one cross-label track and one uncertain manual review. Assert the active label never changes implicitly, duplicate import is idempotent, goals are independently adjustable, finalization remains blocked until all goals are met, and final manifest/split audit succeeds afterward.

- [ ] **Step 2: Run and verify RED**

```powershell
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' -m unittest `
  tests.maest522.test_sequential_workflow -v
```

Expected: missing integrated workflow behavior.

- [ ] **Step 3: Complete the smallest integration fixes**

Connect only missing API/domain boundaries revealed by the end-to-end test. Do not add background candidate discovery, automatic label switching, or automatic training.

- [ ] **Step 4: Rewrite the Russian guides**

Document this exact operator order:

1. Select Minimal-Deep-Tech.
2. Import trusted `+` and `-` M3Us until current goals are met.
3. Repeat for Microhouse, then RoMinimal.
4. Run fingerprints and freeze one split only after all current goals are met.
5. Export and train only after the final coverage/leakage audit.

Include correction semantics, soft-goal behavior, atomic conflict recovery, and the final exact label indices.

- [ ] **Step 5: Verify GREEN and documentation sentinels**

Run the sequential workflow test and:

```powershell
rg -n 'Electronic---Minimal-Deep-Tech|1000.*1000|атомар|один.*жанр' `
  docs/maest522-annotation-guide.md docs/maest522-training-guide.md `
  docs/maest522-publication-guide.md
```

Require the test to pass and the required Russian workflow text to be present.

- [ ] **Step 6: Commit**

```powershell
git add tests/maest522/test_annotation_workflow.py `
  tests/maest522/test_sequential_workflow.py `
  docs/maest522-annotation-guide.md docs/maest522-training-guide.md `
  docs/maest522-publication-guide.md
git commit -m 'test: cover sequential label collection workflow'
```

---

### Task 9: Full Verification, Live Migration, and Browser Smoke Test

**Files:**
- No committed source files unless verification exposes a defect; any defect follows a new RED/GREEN cycle in the owning task's test file.
- Mutates after backup: `annotation-projects/maest522/annotations.db` only.
- Creates recoverable backup: `annotation-projects/backups/maest522-v1-before-sequential-<timestamp>/`.

**Interfaces:**
- Proves: repository tests/compilation pass, the live v1 DB migrates to v2 with exact 100/100 seed counts, server restarts, and the UI/API expose the new label order and progress.

- [ ] **Step 1: Stop the exact annotation server and resolve targets**

Identify port 8522 ownership, verify both process command lines contain `tools\annotation_ui.py` and the exact live DB path, then stop only those PIDs. Resolve the live data directory and backup destination and verify both remain inside the worktree `annotation-projects` root.

- [ ] **Step 2: Run repository verification before live mutation**

```powershell
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' -m py_compile `
  src\config.py src\environment.py src\pipeline.py src\extractor.py `
  src\tagger.py src\report.py src\main.py src\storage.py `
  tools\annotation_ui.py
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' -m compileall -q tools\maest522
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' src\main.py --help
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' -m pip check
git -c safe.directory='E:/Projects/music-genre-tagger/.worktrees/maest522' diff --check
```

Require every command to exit 0 before migration.

- [ ] **Step 3: Rehearse migration on a disposable copy**

Copy the stopped live annotation directory to a task-specific temporary directory. Run the migration CLI against the copy with source IDs 1 and 2. Verify schema 2, integrity `ok`, zero FK violations, label order, goals `1000/1000`, progress `100/100`, two batches, and 200 events.

- [ ] **Step 4: Create the recoverable live backup**

Move/copy the complete stopped v1 directory, including DB/WAL/upload files, to the timestamped backup directory. Verify file counts and SHA-256 hashes, then restore an identical working copy to the original live path for migration. Do not delete the backup.

- [ ] **Step 5: Migrate the live database explicitly**

Run:

```powershell
& 'E:\Projects\music-genre-tagger\.venv\Scripts\python.exe' `
  -m tools.maest522.migrate_annotation_db `
  --db '.\annotation-projects\maest522\annotations.db' `
  --backup '.\annotation-projects\backups\annotations-v1-pre-migration.sqlite' `
  --project-id 1 `
  --positive-source-id 1 `
  --negative-source-id 2
```

Require the CLI report to show converted positive `100`, converted negative `100`, integrity `ok`, foreign keys `0`, and schema `2`.

- [ ] **Step 6: Restart hidden and verify HTTP/API state**

Start `tools\annotation_ui.py` with `-WindowStyle Hidden`, redirect stdout/stderr to the ignored annotation data directory, and wait for port 8522. Verify HTTP 200 for `/`, `/static/app.js`, and project progress endpoints.

Expected project state:

```text
Electronic---Minimal-Deep-Tech: positive 100/1000, negative 100/1000
Electronic---Microhouse: positive 0/1000, negative 0/1000
Electronic---RoMinimal: positive 0/1000, negative 0/1000
```

- [ ] **Step 7: Perform browser smoke test**

Refresh `http://127.0.0.1:8522/`, open project 1, verify one active-label selector, select each label, verify progress, preflight the existing positive M3U without committing, and confirm the report is idempotent with 100 existing positive labels and no conflicts.

- [ ] **Step 8: Inspect final repository and data state**

```powershell
git -c safe.directory='E:/Projects/music-genre-tagger/.worktrees/maest522' status --short
git -c safe.directory='E:/Projects/music-genre-tagger/.worktrees/maest522' log -10 --oneline
```

Require no uncommitted source changes and no tracked local database/log artifacts. Report the backup path and recovery method to the user.

---

## Plan Self-Review Checklist

- Every approved design requirement is covered by Tasks 1-9.
- The exact label order is consistent in constants, checkpoint/HF tests, UI, manifest tensor order, and live migration expectations.
- The legacy label appears only in migration tests/module and the design/plan history.
- Every production behavior starts with a failing test and an observed RED run.
- The live database is touched only after full tests and a successful disposable-copy rehearsal.
- No step scans or modifies `M:\Volumes` audio.
- No step uploads to Hugging Face, trains a model, commits a database, or changes the original 519 labels.
