# Sequential Single-Label Annotation Design

**Date:** 2026-08-12
**Status:** Approved in conversation
**Scope:** MAEST 522 local annotation database, API, UI, manifest export, and the
safe migration of the current project database

## Context

The initial annotation workflow creates quota rounds for all three extension
labels at once and requires a complete three-label review for every queue item.
The operator instead selects tracks in an external listening application and
produces trusted positive and negative M3U playlists. They need to finish one
label without switching listening context every 100 tracks:

1. `Electronic---Minimal-Deep-Tech`
2. `Electronic---Microhouse`
3. `Electronic---RoMinimal`

The current live annotation project contains 200 tracks, no queue rounds, and
no annotation events. Source 1 is a trusted 100-track positive Minimal-Deep-Tech
playlist and source 2 is a trusted 100-track negative Minimal-Deep-Tech
playlist.

## Public Label Contract

The three appended labels, in fixed classifier-row order, are:

1. `Electronic---Minimal-Deep-Tech` at index 519
2. `Electronic---Microhouse` at index 520
3. `Electronic---RoMinimal` at index 521

This exact spelling is global across the native checkpoint, Hugging Face
configuration, UI, SQLite v2 ledger, manifests, training, evaluation, model and
dataset cards, and release reports. The original 519 label strings and row
order remain unchanged.

## Goals

- Keep one project and one final leakage-safe split for all three labels.
- Let the operator select one active label and stay on it until they choose to
  switch.
- Treat trusted positive and negative M3U files as completed human annotation,
  without requiring a second listening pass.
- Default to soft goals of 1,000 positive and 1,000 negative labels per genre.
- Preserve missing and uncertain labels as masked supervision, never implicit
  negatives.
- Keep every import and correction auditable.
- Reject conflicting imports atomically.
- Preserve and migrate the existing 100 positive and 100 negative DeepTech
  labels after an explicit backup.
- Defer fingerprints, the final split, training, and test evaluation until all
  three labels reach their current goals.

## Non-goals

- The MAEST 519-to-522 model architecture and continual-training stages do not
  change in this feature.
- The annotation application will not select candidates from the full music
  library. Candidate discovery remains in the operator's external application.
- A trusted playlist import will not be represented as an artificial active-
  learning queue or listening event.
- Lowering a goal will not delete labels or audio records.
- This feature will not infer an unreviewed label as negative.

## Chosen Architecture

Use a confirmed-label ledger in the existing annotation database. Candidate
sources, acquisition queues, and listening reviews remain distinct from
trusted completed labels.

### Schema version 2

Add `label_goals`:

| Column | Contract |
| --- | --- |
| `project_id` | Existing project foreign key |
| `label` | One exact value from `NEW_LABELS` |
| `positive_target` | Positive integer, default `1000` |
| `negative_target` | Positive integer, default `1000` |
| `updated_at` | UTC ISO-8601 timestamp |

The primary key is `(project_id, label)`. Creating a project inserts exactly
three goal rows.

Add `confirmed_label_batches`:

| Column | Contract |
| --- | --- |
| `id` | Integer primary key |
| `project_id` | Existing project foreign key |
| `label` | One exact extension label |
| `state` | `positive` or `negative` |
| `source_kind` | `m3u` or `m3u8` |
| `source_path` | Local source path used for the import |
| `playlist_sha256` | Digest of the exact playlist bytes |
| `discovered_count` | Parsed non-comment entries |
| `new_count` | New current labels created by the batch |
| `existing_count` | Idempotent labels already present |
| `created_at` | UTC ISO-8601 timestamp |

Add append-only `confirmed_label_events`:

| Column | Contract |
| --- | --- |
| `id` | Integer primary key and event ordering source |
| `project_id` | Existing project foreign key |
| `track_id` | Existing project track foreign key |
| `label` | One exact extension label |
| `state` | `positive`, `negative`, or `uncertain` |
| `event_kind` | `trusted_import`, `manual_review`, or `correction` |
| `batch_id` | Nullable trusted batch foreign key |
| `note` | Required non-empty reason for `correction`, otherwise optional |
| `created_at` | UTC ISO-8601 timestamp |

The current state for `(project_id, track_id, label)` is the event with the
largest `id`. There is no destructive update or delete operation for events.
Indexes cover current-state lookup, progress by label/state, and batch audit.

### Schema migration

Schema v1 is never migrated silently during server startup. A dedicated
backup-first migration command performs these steps in one exclusive
transaction:

1. Validate v1, `PRAGMA integrity_check`, and foreign keys.
2. Require and verify a byte-for-byte backup outside the live database path.
3. Create v2 tables and goal rows, then set `schema_meta.version = 2`.
4. Optionally convert explicitly selected trusted v1 source IDs into confirmed
   events.
5. Commit, checkpoint WAL, reopen, and repeat integrity and foreign-key checks.

For the current database, the explicit conversion mapping is:

- source 1 -> `Electronic---Minimal-Deep-Tech = positive`
- source 2 -> `Electronic---Minimal-Deep-Tech = negative`

The migration must verify that both sources belong to project 1, each links to
exactly 100 unique tracks, their track sets do not overlap, all files exist,
and their persisted v1 suggested label is the legacy
`Electronic---DeepTech-Minimal` value before writing any event. This explicit
v1 migration boundary maps that stored value to
`Electronic---Minimal-Deep-Tech`; the legacy value is not accepted by the v2
API, manifests, model metadata, training pipeline, or publication artifacts.

## Trusted Playlist Import

Trusted M3U import has a read-only preflight and a separate atomic commit.

### Preflight

The operator supplies project, active label, desired state, and either a local
M3U path or uploaded M3U text with a base directory. Preflight:

1. Parses UTF-8/UTF-8-BOM M3U or M3U8 entries.
2. Rejects remote URLs and unsupported extensions.
3. Resolves paths and verifies every file exists.
4. Rejects duplicate paths inside the playlist.
5. Calculates each exact file SHA-256 using the existing track identity rules.
6. Deduplicates physical copies through the project track SHA-256.
7. Compares each track with the current confirmed state for the active label.
8. Returns discovered, new, idempotent, missing, duplicate, and conflict
   details plus the playlist SHA-256.

Preflight writes nothing.

### Commit

Commit receives the same import request and the expected playlist SHA-256. It
revalidates the playlist digest and all conflict conditions before opening the
write transaction. Within one transaction it inserts any new tracks, one batch
row, and one `trusted_import` event per new current label.

- Same track, same label, same state: idempotent; count as existing.
- Same track, same label, opposite state: reject the complete batch.
- Same track, different labels: allowed; this is a multilabel dataset.
- Any missing file, changed digest, duplicate, or validation error: reject the
  complete batch.
- Reaching or exceeding a goal does not block import. Goals are planning
  targets, not data caps.

### Corrections

An existing opposite state can change only through an explicit correction
operation. A correction requires the track, label, replacement state, and a
non-empty reason, then appends a `correction` event. It never edits the source
batch or prior events.

## User Interface

The project screen has one `Active label` selector persisted in browser
localStorage per project. Every API request still carries the exact label, so
the server has no implicit UI state. The selector never changes automatically
after a batch or every 100 tracks.

For the active label, display:

- positive current count and editable positive target;
- negative current count and editable negative target;
- uncertain count;
- completion state computed from the two current targets;
- recent trusted batches and their audit counts.

The trusted import panel contains:

- local M3U/M3U8 path or browser upload;
- `Positive playlist (+)` / `Negative playlist (-)` choice;
- `Preflight` button;
- a preflight result with new, existing, missing, duplicate, and conflicting
  paths;
- `Import batch` button enabled only for the unchanged clean preflight.

The optional listening panel shows only the active label and the controls
`yes`, `no`, and `uncertain / skip`. Saving appends one `manual_review` event.
An uncertain event advances to the next item, does not count toward positive
or negative goals, and is masked during training.

The old three-label-at-once wording and validation are removed. Existing
candidate/acquisition functionality remains available as an advanced path but
round creation accepts one exact label and progress is calculated for that
label only.

## Progress and Completion

Current progress is calculated from the latest confirmed event per
`(track, label)`:

- positive progress counts only latest `positive` states;
- negative progress counts only latest `negative` states;
- uncertain is reported separately;
- idempotent reimports never increment progress.

Targets are soft and independently editable per label. Reducing a target below
the current count deletes nothing and marks that side complete. Values above
the target are displayed directly, for example `1050 / 1000`.

The final-data controls are enabled only when every label meets both current
targets. A conflicting import never changes current data and therefore does
not create a persistent unresolved-conflict state. The operator still makes
the explicit finalization action; reaching a goal does not start fingerprints,
split creation, export, or training automatically.

## Final Split and Manifest

After all labels are ready:

1. Fingerprint the union of confirmed tracks.
2. Build duplicate/release/artist groups using the existing group rules.
3. Freeze one project-wide 70/15/15 group split.
4. Run the existing leakage audit.
5. Report per-label positive and negative counts in each split and block export
   if a split has no supervised examples for either side of any label.
6. Export one three-label state vector and supervision mask per track.

For each label:

- latest `positive` -> target `1`, mask `1`;
- latest `negative` -> target `0`, mask `1`;
- latest `uncertain` or no event -> placeholder target `0`, mask `0`.

Training data validation must accept masked missing labels while still
rejecting unknown state names, missing label keys in the exported contract,
digest drift, or split leakage. No model training or test evaluation runs
before the three-label dataset passes this final audit.

## API Boundaries

Add project-scoped endpoints for:

- reading and updating label goals;
- trusted-playlist preflight;
- trusted-playlist atomic commit;
- current per-label progress;
- append-only correction;
- single-label queue creation and single-label manual review.

Requests continue to reject unknown fields and unknown labels. Mutating
endpoints validate that the project is not split-frozen. Once the split is
frozen, goals, trusted imports, and corrections are immutable for that project;
a changed dataset requires a new project and a new split audit.

## Error Handling and Recovery

- All import errors identify the playlist and affected path without modifying
  source audio.
- SQLite writes use the existing foreign-key, WAL, busy-timeout, and transaction
  settings.
- A trusted batch never partially commits.
- Startup reports an actionable v1 migration command rather than changing the
  database implicitly.
- The live migration is performed only after stopping the annotation server,
  creating and verifying the backup, and confirming exact source IDs.
- A failed migration restores service from the untouched backup; no source
  audio is changed.

## Verification

Automated tests must cover:

- fresh schema v2 creation and v1 rejection at ordinary startup;
- backup-first v1-to-v2 migration on a disposable copy;
- exact conversion of the two 100-track DeepTech sources;
- clean preflight with no writes;
- atomic positive and negative commits;
- idempotent same-state reimport;
- complete rollback for opposite-state conflict, missing file, duplicate, and
  changed playlist digest;
- cross-label overlap acceptance;
- append-only correction with a required reason;
- editable soft targets and progress above/below target;
- active-label-only queue payload, manual review, and progress;
- uncertain skip behavior and replacement candidate selection;
- partial three-label manifest masks;
- group split leakage audit and per-label split coverage;
- unchanged model/training/release tests;
- Russian UI assets and a local HTTP/browser smoke test.

Before touching the live database, run the focused migration and import tests.
After implementation, run the complete repository test suite, Python compile
checks, dependency checks, and an HTTP smoke test. Then stop the server, back up
the live annotation directory, migrate, verify counts and integrity, restart,
and verify that the UI reports DeepTech progress as `100 positive / 100
negative` with zero Microhouse and RoMinimal labels.

## Acceptance Criteria

- The operator can remain on one label for an arbitrary number of batches.
- A trusted positive or negative M3U becomes completed human supervision
  without a duplicate listening pass.
- Default goals are `1000/1000` and remain editable without deleting data.
- Conflicting batches are atomic and report exact paths.
- The current project safely begins at DeepTech `100/100` after migration.
- Missing or uncertain labels are masked, never negative.
- All three labels share one final audited split.
- Training remains blocked until the complete dataset audit passes.
