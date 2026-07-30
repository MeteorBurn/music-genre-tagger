# Three-window MAEST analysis design

## Goal

Release a new, intentionally incompatible 2.0 version of Music Genre Tagger
that uses `maest-infer==0.2.0` and derives each track's final top-three genres
from the mean sigmoid scores of up to three representative 30-second windows.

Backward compatibility with databases, JSON snapshots, or analysis metadata
created by earlier versions is explicitly out of scope.

## Dependency policy

`requirements.txt` must require exactly:

```text
maest-infer==0.2.0
```

The exact pin makes the model API and inference behavior reproducible. The
existing environment checks must reject an older installation and install the
required version through the project's established dependency workflow.

## Window selection

The system decodes each track once to mono 16 kHz audio. Window placement is
computed from the decoded sample count rather than container metadata so that
the slices match the audio actually passed to MAEST.

The target window duration is 30 seconds. Candidate window centers are located
at 20%, 50%, and 80% of the decoded duration. Each center is converted to a
start position and clamped to the valid interval:

```text
start = clamp(center - 15 seconds, 0, duration - 30 seconds)
```

Start positions are represented as sample indices while slicing. Duplicate
starts caused by clamping are removed while preserving chronological order.

Expected examples:

| Track duration | Window starts |
| ---: | --- |
| 20 seconds | 0 seconds |
| 40 seconds | 0, 5, 10 seconds |
| 60 seconds | 0, 15, 30 seconds |
| 180 seconds | 21, 75, 129 seconds |
| 300 seconds | 45, 135, 225 seconds |

For a track of 30 seconds or less, the complete track is analyzed once. No
silence padding is added. Existing rejection of empty or extremely short audio
remains in effect.

## Inference and aggregation

Each unique window is passed to MAEST independently. For every window,
`predict_labels()` returns a sigmoid score for every model label.

Before aggregation, the implementation must verify that every successful
window returns:

- the same ordered label vocabulary;
- a one-dimensional score vector;
- the same number of scores and labels.

The final per-label score is the arithmetic mean across windows:

```text
track_scores[label] = mean(window_scores[*][label])
```

The final top three labels are selected only after calculating the complete
mean vector. Labels retain the existing cleanup rule:

```python
label.split("---", 1)[-1]
```

Stored confidence values are the rounded aggregated scores, preserving the
current four-decimal representation.

If inference fails for any selected window, analysis of that track fails and
records the error through the existing per-file error path. Partial window
results must not be silently treated as a complete track result.

## Configuration

The obsolete single-window `AUDIO_OFFSET` setting is removed. The analysis
configuration exposes constants for:

```python
AUDIO_WINDOW_DURATION = 30
AUDIO_WINDOW_POSITIONS = (0.2, 0.5, 0.8)
NUM_GENRES = 3
```

Window positions are fractions of decoded track duration. They are internal
configuration values in version 2.0; no new CLI flags are required.

## Storage and exported data

Version 2.0 uses a new database schema. No automatic migration is provided.
When an existing `tracks.db` has the old schema, startup must stop with an
actionable error explaining that the database belongs to an incompatible
version and must be moved or deleted before analysis.

The old scalar `audio_segment_offset` field is removed. Analysis metadata is:

```json
{
  "audio_segment_offsets": [21.0, 75.0, 129.0],
  "audio_segment_duration": 30.0,
  "audio_segment_count": 3,
  "aggregation": "mean"
}
```

`audio_segment_offsets` stores actual starts derived from sample indices and is
serialized as a JSON array in SQLite. `audio_segment_duration` stores the
configured target duration; a short track can have a shorter actual window.
`audio_segment_count` stores the number of unique analyzed windows.

The same analysis metadata appears in optional `tracks.json` exports. Excel
keeps its existing user-facing columns because window metadata is diagnostic
and does not affect tagging.

Incremental identity remains path-based in this change. Redesigning track
identity or invalidating existing results by content is out of scope because
version 2.0 starts with a fresh database.

## Tests

The project gains a focused test suite covering:

- exact starts for 20, 30, 40, 60, 180, and 300-second decoded audio;
- uniqueness and chronological ordering of clamped starts;
- slicing by sample index at 16 kHz;
- one-window behavior for tracks no longer than 30 seconds;
- arithmetic mean aggregation across three complete score vectors;
- label-vocabulary and score-shape mismatch failures;
- top-three selection after aggregation rather than per-window voting;
- serialized version 2.0 analysis metadata;
- rejection of the legacy database schema.

Tests must not download model weights or run a real MAEST forward pass. A
separate manual smoke test with one local audio track and the real checkpoint
is required when the model and environment are available.

## Documentation

`README.md` and `AGENTS.md` must describe:

- MAEST 0.2.0 as the required inference package;
- three-window placement and mean aggregation;
- behavior for tracks no longer than 30 seconds;
- the new database and JSON analysis metadata;
- the intentional incompatibility of version 2.0 databases;
- removal of `AUDIO_OFFSET`.

## Acceptance criteria

The change is accepted when:

1. dependency resolution requires `maest-infer==0.2.0`;
2. deterministic tests prove the documented window starts;
3. final top-three genres come from the mean of complete per-label score
   vectors;
4. short tracks are analyzed once without padding;
5. new SQLite and JSON records contain the documented analysis metadata;
6. legacy databases fail with an actionable incompatibility message;
7. focused tests and the project's Python compilation check pass;
8. a real one-track smoke test is either successful or explicitly reported as
   unavailable with the reason.
