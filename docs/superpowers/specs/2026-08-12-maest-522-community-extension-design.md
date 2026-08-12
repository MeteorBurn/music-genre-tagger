# MAEST 522-label community extension design

Date: 2026-08-12

Status: superseded for annotation workflow by
`2026-08-12-sequential-single-label-annotation-design.md`; the model/training
architecture remains applicable.

## 1. Objective

Build a public community extension of
`discogs-maest-30s-pw-129e-519l` that preserves the original 519 output
classes and appends three independent multilabel outputs:

| Index | Label |
| ---: | --- |
| 519 | `Electronic---Minimal-Deep-Tech` |
| 520 | `Electronic---Microhouse` |
| 521 | `Electronic---RoMinimal` |

The result must remain a single fine-tuned MAEST model. It must not be a
separate classifier trained on frozen MAEST embeddings.

The chosen method is hybrid continual fine-tuning: supervised learning for
the three new outputs, replay distillation against a frozen MAEST-519 teacher,
and staged unfreezing of the final transformer blocks.

## 2. Available data and compute

- Approximately 45,000 local tracks are available.
- Approximately 75% of the library is in or near the target style domain.
- The initial candidate pool comes from target-style folders and user-provided
  M3U/M3U8 playlists.
- Annotation is organized as up to 10 rounds of 100 candidates per target
  label, for a maximum review budget of 1,000 queued candidates per label. A
  track may be positive for more than one new label, and overlapping candidates
  are reviewed only once.
- The remaining library is available as an unlabeled replay pool and as a
  source of hard negatives.
- Training hardware is an NVIDIA GeForce RTX 3090 with 24 GB VRAM.

The original Discogs20 audio and its 519-label ground truth are not public.
Consequently, preservation of the original outputs can be evaluated as
behavioral regression against the original teacher, but not as an independent
ground-truth accuracy measurement for all 519 classes.

## 3. Scope

The project includes:

1. A local annotation application with folder and M3U/M3U8 ingestion.
2. Dataset identity, deduplication, annotation, and leakage-safe split tools.
3. Explicit 519-to-522 checkpoint expansion.
4. Hybrid continual fine-tuning with replay distillation.
5. Evaluation of the new labels and regression of the original outputs.
6. Native MAEST and Hugging Face publication artifacts.
7. Reproducible manifests, reports, conversion scripts, and model card.

The project does not include:

- redistribution of copyrighted source audio;
- claims that the extension is an official MAEST, MTG, or Discogs release;
- renaming or reordering the original 519 labels;
- automatic treatment of folder names, playlists, or missing labels as
  ground truth;
- full-backbone fine-tuning by default;
- publication under a guessed permissive license.

## 4. Canonical model contract

The public model name will follow this form:

`discogs-maest-30s-pw-129e-522l-community-v1`

The exact version suffix may advance for later releases, but the base model
name must include `community` and must not replace the original 519-label
architecture name.

The first 519 labels, indices, and output semantics remain unchanged. The
three new labels occupy indices 519 through 521 in the order shown in section
1.

The architecture remains:

- 16 kHz mono input;
- 30-second input windows;
- 96 mel bands;
- 16 by 16 patches with 10 by 10 stride;
- 12 transformer blocks;
- hidden size 768;
- CLS and DIST tokens;
- mean of CLS and DIST representations for the normal inference path;
- independent sigmoid outputs for multilabel classification.

Teacher and student must receive identical audio crops and identical mel
preprocessing during distillation.

## 5. Explicit checkpoint expansion

Checkpoint expansion must use explicit tensor surgery. Loading a 519-output
checkpoint into a 522-output architecture with `strict=False` is not
sufficient because matching parameter names with incompatible tensor shapes
still fail to load.

The following tensors are expanded:

```text
head.1.weight       [519, 768] -> [522, 768]
head.1.bias         [519]      -> [522]
head_dist.weight    [519, 768] -> [522, 768]
head_dist.bias      [519]      -> [522]
```

The first 519 rows are copied exactly. The backbone, positional embeddings,
CLS/DIST tokens, final normalization, and all other compatible tensors are
copied without modification.

The new rows use small random initialization and biases corresponding to a
low initial positive probability. Neighbor-label initialization may be tested
only as a recorded ablation; it is not the default.

The standard MAEST `distilled_type="mean"` path uses the main normalized head
after averaging CLS and DIST. The legacy 519 rows of `head_dist` are preserved
for native compatibility. Its three extension rows receive a small auxiliary
supervised loss on the DIST representation, but the standard public prediction
path remains the main head.

Before any training, a no-op expansion test must demonstrate that the first
519 student logits match the teacher logits for the same preprocessed inputs
within floating-point tolerance.

## 6. Annotation taxonomy

Folder membership and playlist membership are candidate provenance only.
Ground truth is created by listening.

Each reviewed track receives one independent state for each new label:

- `positive`;
- `negative`;
- `uncertain`;
- `unreviewed`.

`uncertain` and `unreviewed` values are masked out of supervised BCE. They are
never converted implicitly into negative targets.

Before bulk annotation, a versioned `taxonomy.md` defines:

- inclusion and exclusion criteria for each label;
- allowed label overlaps;
- reference tracks;
- neighboring genres;
- boundary examples and the reason for each decision.

Approximately 10% of reviewed examples are presented again after a delay,
without the previous answer, to measure single-annotator self-consistency.

## 7. Candidate and active-learning rounds

Annotation proceeds in rounds:

1. Build the first balanced seed round with up to 100 candidates from each
   label-specific source queue plus explicitly selected neighboring negatives.
   Deduplicate overlaps before review.
2. Freeze group-level train, validation, and test assignments for the full
   eligible candidate universe using the procedure in section 8.
3. Train a head-only seed model using only the train portion of the seed.
4. Rank the remaining eligible training pool using uncertainty, disagreement
   between the three windows, hard-negative score, and diversity.
5. Review the next training batch without displaying the model score or predicted
   label.
6. In parallel, fill the fixed validation and test annotation queues using
   sampling rules frozen before the seed model is trained; student predictions
   never rank these queues.
7. Close the round after up to 100 newly queued candidates per target label,
   snapshot its manifests, train, and evaluate.
8. Repeat for at most 10 rounds, stopping earlier when validation quality
   saturates and the hard-negative and data-coverage requirements are met.

Every selected track is reviewed for all three labels. A positive for one
label may therefore be a positive, negative, or uncertain example for either
of the other labels. Round quotas count source-queue candidates, not guaranteed
positives. Reports therefore show both queued-candidate counts and confirmed
positive counts for each label.

Hard-negative strata include at least `Minimal`, `Minimal Techno`, `Tech
House`, `Deep House`, `Deep Techno`, `Dub Techno`, `House`, and `Techno` where
such examples are present in the library. These strata are manually reviewed;
their metadata or teacher prediction alone does not establish a negative.

## 8. Track identity and leakage prevention

Filesystem paths are not identities. Before splitting, tracks are assigned to
recording groups using the following evidence in priority order:

1. exact audio/content hash;
2. acoustic fingerprint for format copies, edits, and remasters;
3. Discogs master/release or album identity when available;
4. artist identity.

Exact copies and strongly matching versions are kept in one group. Every
30-second window of a track remains in the track's group. No group may appear
in more than one split.

After the initial seed is reviewed and before the first model-guided round, the
full eligible candidate universe is assigned to splits by group. The
assignment uses the seed labels, playlist/folder provenance, fixed teacher
output patterns, and artist identity for coarse stratification. The assignment
is then frozen; later annotations inherit their group's existing split.

The target group allocation is:

- train: 70%;
- validation: 15%;
- locked test: 15%.

Artist identities should also be disjoint where the data permits. The split
generator records unavoidable exceptions explicitly rather than silently
moving individual tracks.

Validation and test receive separate blind annotation queues. Their ordering
is fixed before student training and may use only the frozen provenance and
teacher strata, never a trained student's output. This allows enough held-out
positives to be reviewed without leaking model-guided selection into the
holdout.

The locked test is created before active-learning model selection. It is not
used for candidate ranking, early stopping, threshold selection, or loss
computation. Its completed annotation export and model results remain sealed
until the final release evaluation.

Each new label should have at least 100 distinct positive recording groups in
the locked test for release-grade reporting. If this is not possible, the
result is reported as exploratory and additional annotation is required before
a quality claim is made.

## 9. Replay corpus

Unlabeled tracks are divided, using the same group boundaries, into:

- `replay_train`, used for teacher distillation;
- `regression_holdout`, used only for teacher/student behavioral comparison.

Teacher logits are precomputed for deterministic 30-second windows centered
at 20%, 50%, and 80% of each track. Starts are clamped and duplicate windows
are removed. A student replay sample always uses the exact crop associated
with its stored teacher logits.

Replay sampling is stratified using teacher output patterns and broad parent
categories to reduce domination by the target electronic domain. This cannot
create coverage for old classes absent from the library, and the limitation
must be stated in the model card.

## 10. Local annotation application

The annotation application uses a local FastAPI backend and a small vanilla
HTML/CSS/JavaScript frontend. It binds only to `127.0.0.1`. No source audio is
uploaded to an external service.

The application displays:

- artist, title, release, path, and duration;
- a full-track audio player;
- jumps to 20%, 50%, and 80% of the track;
- independent positive, negative, and uncertain controls for all three
  labels;
- progress for the current 100-candidate tranche, round number from 1 to 10,
  and separate queued/reviewed/positive counts;
- previous, next, save, and explicit skip actions.

The model prediction, uncertainty value, queue reason, and split name are
hidden from the annotator before the decision is saved.

The default keyboard controls are:

- `Space`: play or pause;
- `1`, `2`, `3`: select a label;
- `P`, `N`, `U`: set positive, negative, or uncertain;
- `A`, `S`, `D`: jump to 20%, 50%, or 80%;
- arrow keys: seek;
- `Enter`: atomically save all three states and open the next track;
- `Backspace`: return to the previous track.

A track cannot be marked reviewed until all three states are set. It may be
skipped explicitly as broken/unplayable, wrong file, duplicate, or
insufficient evidence.

Browser-compatible formats are streamed without changing the original file.
Unsupported formats are transcoded by FFmpeg to a preview cache addressed by
content hash. The cache has a bounded LRU size. Decode failure creates a
playback error or skip record, never a negative genre annotation.

## 11. M3U and M3U8 ingestion

The application accepts folders, one or more playlists, or a playlist-only
initial project. A playlist defines seed candidates and their order; it does
not define genre labels.

Supported playlist features are:

- `.m3u` and `.m3u8`;
- absolute Windows, UNC, and OS-native paths;
- relative paths resolved from the playlist directory;
- `file://` entries;
- `#EXTM3U` and `#EXTINF` metadata;
- UTF-8, UTF-8 with BOM, and an explicit fallback encoding for older Windows
  playlists.

A local server-side playlist path is the canonical import because it preserves
the base directory required for relative entries. Browser-uploaded playlist
content requires the user to provide a base directory when relative entries
are present.

Import preserves playlist order and reports total entries, resolved files,
missing files, duplicates, unsupported formats, comments, and external URLs.
HTTP and streaming URLs are not downloaded. A referenced track outside the
selected library root may be added as an explicit `external_seed`; its parent
directory is not scanned recursively.

The manifest records playlist ID, name, position, original entry, resolved
path, and resolution status.

## 12. Annotation storage

SQLite is the annotation source of truth. The logical schema contains:

- tracks;
- recording groups;
- append-only annotation events;
- current annotation state;
- split assignments;
- annotation rounds;
- queue items;
- playlist provenance;
- playback and decode errors.

Saving all three label decisions and advancing the queue occurs in one
transaction. Corrections append a new event rather than deleting history.

Reproducible exports include:

- `annotations.jsonl`;
- `splits.jsonl`;
- label matrices and ground-truth files consumed by training;
- aggregate annotation statistics;
- duplicate and leakage audit reports.

## 13. Training batches and losses

An effective training batch contains approximately equal amounts of:

- manually labeled positives and hard negatives;
- replay samples with teacher logits.

The labeled sampler balances the three new labels and hard-negative strata.
The replay sampler balances available teacher prediction patterns. Multiple
windows from the same recording group are not used to fake independent
examples.

During training, the classifier is represented as a frozen 519-output legacy
module plus a trainable three-output extension module. Their logits are
concatenated. This prevents AdamW weight decay from changing legacy rows even
when their gradients are masked. The two modules are merged into one standard
522-output linear layer for export.

The objective is:

```text
L = lambda_new * L_new
  + lambda_KD  * L_old
  + lambda_SP  * L_parameters
  + lambda_aux * L_dist
```

Where:

- `L_new` is masked BCE with logits for reviewed states of the three new
  labels;
- `L_old` is Smooth L1 between teacher and student logits for indices 0 to
  518;
- `L_parameters` is L2-SP regularization toward the original values of the
  trainable transformer parameters;
- `L_dist` is a low-weight auxiliary BCE for the three new `head_dist` rows.

Hard teacher pseudo-labels are not used. Preserving soft logits retains more
of the teacher's ranking and calibration information.

Mixup is disabled for the initial head warm-up. It may be enabled later with a
small alpha; audio, manual targets, masks, and teacher logits must be mixed
consistently. Moderate SpecAugment and structured patchout remain available as
training regularizers.

## 14. Staged fine-tuning

### Stage 0: baseline and expansion validation

- Expand the checkpoint.
- Prove exact copying of the first 519 rows.
- Measure teacher/student no-op logit parity.
- Freeze the final split and regression holdout manifests.

### Stage 1: extension-head warm-up

- Freeze the full backbone, final normalizations, and legacy head.
- Train only the three new main-head rows and three auxiliary DIST-head rows.
- Select a working head-only baseline using validation data.

### Stage 2: hybrid continual fine-tuning

- Unfreeze transformer blocks 10 and 11 and the final backbone normalization.
- Keep the original 519 classifier rows frozen.
- Train with supervised new-label loss, replay distillation, and L2-SP.
- Use a backbone learning rate 10 to 30 times smaller than the extension-head
  learning rate.
- Stop according to both new-label quality and old-label regression.

### Stage 3: conditional wider unfreezing

Blocks 8 and 9 may also be unfrozen only when:

- new-label validation quality has reached a reproducible plateau;
- the error is not explained by annotation inconsistency or missing hard
  negatives;
- the old-output regression gates remain satisfied.

Full-backbone unfreezing is outside the default design and requires a separate
recorded experiment.

## 15. Initial runtime configuration

The initial RTX 3090 configuration is:

- BF16 autocast when supported stably by the selected runtime, otherwise
  FP16;
- physical batch size selected by a short memory benchmark, expected in the
  range 8 to 16 windows;
- gradient accumulation to an effective batch of 32 to 64;
- AdamW;
- gradient clipping;
- extension-head learning rate near `1e-4`;
- unfrozen-backbone learning rate in the range `3e-6` to `1e-5`;
- small weight decay;
- deterministic seeds and recorded software versions.

SWA is not automatic. The best validation checkpoint may be compared with an
average of the final stable checkpoints. The candidate with the best new-label
quality that still passes every regression gate is selected.

## 16. New-label evaluation

Evaluation is performed at track level. Sigmoid probabilities from the 20%,
50%, and 80% windows are mean-aggregated before metrics are calculated.

Per-label metrics are:

- Average Precision / PR-AUC as the primary metric;
- precision, recall, and F1 at a label-specific threshold;
- recall at fixed precision targets such as 80% and 90%;
- Brier score and calibration curve;
- false-positive rate on manually confirmed hard negatives;
- consistency and disagreement across windows.

Macro and micro summaries are reported, but per-label results remain primary.
Confidence intervals use bootstrap resampling by recording group, never by
window.

Each label threshold is chosen using validation data only. The locked test is
opened once after the checkpoint, thresholds, and release gates are frozen.

## 17. Original-output regression evaluation

On `regression_holdout`, compare teacher and student for the first 519 outputs
using:

- mean and percentile absolute logit/probability drift;
- Spearman rank correlation;
- top-5 and top-10 overlap;
- retention of teacher top-3 labels in the student top-5;
- threshold crossing counts;
- cosine similarity of CLS, DIST, and mean embeddings;
- per-class drift for labels frequently activated by the holdout.

Default release gates are:

- mean probability drift no greater than `0.01`;
- mean top-10 overlap, defined as intersection size divided by 10, at least
  `90%`;
- teacher top-3 to student top-5 retention at least `98%`;
- mean embedding cosine similarity at least `0.98`;
- for every frequently activated old label, mean absolute probability drift
  no greater than `0.03` and 95th-percentile absolute drift no greater than
  `0.10`.

A frequently activated old label is one whose teacher probability is at least
`0.10` on 50 or more distinct regression-holdout recording groups. Any revision
to these numeric gates must be justified and frozen using validation data
before locked-test evaluation. Test results may not be used to relax a gate.

## 18. Required ablations

The evaluation report compares:

1. the original MAEST-519 teacher;
2. the frozen-backbone three-row extension;
3. last-two-block fine-tuning without distillation, as a diagnostic only;
4. hybrid fine-tuning with distillation;
5. hybrid fine-tuning with distillation and hard negatives.

These experiments establish whether backbone adaptation improves the new
labels, whether distillation prevents forgetting, and whether hard negatives
improve the intended fine-grained boundaries.

## 19. Native MAEST compatibility

The native implementation adds a distinct architecture name such as:

`discogs-maest-30s-pw-129e-522l-community`

It must:

- construct exactly 522 outputs;
- expose the original ordered labels plus the three extensions;
- load the released checkpoint strictly;
- retain the public `get_maest()` and `predict_labels()` behavior;
- never silently discard an incompatible classifier head.

The native release should first live in a versioned compatible community
package or branch. An upstream pull request to `maest-infer` may follow, but
the public model must not depend on an unmerged upstream change.

## 20. Hugging Face artifact

The Hugging Face repository publishes:

- `model.safetensors` as the primary safe weight format;
- a native compatible state dict/checkpoint and its SHA-256;
- `config.json` with 522-entry `id2label` and `label2id` mappings;
- `problem_type: "multi_label_classification"`;
- `preprocessor_config.json` matching MAEST preprocessing;
- `thresholds.json`;
- training configuration, seeds, and software versions;
- aggregate dataset and annotation statistics;
- evaluation and ablation reports;
- conversion and parity scripts;
- one-window and full-track inference examples.

The model card sets the original public MAEST model as `base_model` where Hub
metadata supports it and clearly labels this release as a community
fine-tune.

## 21. Artifact parity

Parity is verified at two levels:

1. Given the same precomputed mel tensor, native MAEST and Hugging Face logits
   must match within a strict floating-point tolerance.
2. Given the same waveform, compare end-to-end probabilities and top-K output
   after each implementation's feature extractor. Any preprocessing difference
   must be quantified and documented.

Smoke tests load and infer with both the native API and
`AutoModelForAudioClassification`. The public examples must execute against
the exact uploaded revision.

## 22. Publication and licensing

The model card must disclose:

- community, non-official status;
- the original MAEST repository, paper, and base checkpoint;
- use of a private 45,000-track library;
- manual annotation scale and single-annotator limitation;
- subjective and overlapping genre boundaries;
- strong electronic-music domain bias;
- behavioral, rather than ground-truth, validation of the old 519 outputs;
- known coverage gaps and intended use;
- that source audio is not redistributed.

The upstream repository's AGPL-3.0 code license does not by itself establish a
permissive license for the published model weights. The official Hugging Face
model card currently does not state a weight license. Before public release,
the maintainer must seek clarification from the original authors. Until a
valid license is established, the derivative weights must not be presented as
MIT or Apache licensed; Hub metadata must use an accurate `unknown` or `other`
status with provenance and restrictions explained.

## 23. Release decision

A model is releasable only when all four conditions pass:

```text
new-label quality gate
AND hard-negative gate
AND old-519 behavioral regression gate
AND native/Hugging-Face artifact parity gate
```

No expected behavior may substitute for fresh evaluation output. Failed gates
produce a documented non-release experiment, not a weakened post-hoc test
criterion.

## 24. Primary references

- MAEST implementation and training code:
  <https://github.com/palonso/MAEST>
- `maest-infer` inference implementation:
  <https://github.com/openmirlab/maest-infer>
- MAEST paper:
  <https://archives.ismir.net/ismir2023/paper/000098.pdf>
- Official 519-label Hugging Face conversion:
  <https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e-519l>
- Hugging Face model release checklist:
  <https://huggingface.co/docs/hub/en/model-release-checklist>
- Hugging Face model card guidance:
  <https://huggingface.co/docs/hub/model-cards>
