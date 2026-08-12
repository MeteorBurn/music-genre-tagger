# MAEST 522 Training and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the official MAEST 519-label checkpoint into a native 522-label model and perform hybrid continual fine-tuning that learns three new labels without materially degrading the original 519 outputs.

**Architecture:** The official MAEST checkpoint remains the immutable teacher. A student preserves frozen 519-row legacy heads and adds three trainable rows to both distilled heads. Training consumes the frozen annotation manifest, fixed 20/50/80 windows, replay examples for old-label behavior, cached teacher logits, and staged unfreezing. Evaluation has two independent surfaces: new-label classification quality and old-label regression gates.

**Tech Stack:** Python 3.10, PyTorch, torchaudio, maest-infer 0.2.0, NumPy, SoundFile/ffmpeg, scikit-learn, safetensors.

## Global Constraints

- The first 519 label strings and row ordering are byte-for-byte identical to the official label list.
- The new rows are exactly 519 Microhouse, 520 RoMinimal, 521 DeepTech-Minimal.
- Never train a separate classifier over detached MAEST embeddings; gradients flow through native MAEST heads and, in later stages, selected transformer blocks.
- The old 519 head rows remain frozen throughout training.
- Teacher outputs are generated with the original checkpoint, preprocessing, and fixed windows; cache keys include all three identities.
- Split membership comes only from the frozen annotation manifest. Training code must reject duplicate group IDs across splits.
- `uncertain` targets are masked, not coerced to negative.
- Select checkpoints using validation only. Run test evaluation once for a release candidate.
- Record seeds, checkpoint SHA-256, manifest SHA-256, label-list SHA-256, preprocessing version, dependency versions, and every stage transition.

---

### Task 1: Add training dependencies, canonical labels, and exact 519-to-522 tensor surgery

**Files:**

- Create: `requirements-training.txt`
- Create: `tools/maest522/model_labels.py`
- Create: `tools/maest522/checkpoint.py`
- Create: `tests/maest522/test_model_labels.py`
- Create: `tests/maest522/test_checkpoint.py`

- [ ] Add bounded dependencies without changing the application's baseline requirements.

```text
scikit-learn>=1.5,<2
safetensors>=0.4,<1
```

- [ ] Add a test that loads the official label file through `maest_infer`, appends the three labels, and asserts exact indices and uniqueness.

```python
labels = build_522_labels(load_official_519_labels())
self.assertEqual(len(labels), 522)
self.assertEqual(labels[519:], list(NEW_LABELS))
self.assertEqual(len(set(labels)), 522)
```

- [ ] Add deterministic tensor-surgery tests for all four tensors.

```python
expanded = expand_classifier_state_dict(original, prior_probability=0.01, seed=522)
for prefix in ("head.1", "head_dist"):
    torch.testing.assert_close(
        expanded[f"{prefix}.weight"][:519],
        original[f"{prefix}.weight"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        expanded[f"{prefix}.bias"][:519],
        original[f"{prefix}.bias"],
        rtol=0,
        atol=0,
    )
self.assertEqual(expanded["head.1.weight"].shape, (522, 768))
self.assertEqual(expanded["head_dist.weight"].shape, (522, 768))
```

- [ ] Run both tests and confirm failure.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_model_labels tests.maest522.test_checkpoint -v
```

- [ ] Implement canonical label creation and strict source validation.

```python
def build_522_labels(labels_519: Sequence[str]) -> tuple[str, ...]: ...

def validate_checkpoint_519(state_dict: Mapping[str, Tensor]) -> None: ...

def expand_classifier_state_dict(
    state_dict: Mapping[str, Tensor],
    prior_probability: float = 0.01,
    seed: int = 522,
) -> dict[str, Tensor]: ...
```

Initialize the three new weight rows with a local seeded `torch.Generator` and truncated normal matching the original head row standard deviation. Initialize new biases to `log(p / (1 - p))`. Clone every tensor so the teacher mapping cannot be mutated by aliasing. Reject checkpoints whose head shapes are not exactly `(519, 768)` and `(519,)`.

- [ ] Add an atomic CLI conversion that writes `expanded-init.ckpt`, `labels-522.txt`, and `expansion-report.json` with source/output SHA-256 values.

- [ ] Run focused tests and a conversion against `src/models/discogs-maest-30s-pw-129e-519l-swa.ckpt` into a task-specific temporary directory.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_model_labels tests.maest522.test_checkpoint -v
.\.venv\Scripts\python.exe -m tools.maest522.checkpoint --help
```

- [ ] Commit checkpoint expansion.

```powershell
git add requirements-training.txt tools/maest522/model_labels.py tools/maest522/checkpoint.py tests/maest522/test_model_labels.py tests/maest522/test_checkpoint.py
git commit -m "feat: expand MAEST classifier to 522 labels"
```

### Task 2: Build a native continual model with frozen legacy rows

**Files:**

- Create: `tools/maest522/continual_model.py`
- Create: `tests/maest522/test_continual_model.py`

- [ ] Add a model test using a small fake MAEST backbone that proves shape, concatenation order, and gradient isolation.

```python
output = model(batch)
self.assertEqual(output.logits.shape, (2, 522))
self.assertEqual(output.legacy_logits.shape, (2, 519))
self.assertEqual(output.new_logits.shape, (2, 3))
output.logits.sum().backward()
self.assertIsNone(model.legacy_head.weight.grad)
self.assertIsNotNone(model.extension_head.weight.grad)
```

- [ ] Run the test and confirm failure.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_continual_model -v
```

- [ ] Implement a wrapper around the actual MAEST feature path.

```python
@dataclass(frozen=True)
class ContinualOutput:
    logits: Tensor
    legacy_logits: Tensor
    new_logits: Tensor
    auxiliary_new_logits: Tensor
    cls_embedding: Tensor
    dist_embedding: Tensor


class ContinualMaest522(nn.Module):
    def forward(self, mel: Tensor) -> ContinualOutput:
        cls_embedding, dist_embedding = self.backbone.forward_features(mel)
        pooled = (cls_embedding + dist_embedding) / 2
        pooled = self.head_norm(pooled)
        legacy_logits = self.legacy_head(pooled)
        new_logits = self.extension_head(pooled)
        auxiliary_new_logits = self.extension_dist_head(dist_embedding)
        return ContinualOutput(
            logits=torch.cat((legacy_logits, new_logits), dim=-1),
            legacy_logits=legacy_logits,
            new_logits=new_logits,
            auxiliary_new_logits=auxiliary_new_logits,
            cls_embedding=cls_embedding,
            dist_embedding=dist_embedding,
        )
```

Load legacy weights from `head.1[:519]`; load the extension rows from `head.1[519:]`; load the auxiliary extension from `head_dist[519:]`. Preserve the original `head_dist[:519]` in the checkpoint exporter even though mean-head inference does not consume it.

- [ ] Implement named stage policies.

```python
class TrainingStage(str, Enum):
    EXTENSION_HEADS = "extension_heads"
    BLOCKS_10_11 = "blocks_10_11"
    BLOCKS_8_11 = "blocks_8_11"

def apply_training_stage(model: ContinualMaest522, stage: TrainingStage) -> None: ...
```

Stage 1 trains extension heads only. Stage 2 also trains transformer blocks 10 and 11 plus final normalization. Stage 3 also trains blocks 8 and 9 and is enabled only when Stage 2 misses validation targets without crossing regression gates.

- [ ] Run tests and verify trainable parameter names against an actual MAEST instance.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_continual_model -v
```

- [ ] Commit the continual model.

```powershell
git add tools/maest522/continual_model.py tests/maest522/test_continual_model.py
git commit -m "feat: add MAEST continual model wrapper"
```

### Task 3: Produce exact audio windows and immutable teacher/replay caches

**Files:**

- Create: `tools/maest522/training_data.py`
- Create: `tools/maest522/cache.py`
- Create: `tests/maest522/test_training_data.py`
- Create: `tests/maest522/test_cache.py`

- [ ] Add tests for 20%/50%/80% centered offsets, short-track clamping, duplicate-offset removal, uncertainty masks, split leakage rejection, and cache invalidation.

```python
sample = decode_manifest_row(row)
self.assertEqual(sample.targets.tolist(), [1.0, 0.0, 0.0])
self.assertEqual(sample.target_mask.tolist(), [1.0, 1.0, 0.0])
```

- [ ] Run tests and confirm failure.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_training_data tests.maest522.test_cache -v
```

- [ ] Implement the same window selection contract as production: 30 seconds, centers at 0.2/0.5/0.8, clamped starts, deduplicated offsets, mono 16 kHz. Use the installed MAEST `MelSpectrogram` implementation so preprocessing is not approximated.

```python
def select_window_offsets(
    duration_seconds: float,
    window_seconds: float = 30.0,
    positions: tuple[float, ...] = (0.2, 0.5, 0.8),
) -> tuple[float, ...]: ...

def load_manifest(path: Path, expected_split_audit_sha256: str) -> DatasetManifest: ...
```

Reject a manifest when group IDs overlap splits, an audio reference is missing, a completed row lacks one of three labels, or the manifest/audit digest differs from the training config.

- [ ] Implement content-addressed cache records.

```python
@dataclass(frozen=True)
class CacheKey:
    track_id: str
    offset_milliseconds: int
    teacher_sha256: str
    preprocessing_version: str

class TeacherCache:
    def get(self, key: CacheKey) -> TeacherRecord | None: ...
    def put_atomic(self, key: CacheKey, record: TeacherRecord) -> None: ...
```

Each teacher record stores 519 logits and the pooled 768-dimensional embedding. Cache files are `safetensors`; a SQLite index records key, shape, dtype, checksum, and creation time. Verify checksum and tensor shapes on read.

- [ ] Define replay inputs as a separate manifest containing unlabeled/legacy Discogs-like tracks. Replay items never contribute to the three-label BCE; they contribute teacher distillation and L2-SP only. Keep a disjoint regression-holdout manifest for release gates.

- [ ] Run focused tests.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_training_data tests.maest522.test_cache -v
```

- [ ] Commit the data/cache layer.

```powershell
git add tools/maest522/training_data.py tools/maest522/cache.py tests/maest522/test_training_data.py tests/maest522/test_cache.py
git commit -m "feat: add continual training data caches"
```

### Task 4: Implement masked new-label loss and old-label protection

**Files:**

- Create: `tools/maest522/losses.py`
- Create: `tests/maest522/test_losses.py`

- [ ] Add numerical tests for masking, class weights, teacher-logit distillation, L2-SP, and auxiliary distillation-token supervision.

```python
losses = continual_loss(
    output=student_output,
    new_targets=targets,
    new_mask=mask,
    teacher_legacy_logits=teacher_logits,
    trainable_parameters=trainable,
    reference_parameters=reference,
    weights=LossWeights(new=1.0, distill=2.0, l2sp=1e-4, auxiliary=0.25),
)
self.assertEqual(losses.total.ndim, 0)
self.assertEqual(losses.new_label.item(), expected_masked_bce)
```

- [ ] Run the test and confirm failure.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_losses -v
```

- [ ] Implement the objective.

```python
L_new = masked_weighted_bce(student.new_logits, targets, mask, positive_weights)
L_old = F.smooth_l1_loss(student.legacy_logits, teacher_legacy_logits)
L_l2sp = l2_starting_point(unfrozen_pretrained_parameters, reference_parameters)
L_aux = masked_weighted_bce(
    student.auxiliary_new_logits, targets, mask, positive_weights
)
total = w.new * L_new + w.distill * L_old + w.l2sp * L_l2sp + w.auxiliary * L_aux
```

Compute positive weights from the training split only, clip them to `[1, 10]`, and persist the values. Apply L2-SP only to unfrozen pretrained backbone/final-normalization parameters; the three new head rows are excluded. When a batch contains no supervised target cells, return a differentiable zero for `L_new` and `L_aux`.

- [ ] Run focused tests including `torch.autograd.gradcheck` on a double-precision tiny case.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_losses -v
```

- [ ] Commit loss protection.

```powershell
git add tools/maest522/losses.py tests/maest522/test_losses.py
git commit -m "feat: protect legacy MAEST outputs during training"
```

### Task 5: Implement staged, resumable hybrid continual fine-tuning

**Files:**

- Create: `tools/maest522/train.py`
- Create: `tests/maest522/test_train.py`

- [ ] Add CPU tests with the fake backbone for gradient accumulation, deterministic resume, optimizer groups, early stopping, and stage transitions.

- [ ] Define a serializable training configuration.

```python
@dataclass(frozen=True)
class TrainingConfig:
    teacher_checkpoint: Path
    annotation_manifest: Path
    replay_manifest: Path
    output_dir: Path
    seed: int = 522
    batch_size: int = 8
    accumulation_steps: int = 4
    head_learning_rate: float = 3e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_epochs_per_stage: int = 30
    patience: int = 5
```

- [ ] Run the tests and confirm failure.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_train -v
```

- [ ] Implement a pure-PyTorch loop with AMP on CUDA, gradient clipping at `1.0`, seeded sampling, train-only SpecAugment, balanced annotation/replay batches, validation after every epoch, and atomic checkpoints.

The annotation/replay target mix is 1:1 by batch when enough replay examples exist. Stage 1 uses extension-head learning rate only. Stage 2/3 use separate optimizer groups for heads and backbone. Rebuild the optimizer at stage boundaries; do not carry stale state for newly unfrozen parameters.

- [ ] Select the best epoch lexicographically:

1. Must pass the validation regression gates available at that stage.
2. Maximize macro average precision across the three new labels.
3. Break ties with macro F1 at validation-derived thresholds.
4. Break remaining ties with lower legacy probability drift.

- [ ] Persist `last.ckpt`, `best.ckpt`, `metrics.jsonl`, `resolved-config.json`, `environment.json`, and `run-manifest.json`. A resume verifies every input digest and refuses silent drift.

- [ ] Run the CPU training tests and a one-batch dry run against the actual checkpoint without committing generated artifacts.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_train -v
.\.venv\Scripts\python.exe -m tools.maest522.train --help
```

- [ ] Commit the trainer.

```powershell
git add tools/maest522/train.py tests/maest522/test_train.py
git commit -m "feat: add hybrid MAEST continual trainer"
```

### Task 6: Implement classification metrics, calibration, and legacy regression gates

**Files:**

- Create: `tools/maest522/metrics.py`
- Create: `tools/maest522/evaluate.py`
- Create: `tests/maest522/test_metrics.py`
- Create: `tests/maest522/test_evaluate.py`

- [ ] Add fixed-array tests for AP, precision, recall, F1, confusion counts, expected calibration error, threshold selection, and every regression gate.

```python
report = evaluate_legacy_regression(teacher, student)
self.assertLessEqual(report.mean_probability_drift, 0.01)
self.assertGreaterEqual(report.mean_top10_overlap, 0.90)
self.assertGreaterEqual(report.teacher_top3_in_student_top5, 0.98)
self.assertGreaterEqual(report.embedding_cosine, 0.98)
```

- [ ] Run tests and confirm failure.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_metrics tests.maest522.test_evaluate -v
```

- [ ] Implement new-label evaluation per label and macro/micro aggregate. Choose one sigmoid threshold per new label on validation by maximum F1 subject to a configurable minimum precision; freeze those thresholds before test evaluation. Report hard-negative subsets separately.

- [ ] Implement the release regression gates exactly:

```text
mean absolute probability drift <= 0.01
mean |teacher_top10 intersect student_top10| / 10 >= 0.90
teacher top-3 contained in student top-5 >= 0.98
mean pooled-embedding cosine >= 0.98
frequently-active old class mean drift <= 0.03
frequently-active old class p95 drift <= 0.10
```

Define frequently active as an old label whose teacher probability is at least `0.10` on at least 25 regression-holdout windows. Persist the qualifying label indices and counts.

- [ ] Aggregate each track by mean window probability before track-level metrics, while also publishing window-level variance for stability. Bootstrap 95% confidence intervals by resampling group IDs, not individual windows.

- [ ] Write machine-readable `evaluation.json`, flat `metrics.csv`, and a Markdown report. The command needs an explicit `--allow-test` flag and records whether test has already been evaluated in the run directory.

- [ ] Run focused tests and evaluate the teacher against itself as a zero-drift parity check.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_metrics tests.maest522.test_evaluate -v
.\.venv\Scripts\python.exe -m tools.maest522.evaluate --help
```

- [ ] Commit evaluation.

```powershell
git add tools/maest522/metrics.py tools/maest522/evaluate.py tests/maest522/test_metrics.py tests/maest522/test_evaluate.py
git commit -m "feat: evaluate MAEST 522 quality and regression"
```

### Task 7: Close the annotation loop with train-only active learning

**Files:**

- Create: `tools/maest522/active_learning.py`
- Create: `tests/maest522/test_active_learning.py`

- [ ] Add tests that exclude reviewed tracks, all val/test tracks, group siblings, and tracks at the per-label cap.

- [ ] Implement a reproducible acquisition score over the training pool.

```python
score = (
    0.35 * normalized_entropy
    + 0.25 * window_disagreement
    + 0.25 * hard_negative_score
    + 0.15 * diversity_score
)
```

`hard_negative_score` emphasizes high student probability for a label contradicted by source hints or teacher-neighbor context. `diversity_score` uses farthest-first selection on normalized MAEST pooled embeddings after the other components rank a bounded candidate set.

- [ ] Export only `track_id -> acquisition_score` plus a diagnostic JSON. The annotation queue module remains responsible for quotas and final deduplication.

```python
def score_training_pool(
    model: ContinualMaest522,
    manifest: DatasetManifest,
    reviewed_track_ids: set[str],
    seed: int = 522,
) -> dict[str, float]: ...
```

- [ ] Run the focused test and a dry-run command.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_active_learning -v
.\.venv\Scripts\python.exe -m tools.maest522.active_learning --help
```

- [ ] Commit acquisition logic.

```powershell
git add tools/maest522/active_learning.py tests/maest522/test_active_learning.py
git commit -m "feat: rank train-only MAEST annotation candidates"
```

### Task 8: Run the pre-publication training validation suite

**Files:**

- Create: `docs/maest522-training-guide.md`
- Create: `tests/maest522/test_training_workflow.py`

- [ ] Add an end-to-end tiny-model test covering expanded load, cached teacher inference, Stage 1 optimization, checkpoint resume, validation threshold selection, and active-learning export.

- [ ] Run the MAEST 522 training tests.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/maest522 -p "test_*train*.py" -v
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_checkpoint tests.maest522.test_continual_model tests.maest522.test_losses tests.maest522.test_metrics tests.maest522.test_evaluate tests.maest522.test_active_learning -v
```

- [ ] Document cache preparation, Stage 0 parity, Stage 1/2/3 commands, validation-only iteration, active-learning round export, final test authorization, artifact locations, interruption/resume, and GPU memory knobs.

- [ ] Run syntax, help, and diff checks.

```powershell
.\.venv\Scripts\python.exe -m py_compile tools\maest522\checkpoint.py tools\maest522\continual_model.py tools\maest522\training_data.py tools\maest522\cache.py tools\maest522\losses.py tools\maest522\train.py tools\maest522\metrics.py tools\maest522\evaluate.py tools\maest522\active_learning.py
.\.venv\Scripts\python.exe -m tools.maest522.train --help
.\.venv\Scripts\python.exe -m tools.maest522.evaluate --help
git diff --check
```

- [ ] Commit the guide and workflow test.

```powershell
git add docs/maest522-training-guide.md tests/maest522/test_training_workflow.py
git commit -m "docs: add MAEST 522 training workflow"
```

## Completion Gate

- The expanded initialization is bit-identical for all four legacy head tensors.
- The student exposes one native 522-vector, not a detached secondary classifier.
- Only train split examples and replay examples update weights.
- All uncertain cells are masked and hard negatives are explicit negatives.
- The selected checkpoint passes every legacy regression gate on validation.
- Final test results use frozen thresholds and a single recorded test run.
- Active-learning output contains train IDs only and is consumable by the annotation queue plan.
