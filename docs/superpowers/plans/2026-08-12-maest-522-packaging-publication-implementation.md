# MAEST 522 Packaging and Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible native MAEST 522 release plus a standard Hugging Face AST checkpoint whose logits match native inference, with model/dataset documentation and a gated upload workflow.

**Architecture:** The trained continual checkpoint is merged back into the original MAEST state-dict layout for native inference. A separate deterministic converter maps the same tensors into `ASTForAudioClassification`, including decomposed positional embeddings and split QKV projections. A release builder assembles immutable artifacts, hashes, reports, cards, and parity evidence in a local staging directory. Upload is a separate explicit command and is never part of tests.

**Tech Stack:** Python 3.10, PyTorch, maest-infer 0.2.0, Transformers, safetensors, huggingface-hub, JSON/Markdown.

## Global Constraints

- Do not replace this application's default `maest_519l_pytorch` model or checkpoint.
- Native 522 inference is a separately named public API and artifact.
- Labels 0-518 are unchanged; labels 519-521 are the three approved extension labels.
- The release builder consumes only a checkpoint that passed the training plan's validation and test gates.
- Native and Hugging Face outputs must be numerically compared on real 30-second windows before staging succeeds.
- Preserve both native heads (`head` and `head_dist`) in the native artifact.
- Hugging Face standard AST uses the mean pooled CLS/distillation representation and one merged 522 classifier.
- Do not assert an upstream license. The 519-label Hugging Face conversion currently has no explicit license metadata, while related upstream code/model repositories show different licenses. If no explicit grant for the exact source checkpoint is verified, use `license: other`, identify code and weight provenance separately, and do not switch the Hub repository to public visibility until the license review is recorded.
- Never include local audio, absolute paths, private annotations, authentication tokens, or caches in a release directory.
- Upload requires an explicit repository ID and `--push`; local preparation is the default.

---

### Task 1: Export and reload a native MAEST 522 checkpoint

**Files:**

- Create: `tools/maest522/native_model.py`
- Create: `tools/maest522/native_export.py`
- Create: `tests/maest522/test_native_export.py`
- Create: `tests/maest522/test_native_model.py`

- [ ] Add a merge test asserting exact legacy rows, trained extension rows, and preservation of all non-head tensors.

```python
merged = merge_native_state_dict(teacher_state, student_state)
for prefix in ("head.1", "head_dist"):
    torch.testing.assert_close(merged[f"{prefix}.weight"][:519], teacher_state[f"{prefix}.weight"])
    torch.testing.assert_close(merged[f"{prefix}.weight"][519:], student_state[f"{prefix}.extension_weight"])
self.assertEqual(merged["head.1.weight"].shape, (522, 768))
```

- [ ] Add a round-trip test: export, reload, run a mel batch, and compare all 522 logits.

- [ ] Run tests and confirm failure.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_native_export tests.maest522.test_native_model -v
```

- [ ] Implement strict native merging.

```python
def merge_native_state_dict(
    teacher_state: Mapping[str, Tensor],
    student_state: Mapping[str, Tensor],
) -> dict[str, Tensor]: ...

def export_native_release(
    teacher_checkpoint: Path,
    trained_checkpoint: Path,
    output_checkpoint: Path,
    labels_path: Path,
) -> NativeExportReport: ...
```

The output state dict retains MAEST keys and writes both `(522, 768)` classifiers. First 519 rows come from the immutable teacher, not from the student copy. Include metadata with format version, architecture name `maest_522l_pytorch`, label digest, source checkpoint digest, training run digest, and evaluation digest.

- [ ] Implement a separately named loader.

```python
def get_maest_522(checkpoint_path: Path, device: str = "cpu") -> nn.Module:
    model = get_maest(arch="maest_519l_pytorch", pretrained=False)
    replace_classifier_outputs(model, n_classes=522)
    model.load_state_dict(load_native_state(checkpoint_path), strict=True)
    return model.to(device).eval()

def predict_522(model: nn.Module, mel: Tensor) -> Tensor:
    return torch.sigmoid(model(mel))
```

Do not call the 519 factory with `n_classes=522`; the installed factory fixes 519 classes. Replace both linear layers before strict loading.

- [ ] Run tests, export into a temporary directory, and verify strict reload.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_native_export tests.maest522.test_native_model -v
.\.venv\Scripts\python.exe -m tools.maest522.native_export --help
```

- [ ] Commit native export support.

```powershell
git add tools/maest522/native_model.py tools/maest522/native_export.py tests/maest522/test_native_export.py tests/maest522/test_native_model.py
git commit -m "feat: export native MAEST 522 checkpoints"
```

### Task 2: Map native MAEST tensors into Hugging Face AST

**Files:**

- Create: `requirements-publication.txt`
- Create: `tools/maest522/hf_mapping.py`
- Create: `tools/maest522/hf_feature_extractor.py`
- Create: `tests/maest522/test_hf_mapping.py`
- Create: `tests/maest522/test_hf_feature_extractor.py`

- [ ] Add publication-only dependencies.

```text
transformers>=4.38,<5
huggingface-hub>=0.28,<2
safetensors>=0.4,<1
```

- [ ] Add unit tests for positional composition, QKV splitting, block mapping, classifier mapping, missing keys, unexpected keys, and shape mismatches. Add waveform-to-mel parity fixtures comparing the publication feature extractor with the installed native MAEST frontend.

```python
hf_state = convert_native_to_hf_state(native_state, config)
self.assertEqual(
    hf_state["audio_spectrogram_transformer.embeddings.position_embeddings"].shape,
    (1, 2 + config.num_patches, 768),
)
torch.testing.assert_close(
    hf_state["classifier.dense.weight"], native_state["head.1.weight"]
)
```

- [ ] Run the test and confirm failure.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_hf_mapping -v
```

- [ ] Implement positional conversion. Native patch tensors are produced with convolution layout `[batch, hidden, frequency, time]` and flattened frequency-major/time-minor. Compose them without interpolation when shapes already match.

```python
patch_position = (
    native_state["freq_new_pos_embed"]
    + native_state["time_new_pos_embed"]
).flatten(2).transpose(1, 2)
position_embeddings = torch.cat(
    (native_state["new_pos_embed"], patch_position), dim=1
)
```

Map `cls_token` and `dist_token` directly to HF `cls_token` and `distillation_token`. Native adds `new_pos_embed` to the two tokens; HF adds the combined `position_embeddings` afterward, which is algebraically equivalent.

- [ ] Implement every encoder mapping for blocks 0-11.

```text
blocks.N.norm1                         -> encoder.layer.N.layernorm_before
blocks.N.attn.qkv [split Q/K/V]        -> encoder.layer.N.attention.attention.query/key/value
blocks.N.attn.proj                     -> encoder.layer.N.attention.output.dense
blocks.N.norm2                         -> encoder.layer.N.layernorm_after
blocks.N.mlp.fc1                       -> encoder.layer.N.intermediate.dense
blocks.N.mlp.fc2                       -> encoder.layer.N.output.dense
norm                                   -> audio_spectrogram_transformer.layernorm
patch_embed.proj                       -> embeddings.patch_embeddings.projection
head.0                                 -> classifier.layernorm
head.1                                 -> classifier.dense
```

Split `attn.qkv.weight` and bias into equal contiguous Q/K/V chunks along output dimension. Reject non-divisible dimensions.

- [ ] Build an `ASTConfig` with `num_labels=522`, `num_mel_bins=96`, `max_length=1876`, frequency/time strides `10`, patch size `16`, hidden size `768`, 12 layers, 12 heads, intermediate size `3072`, hidden activation `gelu`, `layer_norm_eps=1e-6`, `qkv_bias=True`, `problem_type="multi_label_classification"`, and complete `id2label`/`label2id` maps. Derive and validate the patch count from actual convolution arithmetic rather than trusting a constant. Use the official compatible base identifier `mtg-upf/discogs-maest-30s-pw-129e-519l` in provenance.

- [ ] Implement and export the exact Hugging Face feature-extractor contract used by the official conversion: 16 kHz mono, FFT 512, hop 256, 96 Slaney mel bins, `log10(1 + 10000 * mel)`, maximum 1876 frames, mean `2.06755686098554`, standard deviation `1.268292820667291`, and the same normalization factor as native MAEST. Write `feature_extraction_maest.py` plus `preprocessor_config.json` with an `AutoFeatureExtractor` `auto_map`; preserve the upstream Apache-2.0 header and attribution.

- [ ] Load the mapped state with `strict=True` equivalence: after `load_state_dict`, both missing and unexpected key lists must be empty. Save with safetensors.

- [ ] Run focused tests.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_hf_mapping tests.maest522.test_hf_feature_extractor -v
```

- [ ] Commit the converter.

```powershell
git add requirements-publication.txt tools/maest522/hf_mapping.py tools/maest522/hf_feature_extractor.py tests/maest522/test_hf_mapping.py tests/maest522/test_hf_feature_extractor.py
git commit -m "feat: map MAEST 522 to Hugging Face AST"
```

### Task 3: Prove preprocessing and logit parity on real audio windows

**Files:**

- Create: `tools/maest522/parity.py`
- Create: `tests/maest522/test_parity.py`

- [ ] Add tensor-level tests that native input `[batch, 96, time]` becomes HF input `[batch, time, 96]` without numeric modification, and that sigmoid is applied exactly once outside both raw-logit models.

- [ ] Implement a parity report.

```python
@dataclass(frozen=True)
class ParityReport:
    windows: int
    max_absolute_logit_error: float
    mean_absolute_logit_error: float
    max_absolute_probability_error: float
    top10_match_rate: float
    passed: bool

def compare_native_and_hf(
    native_model: nn.Module,
    hf_model: ASTForAudioClassification,
    mel_batches: Iterable[Tensor],
    atol: float = 1e-5,
    probability_atol: float = 1e-6,
) -> ParityReport: ...
```

- [ ] Run the test and confirm failure, then implement collection in float32/eval/no-grad mode.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_parity -v
```

- [ ] Add a CLI that accepts an explicit local audio list, selects the same 20%/50%/80% windows, creates MAEST mels once, feeds the native tensor and its transpose to the two models, and emits `parity.json`. Require at least 30 windows spanning at least 10 tracks for a publishable report.

- [ ] Use strict parity gates: maximum absolute raw-logit difference `<= 1e-5`, maximum absolute sigmoid-probability difference `<= 1e-6`, and exact top-10 index equality for every window. If the local Transformers version produces a justified larger floating-point difference on GPU, rerun both models on CPU before allowing failure thresholds to change.

- [ ] Run unit tests and a real-window parity run in a temporary output directory.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_parity -v
.\.venv\Scripts\python.exe -m tools.maest522.parity --help
```

- [ ] Commit parity tooling.

```powershell
git add tools/maest522/parity.py tests/maest522/test_parity.py
git commit -m "test: verify native and Hugging Face MAEST parity"
```

### Task 4: Build reproducible model and dataset cards

**Files:**

- Create: `tools/maest522/cards.py`
- Create: `tools/maest522/templates/model-card.md.j2`
- Create: `tools/maest522/templates/dataset-card.md.j2`
- Create: `tests/maest522/test_cards.py`

- [ ] Add card tests that render a complete fixture and reject missing provenance, license status, evaluation, intended-use, limitation, or annotation sections.

- [ ] Define the model card front matter and required body fields.

```yaml
library_name: transformers
pipeline_tag: audio-classification
license: other
base_model: mtg-upf/discogs-maest-30s-pw-129e-519l
tags:
  - audio-classification
  - music
  - continual-learning
  - multi-label-classification
```

The body must name the official source checkpoint and SHA-256; explain 519 preservation and three appended rows; list the exact label order; describe 16 kHz/30-second/96-mel preprocessing; state multi-window aggregation; summarize training data without publishing copyrighted audio; publish all new-label and legacy regression metrics; link the annotation protocol; describe known confusion among adjacent minimal-house styles; and state that Discogs labels and human genre judgments are subjective.

- [ ] Define the dataset card for metadata/annotations only. It must document consent/provenance boundaries, state values, uncertainty masking, hard-negative policy, group splitting, duplicate detection, blind holdouts, counts per label/split/state, and fields in the portable JSONL. Do not imply redistribution rights for audio.

- [ ] Render cards deterministically from `release-metadata.json` and fail on unfilled template values.

```python
def render_release_cards(metadata: ReleaseMetadata, output_dir: Path) -> CardReport: ...
```

- [ ] Run focused tests.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_cards -v
```

- [ ] Commit card generation.

```powershell
git add tools/maest522/cards.py tools/maest522/templates tests/maest522/test_cards.py
git commit -m "docs: generate MAEST 522 release cards"
```

### Task 5: Assemble and audit a local release staging directory

**Files:**

- Create: `tools/maest522/release.py`
- Create: `tests/maest522/test_release.py`

- [ ] Add a release test with fixture reports. It must reject failed evaluation gates, failed parity, label mismatch, non-519 source heads, absolute paths in JSON/Markdown, and unrecognized files.

- [ ] Implement an allowlisted release builder.

```python
MODEL_ALLOWLIST = {
    "config.json",
    "model.safetensors",
    "native-maest-522.ckpt",
    "labels-522.txt",
    "feature_extraction_maest.py",
    "preprocessor_config.json",
    "README.md",
    "evaluation.json",
    "parity.json",
    "provenance.json",
    "SHA256SUMS",
}

def stage_release(inputs: ReleaseInputs, output_dir: Path) -> ReleaseReport: ...
```

Build into a new empty directory. Copy by explicit allowlist, write to temporary names, atomically rename, then compute sorted `SHA256SUMS`. Scan text artifacts for Windows drive paths, home-directory fragments, token-shaped strings, and annotation notes. Scan every safetensors key/shape and native tensor key/shape.

- [ ] Create `provenance.json` containing tool versions, git commit, source checkpoint digest, trained checkpoint digest, native artifact digest, HF artifact digest, label digest, dataset manifest digest, split-audit digest, evaluation digest, parity digest, and UTC build time.

- [ ] Validate from the staged directory in a clean subprocess: load native checkpoint; load HF checkpoint with `local_files_only=True`; compare one deterministic synthetic mel and the saved real-window parity fixture; verify cards and all hashes.

- [ ] Run focused tests and a local staging dry run.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_release -v
.\.venv\Scripts\python.exe -m tools.maest522.release --help
```

- [ ] Commit the release builder.

```powershell
git add tools/maest522/release.py tests/maest522/test_release.py
git commit -m "feat: stage audited MAEST 522 releases"
```

### Task 6: Add an explicit, safe Hugging Face upload workflow

**Files:**

- Create: `tools/maest522/hub_publish.py`
- Create: `tests/maest522/test_hub_publish.py`
- Create: `docs/maest522-publication-guide.md`

- [ ] Add mocked Hub tests proving that the default command performs validation only and that upload requires both `--repo-id` and `--push`.

```python
self.assertEqual(main(["--release-dir", str(release_dir)]), 0)
mock_api.upload_folder.assert_not_called()
```

- [ ] Implement publication with `huggingface_hub.HfApi`, never raw HTTP. Read authentication from the standard HF token mechanism; do not accept or log a token CLI argument.

```python
def validate_for_hub(release_dir: Path) -> HubValidationReport: ...

def publish_model(
    release_dir: Path,
    repo_id: str,
    private: bool,
    revision: str,
) -> str: ...
```

Default to `private=True` for a newly created remote repository. Require an explicit `--public` switch to create/publicize a public repository. Use a commit message containing the release version and provenance digest. Never delete remote files.

- [ ] Document a two-phase human workflow: local validation; private upload and smoke test; review rendered card/files/licensing; explicit public visibility change. Include native and Transformers inference examples.

```python
from transformers import ASTForAudioClassification

model = ASTForAudioClassification.from_pretrained("ORG/discogs-maest-30s-pw-129e-522l")
model.eval()
```

The example must show both supported paths: `AutoFeatureExtractor.from_pretrained(..., trust_remote_code=True)` for raw 16 kHz mono audio, and the native MAEST mel frontend with transpose from `[batch, 96, time]` to `[batch, time, 96]`. Do not imply that the generic AST feature extractor is equivalent.

- [ ] Run focused tests, syntax checks, and local validation without upload.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_hub_publish -v
.\.venv\Scripts\python.exe -m py_compile tools\maest522\native_model.py tools\maest522\native_export.py tools\maest522\hf_mapping.py tools\maest522\hf_feature_extractor.py tools\maest522\parity.py tools\maest522\cards.py tools\maest522\release.py tools\maest522\hub_publish.py
.\.venv\Scripts\python.exe -m tools.maest522.hub_publish --help
git diff --check
```

- [ ] Commit publication support.

```powershell
git add tools/maest522/hub_publish.py tests/maest522/test_hub_publish.py docs/maest522-publication-guide.md
git commit -m "feat: prepare MAEST 522 Hugging Face publication"
```

### Task 7: Run the complete release rehearsal

**Files:**

- Create: `tests/maest522/test_release_workflow.py`

- [ ] Add a small end-to-end release test that converts a compact MAEST-shaped fixture into native and HF artifacts, verifies parity, renders cards, stages the release, reloads both formats, and runs Hub validation with no network call.

- [ ] Run the publication test group.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.maest522.test_native_export tests.maest522.test_native_model tests.maest522.test_hf_mapping tests.maest522.test_hf_feature_extractor tests.maest522.test_parity tests.maest522.test_cards tests.maest522.test_release tests.maest522.test_hub_publish tests.maest522.test_release_workflow -v
```

- [ ] Run the repository verification checklist appropriate to the added tooling.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m py_compile src\config.py src\environment.py src\pipeline.py src\extractor.py src\tagger.py src\report.py src\main.py src\storage.py tools\annotation_ui.py
.\.venv\Scripts\python.exe -m compileall -q tools\maest522
.\.venv\Scripts\python.exe src\main.py --help
git diff --check
```

- [ ] Inspect the staged release allowlist, hashes, license language, and provenance manually. Record the reviewer's name/date in a file outside the immutable model artifact, then generate the final staging directory again from source inputs.

- [ ] Commit the workflow test.

```powershell
git add tests/maest522/test_release_workflow.py
git commit -m "test: rehearse MAEST 522 release workflow"
```

## Completion Gate

- Native strict reload produces a 522-logit MAEST model and retains both distilled heads.
- HF AST strict load has no missing or unexpected keys.
- Native/HF parity passes on at least 30 real windows from at least 10 tracks.
- The release contains no audio, private paths, caches, tokens, or annotation notes.
- Evaluation and provenance digests match the artifacts in `SHA256SUMS`.
- Licensing language distinguishes source-code license, checkpoint provenance, dataset annotations, and audio rights.
- Running the publication command without `--push` performs no remote mutation.
