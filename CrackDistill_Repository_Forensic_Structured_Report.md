# CrackDistill Repository Forensic Audit — Structured Report

**Project:** CrackDistill  
**Evidence basis:** Repository forensic investigation performed by Opus  
**Audited commit:** `984df279794c9c7d40808d7438c0cfa28486c7c0`  
**Repository state:** clean working tree at time of audit  
**Purpose:** Consolidate the repository-level evidence into a publication-oriented scientific audit, separating confirmed bugs, refuted concerns, unresolved risks, corrected results, and the highest-value next actions.

---

# 1. Executive Summary

The repository audit materially changes the interpretation of the current CrackDistill paper.

The strongest conclusion is:

> **The distillation story is currently unproven because the implemented KD supervision is geometrically misaligned with the augmented student inputs, while the tiled full-resolution inference result is the strongest verified contribution that survives the audit.**

The audit also finds that several headline manuscript numbers are unsupported by the committed executed artifacts, the nominal multi-seed experiment did not actually vary the Ultralytics seed, the "OOD" set is not an external domain but the uncropped Crack500 validation parents, and the no-KD baseline was never evaluated on the reported OOD/tiled protocol.

At the same time, several earlier concerns were refuted:

- the Bernoulli KL implementation is correct,
- canonical Crack500 split construction is preserved,
- parent-scene leakage across train/val/test was not found,
- reconstructed mosaics are train-only,
- teacher logits are genuinely stored per instance,
- OOD labels come from complete original full-scene masks,
- direct and tiled Dice use the same mask threshold,
- the reported tiled Dice recovery range is broadly supported,
- and the student/teacher spatial resize path is explicitly implemented.

The current paper should therefore be **restructured around verified evidence**, while the KD claims should be re-established after fixing the training pipeline.

---

# 2. Highest-Severity Findings

| Finding | Status | Scientific impact |
|---|---|---|
| Teacher targets are not transformed with student geometric augmentation | **Confirmed critical bug** | Invalidates causal interpretation of all KD results |
| `project.seed` is not forwarded to Ultralytics | **Confirmed bug** | The claimed multi-seed verification is invalid |
| `04_affinity` committed executed result is `0.5348`, not `0.5569` | **Confirmed contradiction** | Current headline in-domain claim must be withdrawn |
| No-KD baseline was never evaluated on OOD/tiled protocol | **Confirmed** | All "gain over baseline" OOD/tiled claims are unsupported |
| Table 3 baseline OOD row duplicates `01_seed42` | **Confirmed** | Baseline comparison is not a measurement |
| OOD set is Crack500 uncropped validation parents | **Confirmed** | This is scale/context shift, not external domain shift |
| τ and α came from OOD-aware Optuna search | **Confirmed** | Manuscript HPO narrative is incorrect; OOD val was optimized directly |
| LayerKD hooks wrong layers/strides | **Confirmed bug** | Mechanistic LayerKD interpretation is invalid |
| Glob-based checkpoint resolver remains active | **Confirmed bug** | Cross-dataset checkpoint identity is unreliable |
| `04_affinity` tiled Dice is `0.2594`, not `0.2683` | **Confirmed contradiction** | Manuscript recommendation table is wrong |
| >100 FPS deployment claim is model-call throughput, not full-scene throughput | **Confirmed overstatement** | Deployment claim must be rewritten |
| Tiled inference improves Dice by roughly 52–85% across evaluated checkpoints | **Confirmed result** | Strongest verified result in the project |

---

# 3. Repository Evidence Map

The audit found strong code and notebook evidence, but important runtime artifacts were absent.

## Present

- `distillation/kd_trainer.py`
- `teacher/sam2_teacher.py`
- `scripts/generate_teacher_logits.py`
- `scripts/convert_crack500.py`
- `scripts/convert_crack500_uncropped.py`
- `scripts/convert_deepcrack.py`
- `scripts/mine_negative_and_mosaic_tiles.py`
- `scripts/tune_kd_weights.py`
- `configs/config.yaml`
- executed training/evaluation notebooks under `final_notebooks/output_runned/`
- executed OOD notebooks under `OODimprovements/output_runned/`
- executed Kaggle notebooks and logs

## Missing from the committed repository

- trained `.pt` checkpoints,
- dataset files,
- cached SAM teacher logits,
- cached teacher features,
- Optuna database,
- prediction dumps,
- executed `02_seed123` artifact.

## Effective runtime environment recovered from logs

- Ultralytics `8.4.120`
- Python `3.12.13`
- PyTorch `2.10.0+cu128`
- Tesla T4
- actual runs used `imgsz=512`
- actual optimizer resolved to AdamW via `optimizer=auto`
- AMP disabled
- actual Ultralytics seed remained `0`

---

# 4. Dataset and Evaluation Protocol Findings

## 4.1 Canonical Crack500 split exists in the repository

The repository logs show:

- train: `1,896`
- validation: `348`
- test: `1,124`

This means the repository itself is consistent with the canonical Crack500 split.

### Consequence

The manuscript's `348 test` statement is a reporting error.

More importantly, the final production runs were evaluated on the **348-image validation split**, not the 1,124-image test split.

---

## 4.2 Parent-scene split leakage was not found

The data conversion path preserves the official train/val/test directory structure.

The audit did not find evidence of random patch-level re-splitting across parent scenes.

### Verdict

> **Earlier leakage concern refuted.**

---

## 4.3 The development set was reused for model selection/reporting

The 348-image validation split was used for routine validation and reported in-domain performance.

The OOD-aware Optuna search also optimized against the uncropped validation-parent evaluation.

### Scientific implication

The current in-domain and OOD headline numbers are development results rather than untouched confirmatory test results.

### Required correction

Freeze the repaired method and perform the final confirmatory evaluation on:

- the canonical 1,124 Crack500 test crops,
- and the 200 uncropped Crack500 test parents where appropriate.

---

# 5. What the "OOD" Dataset Actually Is

The audit resolves the OOD provenance:

> The OOD set is the **50 uncropped Crack500 validation parent images**.

These are the same physical validation scenes from which the 348 in-domain validation crops originate.

Therefore the shift is mainly:

- scale shift,
- crop-context shift,
- full-scene background shift,
- crack-width/resolution shift.

It is **not** an independent domain shift.

## Recommended terminology

Replace:

> OOD generalization

with something such as:

> full-resolution scale/context-shift evaluation on uncropped Crack500 validation parents

until a genuinely independent external domain is evaluated.

---

# 6. Critical Bug #1 — Teacher Targets Are Geometrically Misaligned

This is the most serious implementation finding.

## Observed implementation

`KDYOLODataset.__getitem__` obtains the student sample from the underlying Ultralytics dataset first.

The returned sample has already undergone training augmentation.

The SAM teacher target is then loaded from disk by original filename stem.

No corresponding geometric transform is applied to the cached teacher logits.

## Executed training augmentations included

```text
mosaic = 1.0
fliplr = 0.5
scale = 0.5
translate = 0.1
erasing = 0.4
close_mosaic = 10
```

## Why this is critical

For most of training:

- the student sees a transformed or mosaic-composed image,
- the GT instances correspond to the transformed/composed sample,
- but the SAM tensor remains in the original untransformed coordinate system.

Under mosaic, a training sample may contain four source images while the SAM target loaded by filename stem corresponds to only one original source.

Under horizontal flip, approximately half the relevant samples are mirrored while the teacher target remains unmirrored.

### Scientific consequence

The current experiments cannot establish that the student learned from correctly aligned SAM structural priors.

The safest interpretation of existing runs is:

> **YOLO trained with a spatially noisy auxiliary loss derived from SAM tensors.**

The old scores are still real measurements of those training runs, but they are not valid evidence for the intended KD mechanism.

---

# 7. Critical Bug #2 — Silent Instance-Index Clamping

The KD path contains behavior equivalent to:

```python
mask_idx = torch.clamp(
    target_gt_idx[i][fg_mask_i],
    0,
    sam_logits.shape[0] - 1
)
```

Out-of-range instance indices are therefore silently mapped to the last teacher instance.

This can hide mismatches rather than exposing them.

## Risk

If student GT instance indexing does not perfectly match the order/count of stored SAM teacher instances, the wrong teacher target can be used without raising an error.

## Required fix

Replace silent clamping with assertions such as:

```python
assert target_gt_idx.max() < sam_logits.shape[0]
```

and store an explicit mapping between:

- source connected component,
- emitted YOLO polygon/instance,
- SAM teacher tensor index.

---

# 8. Critical Bug #3 — Multi-Seed Verification Did Not Change the Training Seed

The project configuration exposes a seed, but `kd_trainer.py` does not forward it in the Ultralytics overrides.

The executed log shows:

```text
seed = 0
```

even when notebook naming or banners refer to other seeds such as 42 or 123.

## Consequence

The claimed:

> multi-seed verification

is not actually a multi-seed experiment.

`02_seed123` also lacks a committed executed output.

## Required fix

Forward the seed explicitly into Ultralytics and verify from runtime logs.

---

# 9. Critical Bug #4 — LayerKD Uses the Wrong Layers

The manuscript claims LayerKD at strides:

- P3 / stride 8,
- P4 / stride 16,
- P5 / stride 32.

The executed logs instead show the selected hooks correspond to:

- layer 12 → stride 16,
- layer 15 → stride 8,
- layer 18 → stride 16.

The audit identifies the true final P3/P4/P5-style feature layers feeding the segmentation head as different layer indices.

## Consequences

- the highest configured weight is not assigned to the intended finest-scale feature,
- two hooked layers correspond to stride-16 spatial resolution,
- two layers are matched to the same teacher tensor under the current teacher-feature selection rule,
- the claimed 8/16/32 multi-scale mechanism was not actually implemented.

### Interpretation

The `06_layerkd` result is still a measured run, but the mechanism described in the paper is incorrect.

---

# 10. LayerKD Loss Magnitude Is Poorly Controlled

The CWD implementation uses a KL reduction that effectively scales with channel count.

Observed first-batch loss magnitudes include approximately:

```text
mask_kd: 3.16
feature: 25.12
```

before the final weighting.

The feature-loss weight is also larger than the mask-KD weight.

## Consequence

`06_layerkd` is not a mild supplementary feature-distillation experiment.

It is heavily dominated by feature matching.

This matters because its performance cannot be interpreted as evidence that a modest multi-scale CWD regularizer improves the student.

---

# 11. Missing Teacher Features Are Replaced by Zero Tensors

When a teacher feature is missing, the current implementation substitutes zeros.

After spatial softmax, a zero tensor becomes a uniform probability distribution.

Therefore a missing target does not simply skip the sample.

It actively pushes the student's feature distribution toward uniform activation.

## Required fix

Missing teacher features should cause:

- the feature loss for that sample to be skipped,
- or an explicit failure.

They should not create a synthetic uniform teacher.

---

# 12. Bare Exception Handling Can Disable KD Silently

The KD code contains broad exception handling that can swallow errors and allow training to continue.

This means some KD configurations may fail internally while the overall training run proceeds.

## Scientific consequence

Training completion alone is insufficient evidence that a configured KD term was actually active.

## Required improvement

For every batch or epoch, log:

- active KD terms,
- non-zero loss magnitudes,
- number of valid teacher targets,
- number of skipped targets,
- number of index mismatches,
- number of exceptions.

Unexpected exceptions should terminate the run during research experiments.

---

# 13. Hyperparameter Provenance — Manuscript Narrative Is Incorrect

The repository shows that:

- `τ = 3.7769`
- `α = 0.9612`

came from an **OOD-aware Optuna search**.

This differs from the paper's narrative in which:

- Optuna supposedly selected a lower temperature,
- then a manual high-temperature override was justified by short-horizon bias.

## Further issue

The OOD-aware Optuna objective directly used the uncropped validation-parent evaluation.

Therefore the reported OOD performance is not an untouched validation of the selected hyperparameters.

## Required rewrite

The paper must clearly distinguish:

1. the first short-horizon Optuna experiment,
2. the later OOD-aware Optuna study,
3. which study produced the final τ and α.

The proposed mechanistic story about gradient persistence should be presented as a hypothesis unless it is supported by actual gradient/entropy diagnostics.

---

# 14. `04_affinity` Headline Result Is Not Supported by the Executed Repository Evidence

The committed executed `04_affinity` notebook reports approximately:

```text
Mask mAP50 = 0.5348
Box mAP50  = 0.5831
```

The manuscript reports:

```text
Mask mAP50 = 0.5569
Box mAP50  = 0.5973
```

Later source notebooks already contain the corrected `0.5348` value, while several manuscript/supporting files still retain `0.5569`.

## Important interpretation

The repository evidence does **not** establish where `0.5569` came from.

Therefore the scientifically safe statement is:

> The committed executed artifact supports 0.5348; the provenance of 0.5569 is unresolved.

Avoid accusatory language about fabrication unless intentional manipulation is independently established.

---

# 15. Corrected In-Domain Ranking

Based on the executed logs available to the audit:

| Rank | Variant | Mask mAP50 |
|---:|---|---:|
| 1 | `05_multiscale` | **0.5485** |
| 2 | `10_hires` | 0.5426 |
| 3 | `01_seed42` | 0.5424 |
| 4 | `06_layerkd` | 0.5422 |
| 5 | `09_focal` | 0.5413 |
| 6 | `09_combined` | 0.5409 |
| 7 | `03_dilated` | 0.5387 |
| 8 | `04_affinity` | **0.5348** |

The committed executed evidence therefore does **not** support `04_affinity` as the best in-domain variant.

---

# 16. No-KD Baseline Is Not Properly Established

The manuscript reports a no-KD baseline around:

```text
Mask mAP50 = 0.5400
Box mAP50  = 0.5970
```

However, the repository does not contain a committed executed artifact that reproduces the baseline under the exact same final protocol.

More seriously, the baseline was not included in the OOD/tiled evaluation run.

## Consequence

The paper currently cannot support claims such as:

> `03_dilated` improves OOD mAP50 by +18.7% over the no-KD baseline

because the reported baseline OOD row was not actually measured.

---

# 17. Baseline OOD Row Is a Duplicate, Not a Measurement

The OOD evaluation notebook evaluated seven KD checkpoints.

No baseline checkpoint was part of the OOD evaluation.

Yet the manuscript's baseline OOD row contains the exact same values as `01_seed42`.

Therefore:

> the baseline OOD/tiled row should be removed until a real no-KD checkpoint is evaluated.

This affects all relative OOD improvement claims.

---

# 18. Checkpoint Resolver Bug Is Still Present

The manuscript states that a prior glob-based checkpoint resolver problem was fixed.

Repository evidence shows glob-based discovery remains in evaluation notebooks.

The cross-dataset evaluation also shows multiple intended checkpoint sources collapsing to the same destination path.

## Scientific consequence

Some cross-dataset results reproduce numerically, but the exact identity of the checkpoint used for the "Full KD" condition is not reliably established.

## Required fix

Every evaluation should specify an exact checkpoint path and verify:

```text
checkpoint SHA256
config hash
experiment ID
seed
training dataset
```

No glob fallback should exist in final evaluation code.

---

# 19. Affinity Loss Is Much Smaller Than the Main KD Loss

The audit reports approximately:

```text
mask_kd  = 3.160839
affinity = 0.001256
```

The affinity term is therefore numerically tiny relative to the main KD term.

`01_seed42` and `04_affinity` otherwise share the same observed Mask-KL magnitude.

Yet their final Mask mAP50 differs by approximately:

```text
0.5424 - 0.5348 = 0.0076
```

## Interpretation

This suggests that the observed difference between these runs may be comparable to ordinary training variation or uncontrolled pipeline differences.

It does not support a strong causal claim that the affinity term drives a meaningful improvement.

---

# 20. Corrected Tiled Dice Findings

The executed evaluation supports:

```text
04_affinity tiled Dice = 0.2594
```

The value `0.2683` does not appear in the executed evidence inspected by Opus.

The strongest measured tiled Dice is:

```text
06_layerkd = 0.2747
```

## Consequence

The manuscript recommendation that `04_affinity` is the best full-resolution Dice variant is contradicted by the repository evidence.

---

# 21. Corrected Leaderboard From Executed Logs

| Variant | Mask mAP50 | Mask mAP50-95 | Box mAP50 | Full-res mAP50 | Full-res mAP50-95 | Direct Dice | Tiled Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| `05_multiscale` | **0.5485** | 0.2087 | **0.6001** | 0.0872 | 0.0226 | 0.1570 | 0.2625 |
| `10_hires` | 0.5426 | 0.2002 | 0.5875 | 0.0883 | 0.0197 | — | — |
| `01_seed42` | 0.5424 | 0.2009 | 0.5976 | 0.0848 | 0.0196 | 0.1409 | 0.2414 |
| `06_layerkd` | 0.5422 | 0.2063 | 0.5903 | 0.0944 | **0.0252** | 0.1621 | **0.2747** |
| `09_focal` | 0.5413 | 0.2060 | 0.5847 | 0.0931 | 0.0242 | 0.1445 | 0.2671 |
| `09_combined` | 0.5409 | 0.2080 | 0.5881 | 0.0851 | 0.0200 | 0.1546 | 0.2630 |
| `03_dilated` | 0.5387 | 0.2030 | 0.5819 | **0.1007** | 0.0241 | **0.1722** | 0.2612 |
| `04_affinity` | **0.5348** | 0.2042 | 0.5831 | 0.0831 | 0.0214 | 0.1505 | 0.2594 |
| No-KD baseline | **unverified** | — | — | **not measured** | **not measured** | **not measured** | **not measured** |

## Important caution

These numbers describe the executed runs, but the KD runs are affected by the teacher-alignment bug.

Therefore this table should not be interpreted as a valid ranking of correctly implemented distillation methods.

---

# 22. Tiled Inference Is the Strongest Verified Result

The previous audit questioned whether the direct-vs-tiled improvement might be caused by threshold mismatch.

Repository inspection refutes that concern.

Both direct and tiled Dice use the same mask threshold:

```text
0.35
```

Across seven evaluated checkpoints, tiled inference improves Dice by approximately:

```text
+51.7% to +84.8%
```

This broadly supports the manuscript's stated `+52–83%` range.

## Why this result is unusually strong

It:

- reproduces from the executed evaluator,
- does not depend on the no-KD baseline,
- does not depend on the disputed `04_affinity` score,
- does not depend on the intended KD mechanism being correct,
- directly targets the known resolution-loss problem.

### Publication implication

This is currently the strongest verified result in CrackDistill.

---

# 23. Full-Resolution Deployment Claim Is Overstated

The repository benchmark measures approximately:

```text
9.27 ms
107.8 forward passes / second
```

using:

- batch size 1,
- `512×512`,
- synthetic tensor input,
- model forward timing,
- GPU synchronization.

It does not include:

- decoding,
- tiling,
- H2D transfer,
- mask decoding,
- NMS,
- stitching,
- Gaussian blending,
- output generation.

The tiled pipeline uses approximately 20 tiles per 2000×1500 image.

A serial lower-bound estimate is therefore:

```text
20 × 9.27 ms = 185.4 ms
≈ 5.39 full scenes / second
```

This is an estimate, not a measured end-to-end throughput.

## Important opportunity

The current tiler processes tiles one at a time.

Batching the tiles could materially improve throughput.

---

# 24. What Earlier Concerns Were Refuted

The repository investigation provides important positive findings.

## Refuted: Bernoulli KL may be implemented incorrectly

The Mask-KL implementation contains:

- both Bernoulli terms,
- temperature applied to teacher and student,
- `τ²` scaling,
- log-stable formulation,
- clamping / epsilon protections.

### Verdict

> **Implementation is technically sound.**

---

## Refuted: Crack500 parent-scene leakage

Official split folders are preserved.

### Verdict

> No evidence of train/val/test parent leakage.

---

## Refuted: Mosaic composites may mix evaluation scenes into training

Reconstructed mosaics are built from training crop stems.

### Verdict

> The mosaic reconstruction itself is train-only.

---

## Refuted: Teacher logits are only one tensor per image

Each `.npy` can contain:

```text
(M, 256, 256)
```

where `M` is the number of teacher instances.

### Verdict

> Per-instance teacher targets genuinely exist.

---

## Refuted: Full-resolution ground truth is incomplete stitched crop masks

The uncropped conversion uses the original full-scene mask files.

### Verdict

> The evaluation masks are complete original annotations.

---

## Refuted: Tiled Dice gain is caused by different thresholds

Both direct and tiled paths use the same mask threshold.

### Verdict

> The tiled gain is real under the implemented evaluator.

---

## Refuted: The 160↔256 student/teacher resolution mismatch is undocumented

The KD code explicitly resizes teacher and student masks to a common target shape.

### Verdict

> This part of the KD implementation is deliberate.

---

# 25. Remaining Unresolved Questions

Despite the strong repository audit, several important questions remain unresolved because datasets, caches, and checkpoints are not committed.

## 25.1 Teacher quality

Cannot yet measure:

- teacher Dice,
- teacher clDice,
- teacher boundary quality,
- calibration,
- performance by crack width,
- failure rate by morphology.

Requires:

- cached logits,
- ground-truth masks.

---

## 25.2 Exact baseline provenance

The exact artifact corresponding to:

```text
Mask mAP50 = 0.5400
Box mAP50  = 0.5970
```

is not available in the committed executed evidence.

Requires:

- original baseline checkpoint,
- original validation output.

---

## 25.3 Exact provenance of `0.5569`

The committed executed `04_affinity` result is `0.5348`.

The earlier `0.5569` number may originate from an earlier uncommitted experiment, but this is not established.

Requires:

- original log/notebook/checkpoint that produced `0.5569`.

---

## 25.4 DeepCrack teacher-logit usage

The teacher-logit count suggests DeepCrack logits may coexist in the same library.

Presence alone does not imply training contamination.

Requires:

- actual teacher-logit filenames,
- batch stem trace during Crack500 training.

---

## 25.5 Topology and boundary metrics

Cannot yet compute:

- clDice,
- semantic Dice/IoU from raw prediction dumps,
- boundary F,
- HD95,
- false-positive density,
- recall by crack width.

Requires:

- data,
- checkpoints or prediction dumps.

---

# 26. Scientific Claim Audit

| Claim | Current verdict | Strongest defensible wording |
|---|---|---|
| SAM transfers structural priors | **Unsupported** | Current KD targets were geometrically misaligned under augmentation; prior-transfer claims require rerunning with corrected alignment |
| Mask-KL is the essential driver | **Unsupported** | Mask-KL is the dominant auxiliary-loss term numerically, but causal benefit is not established |
| `04_affinity` is best in-domain | **Contradicted** | The committed executed `04_affinity` result is 0.5348, the lowest of the eight KD variants |
| `03_dilated` improves OOD by +18.7% vs baseline | **Unsupported as a baseline-relative claim** | `03_dilated` scored 0.1007 on 50 uncropped Crack500 validation parents; no no-KD baseline was evaluated on this protocol |
| LayerKD improves fine-grained localization | **Partially supported empirically, mechanism incorrect** | `06_layerkd` has the strongest measured full-res mAP50-95/tiled Dice, but layer selection and weighting differ from the manuscript |
| SAM priors are domain-agnostic | **Unsupported** | Cross-dataset gains are small and exact checkpoint identity remains uncertain |
| Tiled inference recovers ~52–83% Dice | **Supported** | Across seven checkpoints, tiled inference improves Dice by ~51.7–84.8% at the same threshold |
| >100 FPS deployment | **Contradicted as a full-pipeline claim** | Model-only 512×512 forward time is ~9.27 ms; end-to-end tiled throughput has not been measured |
| Zero SAM runtime dependency | **Supported** | Teacher supervision is offline and training-only |
| 79× compression | **Partially supported wording** | Teacher/student parameter ratio is approximately 79× |
| Teacher probabilities are calibrated | **Unsupported** | Calibration was not measured |
| Sub-pixel/hairline topology is improved | **Unsupported** | No crack-width or topology-sensitive metric was evaluated |
| Short-horizon HPO bias is demonstrated | **Unsupported / contradicted by provenance** | Final τ and α came from an OOD-aware Optuna search, not the manual override narrative |
| Mosaic augmentation improves OOD | **Unresolved** | The relevant committed mosaic-KD run failed |
| Cross-dataset transfer is zero-shot | **Partially supported** | Held-out evaluation splits are correct, but exact "Full KD" checkpoint identity is not fully established |

---

# 27. Highest-Information Next Actions

These should be prioritized by **information gain per GPU-hour**, not novelty.

## Action 1 — Run an alignment diagnostic

Estimated cost: very low.

For approximately 200 training batches:

- save augmented student image,
- transformed GT,
- teacher mask actually selected,
- matched instance IDs,
- `mask_idx`,
- whether any index was clamped.

Compute teacher-vs-GT support IoU.

### Goal

Confirm quantitatively how severe Bug #1 is.

---

## Action 2 — Correct the manuscript immediately

No GPU required.

At minimum:

- `04_affinity` Mask mAP50: `0.5569 → 0.5348`
- `04_affinity` Box mAP50: `0.5973 → 0.5831`
- tiled Dice: `0.2683 → 0.2594`
- remove the duplicated baseline OOD row,
- remove all baseline-relative OOD percentages,
- correct actual image size (`512`),
- correct actual optimizer behavior,
- correct actual seed,
- correct tile stride/overlap,
- correct LayerKD layer/stride description,
- correct HPO provenance.

---

## Action 3 — Establish a real matched no-KD baseline

Train no-KD using the exact same:

- Ultralytics version,
- seed,
- image size,
- optimizer,
- augmentation,
- epoch count,
- batch size,
- AMP setting,
- evaluation path

as the corrected KD experiments.

Then evaluate:

- 348 val crops for development,
- 1,124 test crops for confirmation,
- uncropped validation/test parents,
- tiled Dice.

---

## Action 4 — Fix KD target alignment and rerun the minimum pair

First fix:

- teacher geometry,
- instance ID mapping,
- silent clamping,
- seed forwarding.

Then rerun:

```text
no-KD
Mask-KL
```

before testing affinity, dilation, LayerKD, or new losses.

### Why

The first question is simply:

> Does correctly aligned SAM distillation produce any reproducible benefit?

---

## Action 5 — Run a real seed study

After the pipeline is fixed:

- at least 5 seeds for no-KD,
- at least 5 seeds for Mask-KL.

Only after an effect exceeds run-to-run uncertainty should more KD variants be compared.

---

## Action 6 — Compare SAM against GT-derived soft supervision

Run:

```text
A. no-KD
B. SAM Mask-KL
C. Gaussian-soft GT
D. signed-distance soft GT
E. morphology-derived soft GT
F. shuffled/corrupted SAM control
```

This experiment determines whether SAM contributes anything beyond smooth geometry derived directly from annotations.

---

## Action 7 — Evaluate on the untouched test set

Use the 1,124-image canonical Crack500 test split after the method is frozen.

For full-resolution evaluation, use the uncropped **test parents**, not the validation parents used during optimization.

---

## Action 8 — Rebuild LayerKD correctly

If LayerKD remains of interest:

- identify the actual P3/P4/P5 feature layers,
- use correct stride assignments,
- normalize CWD so its scale is controlled,
- skip samples with missing teacher features,
- preferably test P3-only first.

---

## Action 9 — Benchmark true end-to-end deployment

Measure:

```text
decode
→ tile extraction
→ preprocessing
→ batched inference
→ mask reconstruction
→ NMS
→ Gaussian merge
→ final output
```

Report:

- p50,
- p95,
- full-scene FPS,
- VRAM,
- batch size,
- tile count.

Also benchmark batched tile inference.

---

# 28. Recommended Publication Decision Tree

```text
START
│
├─ Are manuscript numbers consistent with committed executed evidence?
│    └─ NO
│       → Correct 04_affinity, tiled Dice, baseline OOD row,
│         optimizer/seed/imgsz/stride descriptions immediately.
│
├─ Is the no-KD baseline measured under the same protocol?
│    └─ NO
│       → Train/evaluate matched baseline.
│
├─ Are SAM teacher targets geometrically aligned with the student?
│    └─ NO
│       → Fix KD pipeline.
│       → Existing KD runs cannot support causal KD claims.
│
├─ After fixing alignment, does SAM-KD beat no-KD across seeds?
│    ├─ NO
│    │   → Distillation contribution is not established.
│    │   → Reframe around tiled high-resolution inference / negative result.
│    │
│    └─ YES
│        ↓
│
├─ Does SAM-KD beat GT-derived soft-target controls?
│    ├─ NO
│    │   → Reframe around structured soft supervision.
│    │
│    └─ YES
│        → Foundation-model prior-transfer claim becomes defensible.
│
├─ Does the effect survive the untouched 1,124-image test set?
│    ├─ NO
│    │   → Treat as development overfitting.
│    │
│    └─ YES
│        → Stronger publication-quality evidence.
│
└─ Deployment
     → Replace model-call FPS with measured full-scene p50/p95 throughput.
```

---

# 29. Recommended Paper Reframing Right Now

Before any retraining, the most defensible paper narrative is not:

> SAM 2 distillation reliably improves a nano YOLO student.

That claim is not supported by the current implementation.

A stronger evidence-based interim narrative is:

> Full-resolution crack inspection is dominated by scale loss when megapixel pavement imagery is directly resized to compact-model input resolution. A tiled Gaussian-apodized inference strategy substantially restores segmentation quality without changing the trained student architecture.

This is currently the result that survives the audit most cleanly.

If corrected SAM-KD later outperforms both:

- no-KD,
- and GT-derived soft-target supervision,

then the broader foundation-model distillation contribution can be restored.

---

# 30. Final Assessment

The repository audit does **not** imply that CrackDistill should be abandoned.

It implies that the project now has a clear experimental fork.

## What is currently real

The strongest verified results are:

- the canonical Crack500 split infrastructure exists,
- parent-scene split leakage was not found,
- teacher targets are genuinely stored per instance,
- the Bernoulli KL implementation is correct,
- the uncropped labels are complete,
- tiled Gaussian-apodized inference produces a large reproducible Dice improvement,
- no SAM teacher is required at deployment.

## What is currently unproven

The project has not yet established that:

- SAM structural priors improve the student,
- affinity KD improves in-domain performance,
- dilated KD improves over a true no-KD OOD baseline,
- LayerKD works for the mechanism described,
- the improvement exceeds seed variance,
- SAM adds information beyond deterministic soft GT,
- the method generalizes to a genuinely external domain,
- the tiled deployment pipeline runs at >100 full scenes/s.

## Most important scientific next question

> **After correcting target alignment and running a matched baseline, does properly implemented SAM supervision produce a reproducible improvement beyond both run-to-run variance and GT-derived soft supervision?**

That should now be treated as the central confirmatory experiment for the CrackDistill research program.

---

# 31. Recommended Immediate Order of Work

1. **Correct manuscript numbers and descriptions.**
2. **Run the teacher/GT alignment diagnostic.**
3. **Fix teacher geometric transforms and seed forwarding.**
4. **Train a true matched no-KD baseline.**
5. **Rerun no-KD vs Mask-KL at multiple seeds.**
6. **Evaluate on the untouched 1,124-image test split.**
7. **Run the SAM-vs-soft-GT control.**
8. **Only then revisit affinity/dilation/LayerKD.**
9. **Benchmark end-to-end batched tiled inference.**
10. **Reframe the paper around whichever contribution survives confirmatory testing.**

---

## Bottom Line

> **The repository evidence currently supports the tiled full-resolution inference contribution much more strongly than the SAM-to-YOLO knowledge-distillation contribution.**

The KD story may still become publishable, but it now requires a corrected training pipeline and a clean confirmatory study rather than additional exploratory KD variants.
