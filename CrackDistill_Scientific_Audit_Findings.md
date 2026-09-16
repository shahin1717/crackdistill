# CrackDistill Scientific Audit Review
## Synthesis of Opus Findings + Independent Assessment

**Project:** CrackDistill  
**Purpose:** Consolidate the major scientific, methodological, evaluation, and deployment findings raised by the Opus audit, while separating hard contradictions from plausible-but-unverified hypotheses.

---

# Executive Summary

The Opus response is highly valuable because it does more than suggest new loss functions: it performs a pre-publication forensic audit of the current CrackDistill evidence.

The key conclusion is:

> **Do not discard CrackDistill, but do not yet trust the headline scientific claims.**

Several issues are likely straightforward to fix: inconsistent manuscript numbers, undefined evaluation details, inconsistent tables, missing provenance, and evaluator reproducibility.

Other issues could materially change the scientific interpretation of the work:

- whether the canonical Crack500 test set was ever used,
- whether the OOD dataset is valid and fully annotated,
- whether cached SAM supervision is geometrically aligned with augmented student inputs,
- whether the same checkpoint was accidentally reused in some OOD evaluations,
- whether SAM contributes anything beyond ordinary softened ground truth,
- whether the reported gains exceed run-to-run variance,
- whether Mask mAP50 is an appropriate primary endpoint for extremely thin crack structures,
- and whether the claimed deployment speed reflects the actual tiled full-resolution pipeline.

The highest-value next step is **not another KD variant**.

The correct order is:

1. Validate data splits and provenance.
2. Freeze and rebuild evaluation.
3. Audit augmentation alignment.
4. Measure teacher quality.
5. Compare SAM supervision against GT-derived soft targets.
6. Run statistically meaningful multi-seed experiments.
7. Re-evaluate with topology- and boundary-sensitive metrics.
8. Test true external generalization.
9. Measure end-to-end deployment latency.
10. Only then optimize KD formulations.

---

# 1. Overall Assessment of the Opus Audit

The Opus audit is strong and should materially change the order of work on CrackDistill.

However, not every Opus statement should be accepted as established fact.

The findings fall into three categories:

## Category A — Hard internal contradictions

These can be checked directly from the manuscript and should be treated seriously.

Examples:

- Crack500 split is reported incorrectly.
- Nominally identical recipes report inconsistent scores.
- Tiled Dice values differ across tables.
- The text says dropping Boundary BCE improves performance although the displayed numbers show the opposite.
- Some leaderboard recommendations contradict the leaderboard itself.
- OOD evaluation is insufficiently defined.
- No confidence intervals or meaningful run-to-run variance are reported.

These findings are strong.

## Category B — Strong but unverified hypotheses

These are scientifically important but require code, manifests, checkpoints, or prediction artifacts.

Examples:

- cached SAM logits may be misaligned after augmentation,
- OOD ground truth may be incomplete,
- the same checkpoint may have been reused accidentally,
- the DeepCrack logits may have contaminated a supposedly zero-shot condition,
- teacher targets may be stored at the wrong granularity,
- SAM teacher quality may be weak or poorly calibrated.

These must be tested rather than assumed.

## Category C — Opus overreach or technical overstatement

Some Opus statements are directionally useful but too categorical.

Examples:

- identical rounded OOD scores are highly suspicious but do not prove checkpoint collision,
- GT-derived box prompts are privileged/label-conditioned supervision, not conventional train/test leakage,
- estimating 5.4 FPS from 20 serial tiles is useful but is not a measured full-pipeline throughput,
- YOLO prototype resolution does not prove a bug without inspecting the actual resize path,
- SAM calibration is unmeasured; that is stronger and safer than declaring it necessarily miscalibrated,
- MixUp is not on by default in current Ultralytics settings,
- cross-paper FPS comparisons such as HrSegNet vs YOLO are not fair unless re-run under matched hardware and protocol.

---

# 2. High-Priority Findings

| Finding | Assessment | Priority |
|---|---|---|
| Crack500 split is wrong in the manuscript | Confirmed, serious | Critical |
| OOD dataset is not properly defined | Confirmed, serious | Critical |
| `0.5500` vs `0.5424` for nominally same recipe | Real contradiction | Critical |
| `04_affinity` tiled Dice differs across tables | Real contradiction | Critical |
| “Dropping BCE improves” contradicts displayed numbers | Confirmed | Critical |
| No seed variance / confidence intervals | Major scientific weakness | Critical |
| Baseline and `01_seed42` have identical OOD values | Extremely suspicious, not proof of collision | Critical |
| Cached teacher target alignment under augmentation | Potentially catastrophic if wrong | Critical |
| Teacher quality never measured against GT | Major missing control | Critical |
| SAM vs GT-derived soft-target control missing | Possibly the most important scientific experiment | Critical |
| Instance mAP may be poorly matched to thin topology | Strong concern, but “degenerate” is too strong | High |
| Full tiled deployment may be far slower than 107.8 FPS | Important, but exact 5.4 FPS is only an estimate | High |
| GT box prompts are privileged information | Correct framing; “leakage” is too strong | High |
| Student output must be 160×160 | Requires code inspection | High |
| SAM is necessarily miscalibrated | Too strong; calibration is simply unmeasured | Medium |
| Mosaic and MixUp are both default-on | Incorrect: Mosaic yes, MixUp no | Medium |
| HrSegNet proves YOLO is inferior | Too strong; matched comparison is still needed | High |

---

# 3. Stop-Work Issue 1 — Reconstruct the Actual Crack500 Protocol

The manuscript reports Crack500 as:

- 1,896 train
- 348 validation
- 348 test

The canonical Crack500 protocol is commonly documented as:

- 1,896 train
- 348 validation
- 1,124 test

This is not a cosmetic issue.

The immediate question is:

> Was `348 test` only a manuscript typo, or were the reported final numbers actually computed on the 348-image validation/development set?

These two scenarios have very different implications.

## If it is only a writing mistake

Then the problem is easy:

- correct the paper,
- show the actual manifest,
- prove that the correct 1,124-image test set was used,
- regenerate all tables.

## If the 348-image development set was used for:

- hyperparameter selection,
- Optuna,
- loss selection,
- architecture selection,
- threshold tuning,
- variant selection,
- and final headline reporting,

then it should **not be presented as an untouched test result**.

That is still recoverable.

The correct action would be:

1. rename it clearly as the development/validation set,
2. freeze the entire pipeline,
3. evaluate once on the canonical 1,124-image test set,
4. do not make further model choices based on that test set.

## Required artifacts

- `data.yaml`
- exact train manifest
- exact validation manifest
- exact test manifest
- mapping from patches to parent scene IDs
- hash of each manifest
- assertion that parent scenes do not cross splits

---

# 4. Stop-Work Issue 2 — Define the OOD Dataset

The manuscript relies heavily on results reported on:

> “uncropped 2000×1500 road photos”

Yet the evaluation set is not sufficiently defined.

For a scientifically defensible OOD result, the paper should state:

- number of images,
- source,
- acquisition device,
- geographic/environmental origin,
- relation to Crack500 parent scenes,
- annotation procedure,
- whether masks are complete,
- whether all crack pixels are labeled,
- whether evaluation images overlap training parents,
- whether thresholds were tuned on these images,
- whether the shift is a true domain shift, a scale shift, a context shift, or a combination.

Until that is resolved, terms such as:

- “OOD generalization,”
- “domain-agnostic regularizer,”
- “uncropped full-resolution robustness”

should be used cautiously.

## Important distinction

Opus proposes a worrying hypothesis:

> the OOD labels may have been reconstructed from crop masks and may therefore omit thin cracks that fell outside the retained Crack500 crop criteria.

That is a **plausible failure mode**, but it is not proven from the manuscript alone.

The scientifically correct question is:

> **Are the OOD annotations complete, scene-disjoint, and independent?**

Do not jump directly to:

> “The OOD evaluation is broken.”

That requires evidence.

---

# 5. Stop-Work Issue 3 — Rebuild Evaluation From Immutable Checkpoint Records

One of the strongest warning signs in the manuscript is that:

- the No-KD baseline,
- and `01_seed42`

have different in-domain results, yet share identical reported OOD/tiled values to four decimals.

This is especially concerning because the project already documents a historical checkpoint resolver collision.

That does **not prove** the same bug recurred.

However, it is strong enough that all OOD results should be regenerated from an evaluator that never guesses which checkpoint to load.

## Recommended result ledger

Every evaluation row should carry:

```text
variant_name
git_commit
config_hash
checkpoint_sha256
seed
train_manifest_hash
eval_manifest_hash
image_size
confidence_threshold
mask_threshold
NMS_IoU
tile_size
tile_stride
tile_overlap
Gaussian_sigma
metric_version
prediction_dump_path
evaluation_timestamp
```

Then generate the paper tables automatically from structured JSON/CSV.

## Rule

> Never manually copy a metric from a notebook into the manuscript again.

The paper should be regenerated from the frozen result ledger.

---

# 6. Stop-Work Issue 4 — Audit Teacher/Student Geometric Alignment

This may be the most dangerous implementation-level risk in the project.

SAM teacher logits are cached offline.

YOLO training can apply geometric transformations such as:

- letterboxing,
- flipping,
- scaling,
- translation,
- mosaic,
- cropping,
- perspective transformations depending on configuration.

If the image and ground-truth masks are transformed, but the cached SAM target is not transformed by the exact same mapping, the student receives spatially incorrect supervision.

In that failure mode, the KD loss is effectively saying:

> “imitate SAM here”

when SAM’s corresponding prediction came from a different physical pixel.

That could easily produce:

- small inconsistent gains,
- unstable loss behavior,
- suppressed improvements,
- apparently useful regularization without meaningful teacher transfer.

## Immediate test

Sample approximately 200 training batches.

For each positive instance, visualize:

1. augmented input image,
2. transformed GT mask,
3. SAM target actually used by the KD loss,
4. student mask support,
5. teacher/student overlap.

Compute an alignment metric such as IoU between:

- the transformed GT support,
- and the teacher support after all transforms.

If alignment is poor, the current KD results cannot be interpreted.

## Important correction to Opus

Opus stated that both Mosaic and MixUp are on by default.

That is not correct for current Ultralytics defaults:

- Mosaic: typically enabled by default.
- MixUp: typically disabled by default.

The geometric-alignment concern remains valid because Mosaic alone is sufficient to make cached supervision non-trivial.

---

# 7. Stop-Work Issue 5 — Does SAM Contribute Anything Beyond Softened Ground Truth?

This is arguably the most important scientific control in the entire project.

The teacher receives a box prompt derived from the ground-truth crack mask.

That means SAM is not operating as an independent annotation-free teacher.

It receives privileged training-time information.

This is not necessarily improper.

It simply changes the scientific question.

The critical question becomes:

> **Does SAM add structural knowledge that cannot be reproduced by a deterministic transformation of the ground truth?**

Without this control, an improvement from SAM KD may simply arise because soft labels are easier to optimize than binary labels.

## Essential control experiment

Run the following arms under identical training budgets.

### A. Standard baseline

```text
L_task
```

### B. SAM teacher

```text
L_task + SAM soft-mask KD
```

### C. Gaussian-smoothed GT

Create a soft target by Gaussian blurring the binary mask.

```text
L_task + soft-GT KD
```

### D. Signed Distance Transform target

Create a distance-aware continuous crack target:

```text
P_soft = sigmoid(-SDT(mask) / temperature)
```

### E. Morphological boundary uncertainty target

Build a soft boundary corridor using:

- dilation,
- erosion,
- distance-to-boundary,
- or other deterministic label-derived uncertainty.

## Interpretation

If:

```text
SAM >> Gaussian / SDT / morphology
```

then you have strong evidence that SAM contributes foundation-model structure beyond simple label smoothing.

If:

```text
SAM ≈ Gaussian / SDT / morphology
```

then the scientific contribution changes.

The result may still be interesting:

> Structured soft supervision improves tiny crack segmentation.

But the claim:

> SAM 2 transfers uniquely valuable foundation-model priors

would not yet be established.

This one experiment can determine what the paper is actually about.

---

# 8. Same-Recipe Reproducibility Problem

The manuscript reports approximately:

- Mask-KL-only result: `0.5500`
- supposedly locked `01_seed42` recipe: `0.5424`

If these runs truly share:

- architecture,
- seed,
- optimizer,
- epoch count,
- temperature,
- alpha,
- dataset,
- resolution,
- augmentations,

then a gap of `0.0076` is substantial relative to the reported improvements.

This raises two possibilities.

## Possibility A

The recipes are not actually identical.

Possible hidden differences:

- α changed,
- augmentation changed,
- checkpoint initialization changed,
- scheduler changed,
- resolution changed,
- logits changed,
- bug fix changed,
- evaluator changed,
- preprocessing changed.

## Possibility B

The pipeline has meaningful run-to-run instability.

Either way, the current single-seed conclusions are too strong.

## Required action

1. Diff configs field by field.
2. Record exact checkpoint hashes.
3. Run at least 5 seeds for key arms.

Minimum key arms:

- no-KD,
- Mask-KL,
- affinity,
- dilated.

Report:

- mean,
- standard deviation,
- 95% CI,
- paired per-image deltas where possible,
- bootstrap intervals clustered by parent scene.

---

# 9. Boundary-BCE Interpretation Is Incorrect

The manuscript states that dropping Boundary BCE improves performance.

But the displayed numbers cited by Opus show:

```text
0.5460 < 0.5470
```

So in that specific comparison, dropping BCE does not improve the metric.

More importantly, the component effects appear to interact.

The stronger conclusion is not:

> “Boundary BCE hurts.”

It is:

> The available factorial cells suggest interaction effects, while the observed deltas are too small relative to reproducibility uncertainty to support a clean single-component causal claim.

## Recommended wording

Replace claims like:

> Feature MSE and Boundary BCE over-constrain the student.

with:

> In the current exploratory ablation, jointly removing Feature MSE and Boundary BCE yielded the strongest single-seed Mask-KL configuration. However, component-level effects were not stable across conditions and require multi-seed factorial confirmation.

---

# 10. Tiled Dice Inconsistency

The same `04_affinity` configuration is reported with different tiled Dice values in different parts of the manuscript.

Additionally, one table reportedly identifies `06_layerkd` as the strongest tiled-Dice result while another recommendation table presents `04_affinity` as the best full-resolution Dice model.

This should be treated as a data-generation problem, not just a writing typo, until proven otherwise.

## Required action

Regenerate:

- all leaderboard rows,
- all recommendation tables,
- abstract headline metrics,
- conclusion metrics,

from a single frozen result source.

---

# 11. Metric-Task Mismatch

Crack500 is fundamentally distributed as semantic crack masks.

The project converts connected components into YOLO instances.

That is allowed, but it introduces an artificial instance definition.

For thin crack networks, a one-pixel change can drastically affect instance IoU.

That makes Mask mAP50 a potentially unstable or incomplete measure of the actual engineering objective.

## Do not necessarily remove mAP

Mask mAP50 should remain because:

- YOLO is an instance-segmentation model,
- it helps compare variants within the current framework,
- it provides continuity with the current experiments.

But it should no longer be the only or dominant endpoint.

## Add metrics that directly measure the stated failure modes

Recommended metrics:

- semantic Dice,
- semantic IoU,
- clDice / centerline Dice,
- boundary F-score at multiple tolerances,
- Hausdorff distance or robust HD95,
- connectivity error,
- connected-component count error,
- skeleton precision,
- skeleton recall,
- recall by crack width,
- false-positive pixels per megapixel,
- false-positive connected components per megapixel,
- crack-free tile precision.

## Why this matters

The paper claims to improve:

- hairline preservation,
- structural continuity,
- topology,
- boundary quality,
- false positives on gravel and stains.

Those should be measured directly.

---

# 12. Teacher Quality Has Not Been Established

The paper distills from SAM 2 but does not first establish how good SAM 2 is on the crack masks being used.

Before discussing “dark knowledge,” measure the teacher itself.

## Teacher audit

For the SAM outputs used during training, compute:

- Dice vs GT,
- IoU vs GT,
- clDice,
- boundary F,
- calibration error if probability claims are retained,
- Brier score,
- teacher performance stratified by crack width,
- teacher performance by morphology,
- teacher performance on branches/intersections,
- failure rate by prompt geometry.

Also compare:

- box prompt,
- point prompt,
- skeleton-sampled multipoint,
- iterative mask refinement if considered later.

## Important language correction

Do not call the SAM maps:

> “spatially calibrated probabilities”

unless calibration is actually measured.

A safer description is:

> soft logits with spatially varying confidence near boundaries.

---

# 13. GT-Conditioned Prompts: Correct Scientific Framing

Opus calls the GT box prompt “label leakage.”

That terminology is too strong.

The teacher receives a tight box derived from the training annotation.

This is better described as:

- privileged training information,
- label-conditioned teacher prompting,
- GT-conditioned teacher refinement,
- training-only supervision.

It is not conventional train/test leakage if test annotations are never used for training.

However, it strengthens the need for the GT-soft-target control.

The scientific question becomes:

> What does SAM contribute beyond the annotation from which its prompt was derived?

---

# 14. Output Resolution / Tensor Shape Claims Need Code Verification

The manuscript describes SAM teacher logits and student mask logits in a common spatial shape.

Opus argues that the standard YOLO segmentation prototype map would be closer to input/4 resolution.

That is a valid reason to inspect the implementation, but it is not enough to declare the manuscript incorrect by itself.

The code may explicitly:

- resize student logits,
- resize teacher logits,
- crop to matched instance boxes,
- interpolate both to another grid,
- or reconstruct masks at full input size before KD.

## Required action

Document the literal tensor-shape trace:

```text
raw image
→ letterboxed image
→ YOLO feature maps
→ mask prototypes
→ mask coefficients
→ reconstructed student mask logits
→ resized KD mask
→ teacher logits
→ matched instance KD target
→ loss
```

For every tensor, record:

```text
[B, C, H, W]
dtype
coordinate system
interpolation method
normalization
activation state
```

This will eliminate ambiguity.

---

# 15. OOD Claims Should Separate Scale Shift From Domain Shift

A model trained on 640×360 crops and evaluated on 2000×1500 uncropped scenes faces multiple changes simultaneously:

- resolution,
- context,
- crack pixel width,
- scene composition,
- negative background prevalence,
- possibly camera and environment,
- possibly geographic domain.

Those should not all be called “domain shift” automatically.

## Recommended decomposition

Evaluate four conditions if possible:

### A. Same-domain / same-scale

Canonical Crack500 patches.

### B. Same-domain / changed-scale

Parent scenes from the same source distribution, if legally and scientifically available.

### C. External-domain / matched-scale

External datasets resized/cropped to similar scale.

### D. External-domain / full-resolution

True external full scenes.

This lets you distinguish:

- resolution robustness,
- contextual robustness,
- true cross-domain robustness.

---

# 16. Cross-Dataset Transfer Claims Need Stronger Controls

The manuscript reports relative gains such as approximately +19%.

But a movement from a very low absolute baseline to another very low absolute score can produce a large relative percentage without practical significance.

Always report:

1. absolute change,
2. confidence interval,
3. relative change second.

Example:

```text
0.0276 → 0.0329
absolute Δ = +0.0053
relative Δ ≈ +19%
```

The absolute change is the more informative number.

## Stronger design

For “domain-agnostic” claims, use more than two datasets.

A stronger study would include:

- Crack500,
- DeepCrack,
- CrackSeg9k constituents,
- NHA12D or another genuinely external source,
- leave-one-domain-out evaluation.

Then evaluate whether SAM KD produces a consistent gain across domains.

---

# 17. Deployment Claim Needs End-to-End Measurement

The reported `107.8 FPS` appears to correspond to approximately:

```text
1000 / 9.27 ms
```

That is likely per network forward pass.

However, full-resolution tiled inference requires multiple forward passes per survey image.

Therefore:

> 107.8 model invocations per second is not the same as 107.8 full-resolution road images per second.

Opus estimates roughly 20 tiles for a 2000×1500 image with the stated tiling configuration.

Serially multiplying:

```text
20 × 9.27 ms
```

gives a useful lower-bound model-time estimate.

But it is not the final pipeline throughput because:

- tiles may be batched,
- GPU utilization changes with batch size,
- pre/postprocessing has non-zero cost,
- memory transfer matters,
- NMS/mask reconstruction matters,
- tile stitching matters.

## Required benchmark

Measure end-to-end:

```text
image decode
→ tiling
→ preprocessing
→ H2D transfer
→ batched network inference
→ mask reconstruction
→ confidence/NMS
→ tile merge
→ Gaussian blending
→ final mask
```

Report:

- p50 latency,
- p95 latency,
- full-scene FPS,
- GPU,
- precision mode,
- batch size,
- image size,
- tile configuration.

The paper can still report the 9.27 ms network invocation time, but it must be clearly labeled as model-only latency.

---

# 18. Lightweight Semantic Baseline Is Missing

A crack-specific semantic segmentation model such as HrSegNet is a legitimate missing comparison.

However, published FPS/mIoU from another paper should not be directly compared with CrackDistill numbers because:

- hardware differs,
- datasets differ,
- input sizes differ,
- software stacks differ,
- task definitions differ.

The correct experiment is matched.

## Matched baseline experiment

Train a strong compact semantic crack segmenter using:

- the exact same Crack500 train split,
- same augmentation budget,
- same test set,
- same input resolution policy,
- same GPU,
- same precision,
- same deployment benchmark.

Then compare:

- semantic Dice,
- clDice,
- boundary F,
- full-resolution speed,
- parameter count,
- FLOPs,
- memory,
- false positives.

If the semantic model wins decisively, the student architecture choice needs rethinking.

If the distilled YOLO student remains competitive, the architecture claim becomes stronger.

---

# 19. Recommended Primary Scientific Question

The current paper implicitly asks:

> Which KD variant produces the highest mAP?

A stronger research question is:

> **Does a foundation segmentation model transfer structural information that cannot be obtained from ordinary labels, and does that information measurably improve topology and cross-scale robustness in a tiny deployable student?**

This reframing makes the experimental logic much stronger.

It naturally prioritizes:

1. valid data,
2. valid evaluator,
3. teacher quality,
4. GT-soft controls,
5. reproducibility,
6. topology metrics,
7. external generalization,
8. deployment.

Affinity, dilation, CWD, multi-scale KD, ensemble prompting, and other advanced techniques come after that foundation.

---

# 20. Recommended Experiment Sequence

## Phase 0 — Zero/low-GPU forensic audit

### EX-01: Dataset and split audit

Verify:

- canonical Crack500 split,
- exact manifests,
- parent-scene disjointness,
- OOD provenance,
- mosaic provenance,
- no train/eval parent overlap.

### EX-02: Frozen evaluator

Regenerate all tables from explicit checkpoints.

### EX-03: OOD label completeness audit

Determine whether correct thin-crack predictions can be unfairly penalized.

### EX-04: Augmentation alignment audit

Visualize teacher targets after the exact student transformation path.

### EX-05: End-to-end deployment benchmark

Measure actual full-scene latency.

---

## Phase 1 — Establish whether the KD effect is real

Run 5 seeds for:

- no-KD,
- Mask-KL,
- affinity,
- dilated.

Report:

- mean,
- std,
- CI,
- parent-cluster bootstrap,
- per-image paired deltas.

---

## Phase 2 — Establish whether SAM itself matters

Compare:

- no-KD,
- SAM soft KD,
- Gaussian GT,
- signed-distance GT,
- morphology-derived soft GT,
- optionally corrupted/shuffled teacher control.

This phase is scientifically decisive.

---

## Phase 3 — Improve task-aligned evaluation

Add:

- Dice,
- IoU,
- clDice,
- boundary F,
- width-stratified recall,
- connectivity metrics,
- false-positive density.

---

## Phase 4 — Data-centric OOD improvements

Test:

- hard-negative mining,
- genuine full-resolution scenes,
- native high-resolution crops,
- crack-free road patches,
- context-preserving sampling,
- 640-native tiles,
- improved threshold/tile tuning using development data only.

---

## Phase 5 — Model-centric improvements

Only after the previous phases are valid:

- P3-only LayerKD,
- uncertainty-gated KD,
- instance-normalized KD,
- clDice-aware KD,
- larger student,
- multi-teacher fusion,
- multi-prompt SAM teacher,
- improved tile merging,
- multi-scale inference.

---

# 21. Top Experiments by Information Gain

1. **Canonical split + parent leakage audit**
2. **Frozen evaluator and checkpoint-hash regeneration**
3. **Teacher/augmentation alignment audit**
4. **OOD label completeness/provenance audit**
5. **Multi-seed baseline vs Mask-KL vs affinity vs dilated**
6. **SAM vs GT-derived soft targets**
7. **Topology/boundary metric re-evaluation**
8. **Teacher quality/calibration audit**
9. **End-to-end full-resolution latency benchmark**
10. **Corrupted/shuffled teacher negative control**

These should be prioritized above another exotic KD loss.

---

# 22. Top Experiments for Actual Performance Improvement

After scientific validity is established:

1. native/high-resolution training on genuine scenes,
2. hard-negative mining,
3. topology-aware loss such as clDice,
4. threshold/tile tuning on development data,
5. matched compact semantic student,
6. native 640×640 tiling,
7. P3-only intermediate KD,
8. per-instance rather than per-anchor KD normalization,
9. confidence/uncertainty-gated KD,
10. slightly larger student on the latency/accuracy Pareto curve.

---

# 23. Recommended Reporting Changes

## Replace relative percentages with absolute deltas first

Instead of:

> +19% OOD improvement

prefer:

> mAP50 increased from 0.0276 to 0.0329 (+0.0053 absolute; approximately +19% relative).

## Avoid “best” claims from single seeds

Use:

> The highest observed single-seed score was...

until statistically supported.

## Avoid “calibrated probabilities”

Use:

> soft logits / soft confidence maps

unless calibration is measured.

## Avoid “domain-agnostic”

Use:

> cross-dataset improvement was observed in the tested transfer directions

until multiple external domains support the broader claim.

## Avoid “107.8 FPS full-resolution”

Use:

> 9.27 ms model-only latency per inference call

until the complete tiled pipeline is benchmarked.

## Avoid “sub-pixel structural understanding”

Unless there is a direct endpoint demonstrating sub-pixel recovery.

---

# 24. Recommended Primary Endpoints

A stronger evaluation suite would use:

## Primary

**clDice / centerline Dice** on original semantic masks.

This directly measures whether crack topology remains connected.

## Co-primary

**False-positive density** on crack-free road background.

Possible forms:

- false-positive pixels / megapixel,
- false-positive connected components / megapixel.

## Secondary

- semantic Dice,
- IoU,
- boundary F,
- HD95,
- connectivity error,
- Mask mAP50,
- Mask mAP50-95,
- box mAP,
- recall by crack width.

Mask mAP should remain for continuity but should not be the sole scientific centerpiece.

---

# 25. Suggested Publication-Rescue Plan

## 0–1 T4 day

- fix split/provenance,
- freeze evaluator,
- regenerate tables,
- audit alignment,
- measure full-pipeline latency,
- recompute topology/boundary metrics from existing predictions if possible,
- correct all manuscript contradictions.

If the split is wrong or teacher alignment is broken:

> stop model development and repair the pipeline first.

---

## 3–5 T4 days

Run 5 seeds for:

- no-KD,
- Mask-KL,
- affinity,
- dilated.

Add:

- teacher quality audit,
- parent-cluster bootstrap CIs.

At this point you can answer:

> Does a reproducible KD effect exist?

---

## 10–20 T4 days

Add:

- GT-soft-target controls,
- corrupted-teacher control,
- hard-negative baseline,
- clDice baseline,
- P3-only LayerKD,
- at least one true external dataset,
- one matched compact semantic baseline,
- confirmatory untouched-test evaluation.

This would support a much stronger paper.

---

# 26. Stronger Experimental Design

A robust final study could contain the following arms:

```text
1. no-KD
2. no-KD + clDice
3. no-KD + hard negatives
4. GT-soft target
5. SAM Mask-KL
6. SAM KD + best structural term
7. corrupted/shuffled SAM control
8. compact semantic student
```

Use:

- 5 seeds each,
- matched compute,
- matched augmentation,
- matched HPO budget,
- frozen thresholds,
- frozen test set,
- parent-scene clustered confidence intervals.

A useful factorial crossing would be:

```text
{no-KD, SAM-KD}
×
{original data pipeline, high-res + negatives pipeline}
```

This separates:

- teacher benefit,
- from data-centric benefit.

That distinction is currently blurred.

---

# 27. Reviewer Questions the Final Paper Must Be Able to Answer

1. Which exact Crack500 test set was used?
2. Were all train/val/test splits parent-scene disjoint?
3. What exactly is the OOD dataset?
4. Are the OOD annotations complete?
5. Why do nominally identical KD recipes differ materially?
6. Why do two checkpoints have identical OOD numbers?
7. How are cached SAM logits transformed under augmentation?
8. What is the teacher’s own performance versus GT?
9. Does SAM beat a Gaussian or distance-transform soft target?
10. Why use instance mAP as the primary metric for semantic crack masks?
11. What is the multi-seed variance?
12. Where did α = 0.9612 come from?
13. What is the true end-to-end tiled throughput?
14. Why is YOLO instance segmentation preferable to a compact semantic model?
15. Does the improvement remain on a genuinely external untouched domain?

If the paper can answer these cleanly, it becomes substantially stronger.

---

# 28. Final Assessment

The most important lesson from the Opus audit is not:

> “Try a better KD loss.”

It is:

> **Make the experimental foundation trustworthy enough that any observed gain can be interpreted scientifically.**

The CrackDistill project still has a potentially strong contribution.

A particularly compelling outcome would be:

1. correct canonical evaluation,
2. geometrically valid teacher targets,
3. statistically significant gain across seeds,
4. SAM outperforming GT-derived soft-target controls,
5. improved clDice/topology rather than only mAP,
6. measurable benefit on a true external domain,
7. acceptable end-to-end full-resolution latency.

If those hold, the paper becomes much stronger than a simple:

> “+3.1% Mask mAP50 from SAM distillation”

story.

The stronger contribution would be:

> **Foundation-model structural knowledge can be transferred into a tiny deployable crack-segmentation student, producing reproducible improvements in topology and cross-scale robustness beyond what can be achieved using binary or deterministically softened ground-truth supervision alone.**

That is the hypothesis the next experiments should now attempt to prove or falsify.
