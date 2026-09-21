# CrackDistill Post-Remediation Independent Audit — Structured Report

**Project:** CrackDistill  
**Audit basis:** Opus post-remediation repository deep dive  
**Current audited HEAD:** `0a60d709d1f419d264f70951fa98ff28cd272cb0`  
**Historical comparison commit:** `984df279794c9c7d40808d7438c0cfa28486c7c0`  
**Audit outcome:** **D — NOT READY: core correctness issues remain**

---

# 1. Executive Summary

The remediation effort fixed several genuine source-code defects, but it did **not** generate any new experimental evidence.

Every committed numerical result still comes from the pre-remediation pipeline.

More importantly, the independent re-audit identified new or remaining issues that prevent the current repository from establishing a valid knowledge-distillation effect.

The central conclusion is:

> **CrackDistill's SAM-to-YOLO distillation contribution is still not scientifically demonstrated. The strongest defensible result remains the tiled full-resolution inference pipeline.**

The most important blockers are:

1. teacher/student geometry remains misaligned because SAM square-stretches images while YOLO letterboxes them;
2. the new baseline can still activate KD depending on runtime teacher-logit discovery;
3. baseline and KD arms now use different augmentation policies;
4. no remediated experiment has actually been executed;
5. the canonical 1,124-image test set remains unused;
6. no GT-soft control exists;
7. no multi-seed confirmatory study exists;
8. manuscript contradictions remain uncorrected.

---

# 2. What Was Actually Fixed

The remediation did successfully address several previously identified defects at the source-code level.

## 2.1 Seed forwarding

The current code forwards the configured seed into Ultralytics and seeds:

- Python,
- NumPy,
- PyTorch CPU,
- CUDA.

### Status

**FIXED IN SOURCE**

### Remaining requirement

The multi-seed experiments must now actually be rerun.

---

## 2.2 CWD channel normalization

The previous CWD loss scaled with channel count.

The current code normalizes by channel count.

### Status

**FIXED IN SOURCE**

---

## 2.3 Missing teacher-feature zero fallback

The prior code substituted zero-valued teacher feature tensors.

The current code instead subsets to valid teacher-feature samples.

### Status

**FIXED IN SOURCE**

---

## 2.4 LayerKD student layer indices

The previous layer indices did not correspond to the intended final segmentation features.

The new implementation uses corrected student layer indices.

### Status

**FIXED IN SOURCE**

### Important caveat

The teacher side still provides only two distinct feature tensors for three student scales, so the claimed three-scale teacher representation remains questionable.

---

## 2.5 Instance-index clamping

The original code silently clamped invalid teacher instance indices to the last teacher instance.

The new code masks invalid indices instead.

### Status

**PARTIALLY FIXED**

### Remaining issue

Invalid instances are silently dropped without:

- a counter,
- diagnostic logging,
- drop-rate reporting,
- fail-fast validation.

The system therefore still cannot prove that teacher-instance alignment is complete.

---

## 2.6 Checkpoint provenance utility

A SHA256/provenance utility now exists and is used in some training notebooks.

### Status

**PARTIALLY FIXED**

The final evaluator still uses glob-based checkpoint discovery, so result provenance is not yet deterministic end-to-end.

---

# 3. Critical Finding — No New Experiments Exist

The remediation is largely code-level.

According to the repository audit:

- no remediated training output exists,
- no remediated checkpoint exists,
- no remediated leaderboard exists,
- no post-fix multi-seed run exists,
- no post-fix baseline run exists,
- no post-fix KD run exists.

The newest executed notebook predates the remediation.

## Consequence

> **A source-code fix does not validate previous experimental results.**

All historical KD numbers remain products of the old invalid pipeline.

The project therefore currently has:

```text
corrected source code in some areas
+
old experimental evidence
```

rather than:

```text
corrected source code
+
new confirmatory evidence
```

---

# 4. Critical New Finding — SAM Stretch vs YOLO Letterbox Geometry

This is now the most important technical problem.

## 4.1 Teacher geometry

SAM 2 processes non-square images through a square resize.

Conceptually:

```text
640 × 360
→ 1024 × 1024
```

The aspect ratio is not preserved.

---

## 4.2 Student geometry

With mosaic disabled, the YOLO training pipeline letterboxes the image into the square input.

For a 640×360 image at 512×512:

```text
640 × 360
→ 512 × 288
→ vertical padding to 512 × 512
```

This preserves aspect ratio.

---

## 4.3 Why direct interpolation is insufficient

The current KD path resizes the SAM tensor and the student tensor to a common output shape.

But equal tensor dimensions do not imply equal coordinate systems.

The teacher representation corresponds to:

```text
anisotropically stretched source coordinates
```

while the student representation corresponds to:

```text
aspect-preserved image
+
letterbox padding
```

Therefore a pixel at row `y` in the teacher does not generally correspond to the same physical road location as row `y` in the student.

---

## 4.4 Quantified mismatch reported by the audit

For a 640×360 source and 512×512 YOLO input:

- resized image content: `512×288`,
- vertical padding: `112 px`,
- approximately `43.8%` of the square student canvas height is padding,
- reported vertical teacher/student spatial-scale difference: approximately `1.778×`,
- maximum source-coordinate displacement reported by the diagnostic: approximately `±140 px`,
- only the image centre line aligns exactly.

For thin 4–8 px cracks, such displacement is catastrophic for pixel-level KD.

---

## 4.5 Scientific consequence

Disabling mosaic/flip/scale augmentation does **not** solve this coordinate-system mismatch.

It only removes one source of misalignment.

### Current status

**NOT FIXED**

### Affected KD mechanisms

Potentially all spatial teacher losses:

- Mask-KL,
- affinity,
- dilated KD,
- Tversky-style teacher losses,
- boundary losses,
- offline LayerKD features.

---

# 5. Required Alignment Fix

The teacher and student must represent the same physical coordinate system.

Two defensible strategies exist.

## Strategy A — Warp cached SAM logits into YOLO letterbox coordinates

For each image:

1. recover original image shape,
2. determine YOLO resize ratio,
3. resize teacher target to the unpadded YOLO content dimensions,
4. add the exact YOLO padding,
5. apply any remaining shared transforms,
6. compare student/teacher only in the aligned coordinate system.

This requires careful preservation of:

- original shape,
- resize ratio,
- padding offsets,
- interpolation mode.

---

## Strategy B — Generate teacher targets from the exact student-preprocessed image

Instead of teaching from SAM's own independent square-stretch preprocessing:

```text
raw image → SAM preprocessing
```

build teacher targets from the exact image geometry used by the student:

```text
raw image
→ YOLO letterbox
→ teacher inference on aligned canvas
```

The SAM input may still internally resize, but its output can then be mapped consistently back to the known letterboxed canvas.

This may be conceptually cleaner.

---

# 6. Mandatory Alignment Diagnostic

Before rerunning expensive training, validate the corrected geometry.

Sample approximately 200 training batches.

For each matched instance compute:

```text
IoU(
    transformed GT support,
    teacher target support
)
```

Report:

- mean IoU,
- median IoU,
- 5th percentile,
- fraction `< 0.50`,
- fraction `< 0.75`,
- fraction `< 0.90`,
- number/fraction of missing teacher instances,
- number/fraction of dropped teacher IDs,
- number/fraction of invalid mappings.

Also save visual overlays.

## Suggested acceptance criterion

The audit proposed a very strict criterion such as:

```text
median alignment IoU > 0.95
```

before treating the KD pipeline as geometrically valid.

The exact threshold can be debated, but the principle is correct:

> alignment must be empirically demonstrated, not inferred from tensor dimensions.

---

# 7. Critical New Finding — "Clean Baseline" May Still Use KD

The new baseline notebook disables:

```text
distillation.enabled = False
```

but the audit reports that:

- model loss patching is unconditional,
- dataset KD wrapping is unconditional,
- individual KD loss flags remain enabled,
- teacher-logit auto-discovery remains active.

Therefore, if teacher logits are found at runtime, the baseline may still receive KD supervision.

## Scientific consequence

A control arm must not depend on whether a Kaggle dataset happens to be attached.

### Current status

**NOT FIXED**

---

# 8. Required Baseline Fix

The baseline must explicitly satisfy all of the following:

```text
distillation.enabled = False
losses.mask_kd.enabled = False
losses.feature.enabled = False
losses.boundary.enabled = False
losses.affinity.enabled = False
losses.dilated.enabled = False
```

More importantly, code paths should be gated.

Conceptually:

```python
if kd_cfg.enabled:
    wrap_dataset_for_kd()
    patch_model_loss()
    load_teacher_targets()
```

Otherwise:

```python
use native YOLO training path
```

A true baseline should never:

- load SAM logits,
- install KD hooks,
- patch KD loss,
- depend on teacher cache availability.

---

# 9. New Regression — Baseline and KD Arms Use Different Augmentation Policies

The remediation disables spatial augmentation only when KD is enabled.

Therefore:

## KD arms

Reportedly run with:

```text
mosaic = 0
fliplr = 0
scale = 0
translate = 0
erasing = 0
```

## Baseline

Retains ordinary Ultralytics augmentation.

---

## Scientific consequence

A future comparison would become:

```text
baseline:
    task loss + strong augmentation

vs

KD:
    task loss + SAM supervision + weak/no spatial augmentation
```

Any performance difference would be confounded.

You would not know whether it came from:

- SAM distillation,
- removing augmentation,
- or interaction between both.

### Current status

**REGRESSION**

---

# 10. Required Fair-Comparison Policy

Choose one augmentation policy and use it for both baseline and KD arms.

For the first confirmatory study, the cleanest design is probably:

```text
mosaic = 0
fliplr = 0
scale = 0
translate = 0
perspective = 0
copy-paste = 0
```

for **both** baseline and SAM-KD.

That deliberately sacrifices augmentation strength to isolate the teacher effect.

Later, a second experiment can introduce correctly transformed KD targets under augmentation.

The first scientific question should be:

> Does aligned SAM supervision help when everything else is equal?

---

# 11. New Bug — KD Exception Handler Returns `None`

The new exception-handling logic attempts to:

- count failures,
- tolerate several transient failures,
- eventually fail under strict mode.

But according to the audit, the exception path does not return the expected KD-loss dictionary.

The calling code then attempts something like:

```python
kd_losses.items()
```

on `None`.

## Consequence

The intended tolerant-error logic is unreachable.

The first KD exception can crash training.

### Current status

**NEW BUG**

### Fix

Ensure all non-raising paths return a valid loss structure.

Even better for research:

> remove broad exception swallowing entirely and fail immediately on unexpected KD errors.

Silent robustness is undesirable during scientific validation.

---

# 12. New Test-Suite Failure

The new unit tests reportedly run:

```text
13 tests
4 errors
```

because notebook JSON files are opened without explicitly requesting UTF-8.

### Current status

**NEW BUG / VERIFICATION FAILURE**

### Fix

Use:

```python
open(path, encoding="utf-8")
```

for notebook JSON reads.

The broader lesson is more important:

> do not claim remediation is verified while the committed verification suite itself fails.

---

# 13. LayerKD — Improved but Still Scientifically Incomplete

The student hooks now correspond to the intended segmentation feature layers.

That is progress.

However the teacher feature mapping remains:

```text
one fine teacher feature
+
one coarse teacher feature reused for two student scales
```

So three student scales are not being matched to three distinct SAM feature scales.

## Current status

**PARTIALLY FIXED**

## Better next experiment

Do not immediately rebuild the full three-level LayerKD system.

First test:

> **P3-only correctly aligned feature KD**

because fine-scale structure is the stated hypothesis.

This is easier to validate and interpret.

---

# 14. Affinity Loss Remains Unsupported

The affinity implementation reportedly remains:

- two directional differences,
- not four,
- extremely small relative to Mask-KL.

The historical first-batch magnitudes were approximately:

```text
Mask-KL ≈ 3.160839
Affinity ≈ 0.001256
```

That is roughly `0.04%` of the primary KD loss magnitude.

The committed historical run scored:

```text
04_affinity = 0.5348
```

not `0.5569`.

### Current status

**CLAIM STILL CONTRADICTED**

The paper should not present this as the primary positive contribution.

---

# 15. Manuscript Still Contains the Old Incorrect Result

The manuscript reportedly still contains:

```text
04_affinity Mask mAP50 = 0.5569
```

in multiple locations.

The repository's committed executed run supports:

```text
0.5348
```

The manuscript file was moved but not materially corrected.

### Current status

**NOT FIXED**

## Required action

Correct every occurrence before any further circulation of the paper.

Do not wait for the next experiment.

Historical incorrect values should be clearly removed from:

- abstract,
- results tables,
- discussion,
- conclusion,
- figure captions,
- README/project summaries.

---

# 16. Test-Set Status

The canonical Crack500 test split remains untouched.

That is good.

But it is untouched because it is still unused.

## Current state

Development decisions use:

- 1,896 train patches,
- 348 validation patches,
- 50 uncropped validation parents.

Available but unused confirmatory sets include:

- 1,124 canonical test patches,
- 200 uncropped test parents.

### Scientific opportunity

This is actually favorable.

After fixing the pipeline, you still possess a genuinely useful untouched confirmatory split.

Do not spend it prematurely.

---

# 17. Hyperparameter Leakage Remains

The historical τ and α were selected with an objective involving the uncropped validation parents.

Therefore the 50-parent full-resolution validation result is not independent confirmation.

## Required confirmatory policy

Freeze:

- τ,
- α,
- threshold,
- tile size,
- tile stride,
- Gaussian sigma,
- all structural loss weights

before evaluating:

- 1,124 test patches,
- 200 uncropped test parents.

The test parent set should not participate in HPO.

---

# 18. GT-Soft Control Still Missing

The project still does not answer the most important scientific alternative explanation:

> Does SAM provide more useful supervision than a deterministic soft representation of the ground truth?

Required arms:

```text
A. no-KD
B. SAM Mask-KL
C. Gaussian-soft GT
D. signed-distance-transform GT
E. optional morphology-derived soft GT
F. shuffled/corrupted teacher control
```

Use identical:

- seeds,
- initialization,
- optimizer,
- augmentations,
- image size,
- epochs,
- evaluator.

Without this experiment, even a positive SAM result cannot cleanly establish foundation-model knowledge transfer.

---

# 19. Teacher Quality Still Unknown

No teacher-quality audit exists.

Before making claims about:

- structural priors,
- calibrated uncertainty,
- boundary understanding,
- topology,

measure the actual teacher against ground truth.

Recommended metrics:

- Dice,
- IoU,
- clDice,
- boundary F,
- skeleton precision,
- skeleton recall,
- performance by crack width,
- performance by branch complexity.

If calibration is claimed:

- ECE,
- Brier score,
- reliability curves.

---

# 20. Statistical Evidence Still Missing

No post-remediation multi-seed experiments exist.

Historical runs are effectively single-run evidence.

Therefore there is currently no defensible estimate of:

- mean performance,
- standard deviation,
- confidence intervals,
- seed variance.

## Minimum confirmatory design

At least:

```text
baseline: 3–5 seeds
Mask-KL: 3–5 seeds
GT-soft: 3–5 seeds
```

Only expand to:

- dilated,
- LayerKD,
- other structural losses

after the basic SAM effect is demonstrated.

---

# 21. Deployment Claim Remains Unresolved

The historical `107.8 FPS` is still a single-tile model-forward metric.

The full-resolution tiled pipeline:

- uses approximately 20 tiles per 2000×1500 scene,
- processes tiles serially,
- has no complete latency benchmark.

A serial model-only estimate remains roughly:

```text
20 × 9.27 ms
≈ 185 ms
≈ 5.4 scenes/s
```

This is only an estimate.

## Required deployment experiment

Measure:

```text
decode
→ tile generation
→ preprocessing
→ H2D
→ batched tile inference
→ mask decode
→ NMS
→ Gaussian blending
→ final output
```

Report:

- p50,
- p95,
- scenes/s,
- tile batch size,
- VRAM,
- GPU,
- precision.

---

# 22. What Still Survives Scientific Scrutiny

Despite the KD problems, two claims remain comparatively strong.

## 22.1 Tiled full-resolution inference

Across seven historical checkpoints, Gaussian-apodized tiling reportedly improves Dice by roughly:

```text
+51.7% to +84.8%
```

relative to direct resize inference.

The direct and tiled paths use the same mask threshold.

This result is independent of the disputed no-KD baseline.

### Status

**SUPPORTED BY CURRENT REPOSITORY EVIDENCE**

---

## 22.2 No SAM runtime dependency

SAM is used only for offline/training supervision.

The deployed student does not require SAM.

### Status

**SUPPORTED**

---

# 23. Scientific Claim Audit

| Claim | Current verdict |
|---|---|
| SAM transfers structural priors | **Unsupported** |
| SAM beats no-KD | **Unknown** |
| SAM beats GT-soft supervision | **Untested** |
| Mask-KL improves the student | **Unsupported** |
| Affinity improves in-domain performance | **Contradicted by committed run** |
| Dilated KD improves over a real no-KD full-res baseline | **Unknown** |
| LayerKD provides validated multi-scale teacher transfer | **Unsupported / partially repaired in source** |
| KD gains exceed seed variance | **Unsupported** |
| Results hold on canonical 1,124-image test set | **Untested** |
| Results hold on uncropped 200 test parents | **Untested** |
| Results hold on a true external domain | **Untested** |
| Tiled inference substantially improves full-res Dice | **Supported** |
| Full-resolution system runs >100 FPS | **Unsupported / contradicted as currently framed** |
| No SAM runtime dependency | **Supported** |
| Teacher probabilities are calibrated | **Unsupported** |
| Hairline topology improves | **Untested** |

---

# 24. Publication Readiness

## Verdict

# **D — NOT READY: CORE CORRECTNESS ISSUE REMAINS**

The source has improved substantially, but the scientific evidence has not yet caught up.

Four independent blockers remain:

1. teacher/student coordinate systems are still misaligned;
2. the baseline is not guaranteed to be KD-free;
3. baseline and KD augmentation policies differ;
4. zero remediated experiments have been run.

Any one of these is enough to prevent causal interpretation of the KD effect.

---

# 25. Minimum Recovery Plan

The next work should be aggressively minimal.

Do **not** create new KD variants.

## Step 1 — Fix geometry

Make teacher and student targets share exactly the same letterboxed coordinate system.

Then run the alignment diagnostic.

### Acceptance requirement

No training until overlays and quantitative alignment demonstrate correctness.

---

## Step 2 — Fix the baseline

Make KD completely unreachable when:

```text
distillation.enabled = False
```

Baseline and KD must use the same augmentation settings.

---

## Step 3 — Fix fail-fast verification

- repair `_kd_loss_from_preds` return path,
- repair tests,
- log invalid target counts,
- remove evaluator glob fallback,
- make checkpoint provenance mandatory.

---

## Step 4 — Run the first real experiment

Only:

```text
Baseline
vs
Mask-KL
```

Use identical:

- seed,
- initialization,
- augmentation,
- image size,
- optimizer,
- epochs,
- evaluator.

Start with one diagnostic seed.

---

## Step 5 — If Mask-KL is promising, run multiple seeds

Recommended:

```text
3–5 baseline seeds
3–5 Mask-KL seeds
```

Measure:

- mean,
- std,
- 95% CI,
- paired per-image test deltas.

---

## Step 6 — Add GT-soft control

Only after the SAM effect is reproducible.

---

## Step 7 — Freeze everything

Then evaluate once on:

```text
1,124 Crack500 test patches
200 uncropped Crack500 test parents
```

Do not tune against these sets.

---

# 26. Suggested Confirmatory Study

A scientifically efficient confirmatory study would contain:

## Core arms

```text
A. no-KD
B. SAM Mask-KL
C. GT-soft target
```

## Seeds

```text
5 per arm
```

## Development

```text
train = 1,896
validation = 348
full-resolution development = 50 validation parents
```

## Final confirmatory evaluation

```text
test = 1,124 patches
full-resolution test = 200 test parents
```

## Primary metrics

- clDice,
- semantic Dice,
- boundary F,
- false-positive density.

## Secondary metrics

- Mask mAP50,
- Mask mAP50-95,
- Box mAP.

## Deployment

Measure full-scene end-to-end p50/p95 latency separately.

---

# 27. Decision Tree After the Fix

```text
Fix geometry + baseline
│
├── SAM-KD ≤ baseline across seeds
│     └── Distillation hypothesis not supported
│         → paper becomes tiled-inference / negative KD study
│
└── SAM-KD > baseline with CI
      │
      ├── SAM-KD ≈ GT-soft
      │     └── contribution = structured soft supervision
      │
      └── SAM-KD > GT-soft with CI
            │
            ├── effect disappears on 1,124 test / 200 parents
            │     └── development overfitting
            │
            └── effect survives untouched tests
                  └── foundation-prior transfer claim becomes defensible
```

---

# 28. Highest-Information Next Actions

## 1. Teacher/student alignment diagnostic

**Cost:** very low  
**Purpose:** determine whether current spatial supervision is valid.

---

## 2. Implement exact teacher-to-letterbox transform

**Cost:** low engineering cost  
**Purpose:** create a truly aligned KD target.

---

## 3. Build a genuinely KD-free matched baseline

**Cost:** ~one training run  
**Purpose:** create the reference required for every KD claim.

---

## 4. Run baseline vs Mask-KL at multiple seeds

**Cost:** moderate  
**Purpose:** determine whether any KD effect exceeds variance.

---

## 5. SAM vs GT-soft control

**Cost:** moderate  
**Purpose:** determine whether SAM itself is scientifically necessary.

---

# 29. Recommended Paper Position Today

If the paper had to be written from the repository **today**, the strongest defensible framing would be:

> **A high-resolution inference study showing that Gaussian-apodized tiling substantially recovers segmentation performance lost through direct downsampling in full-resolution pavement imagery.**

The SAM-distillation component should be described as:

> exploratory / inconclusive pending corrected alignment and confirmatory experiments.

That is a much stronger scientific position than continuing to defend invalidated KD results.

---

# 30. Final Assessment

The remediation work was not wasted.

Several underlying engineering defects were repaired, including:

- seed propagation,
- CWD normalization,
- teacher-feature validity handling,
- student LayerKD hooks,
- partial instance-index validation,
- checkpoint provenance utilities.

But the repository audit found that the **fundamental experiment still has not been performed correctly**.

The central next question is now extremely clear:

> **When teacher targets are correctly mapped into the student's physical coordinate system, and compared against a truly KD-free matched baseline, does SAM Mask-KL produce a reproducible improvement that exceeds seed variance?**

Until that experiment exists, the correct scientific conclusion is:

> **CrackDistill has a verified tiled-inference contribution and an unverified distillation hypothesis.**
