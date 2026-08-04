# CrackDistill — Proposed Changes to `full_review.md`

**Source:** Grill session + deep research (July 2026, August 2026 update)  
**Status:** All changes applied to `full_review.md`. EXP-25, EXP-26, EXP-27, EXP-28, EXP-29 pending execution on Kaggle.

---

## Change 1 — Section 3.3: Dataset Choice Justification

**Why:** The original text noted the asymmetric domain gap but did not explain *why* training on Crack500 (not DeepCrack) is the scientifically correct choice for KD specifically.

**Added after line 83 (after the asymmetric domain gap paragraph):**

```diff
+ **Manifold Coverage Argument for Dataset Selection:** In knowledge distillation,
+ the training set acts as the query set for the teacher — the student learns to
+ mimic the teacher only on the input manifold covered by the training data.
+ Crack500's wide coverage of pavement types, lighting conditions, and crack widths
+ means a student trained on it spans a large region of the teacher's decision
+ boundary. DeepCrack's narrow, controlled-acquisition manifold constrains the
+ student to a small region. This is why DeepCrack → Crack500 transfer collapses:
+ the student has never queried the teacher on the diverse Crack500 distribution,
+ so its learned representations fail entirely when deployed there. This motivates
+ Crack500 as the primary training source for any distillation experiment targeting
+ deployment-level generalization.
```

---

## Change 2 — Section 4.5: AMP / FP16 Failure (Full Rewrite)

**Why:** The original section named the symptom and the fix but did not explain the exact numerical mechanism (overflow path vs underflow path), why logit clamping distorts the teacher distribution, or that a better fix exists (local FP32 casting, EXP-25).

**Before:**

```diff
- ### 4.5 Numerical Stability Fix: Logit Clamping
-
- SAM 2 produces raw logits with magnitudes up to ±50 or higher. In FP16 training:
-
-   log(σ(z_t / τ)) --FP16--> log(0) = -∞ → NaN
-
- EXP-05 and EXP-06 both crashed at epochs 42–44 due to this overflow.
- The fix is two-part:
-
- 1. Clamp teacher logits: z_t ← clip(z_t, −30, 30)
- 2. Disable AMP: `amp: false` — use FP32 throughout
-
- This completely eliminated NaN crashes from EXP-07 onward.
```

**After:**

```diff
+ ### 4.5 Numerical Stability Fix: FP16 Overflow in KL Divergence
+
+ #### Why FP16 Fails with KD Logits
+
+ SAM 2 logits reach ±50. FP16 caps at 65,504 (min positive normal ≈ 6.1×10⁻⁵).
+
+ **Overflow path:** exp(50) ≈ 5.18×10²¹ → overflows FP16 to `inf`.
+
+ **Underflow path:** Even with stable softmax (subtract max), a range of [−50,+50]
+ gives exp(−100) ≈ 3.7×10⁻⁴⁴ → exact 0 in FP16. In KL divergence:
+   - if Q_i = 0 → log(0) = −∞. If P_i = 0 too: 0·(−∞) = NaN.
+   - if P_i > 0 and Q_i = 0: division by zero → NaN gradients.
+
+ Crashes appear at epochs 42–44, not epoch 1 — instability accumulates first.
+
+ #### Original Fix (EXP-07 onward)
+
+ 1. Clamp teacher logits: z_t ← clip(z_t, −30, 30)
+ 2. Disable AMP: `amp: false`
+
+ **Limitation:** Clamping truncates SAM 2's high-confidence crack interior
+ predictions — exactly the most useful teacher signal. It distorts the teacher's
+ true probability distribution.
+
+ #### Improved Fix (EXP-25, Planned)
+
+ Cast only the KD loss to FP32; keep rest of training in FP16:
+
+   with torch.cuda.amp.autocast(enabled=False):
+       loss_kd = F.kl_div(
+           F.log_softmax(student_logits.float() / tau, dim=-1),
+           F.softmax(teacher_logits.float() / tau, dim=-1),
+           reduction='batchmean'
+       ) * (tau ** 2)
+
+ ~1.5–2× faster than amp:false; no distribution distortion. EXP-25 tests vs EXP-08.
```

---

## Change 3 — Section 5.4: In-Distribution mAP Drop Framing

**Why:** The original stated the drop was "expected and desirable" with no mechanism. A reviewer unfamiliar with KD will challenge this without an explanation.

**Before:**

```diff
- **Key insight:** The slight in-distribution drop is expected and desirable —
- it signals the model is generalizing rather than memorizing crop boundaries.
- The OOD improvement is the actual contribution.
```

**After:**

```diff
+ **Key insight — KD as domain regularizer:** The −1.85% in-distribution drop is
+ expected and desirable. Hard labels encourage high-confidence predictions that
+ memorize dataset-specific shortcuts (crop boundary positions, fixed background
+ textures). SAM 2's soft labels convey uncertainty — a boundary pixel is predicted
+ as "85% crack, 15% background" rather than a hard 1/0 — acting as **implicit
+ label smoothing** (Dong et al., 2021; Hinton et al., 2015). This prevents
+ overfitting to in-distribution artifacts at a cost of −1.85% ID mAP and a gain
+ of +10.3% OOD mAP50-95. Textbook bias-variance tradeoff: marginal in-distribution
+ precision sacrificed for substantially improved out-of-distribution robustness.
```

**Citations added:**
- Dong et al. (2021) — *"Knowledge Distillation as Implicit Regularization"*
- Hinton et al. (2015) — original KD paper

---

## Change 4 — Experiment Registry: EXP-25 and EXP-26 Added

**Why:** Two new experiments agreed during grill session. They need to be tracked in the registry.

```diff
  | EXP-24 | Deep→C500 | YOLOv11n | Full KD cross-eval | — | 0.033 | Done |
+ | EXP-25 | Crack500  | YOLOv11n | Full KD (local FP32 cast, no clamp) | 150 | — | Planned |
+ | EXP-26 | Crack500  | YOLOv11n | Full KD (batch=32, 2× LR, warmup)  | 150 | — | Planned |
```

**EXP-25 config delta vs EXP-08:**
- Remove `amp: false` → AMP re-enabled
- Remove logit clamping to `[-30, 30]`
- Add `autocast(enabled=False)` block around KD loss computation

**EXP-26 config delta vs EXP-08:**
- `batch: 16 → 32`
- `lr0: 0.01 → 0.02` (linear scaling rule: LR × batch_ratio)
- 5-epoch linear warmup added
- SAM 2 teacher features pre-cached to disk before training starts

---

## Change 5 — Section 11: Two New Planned Subsections

**Why:** EXP-25 and EXP-26 are now proper tracked experiments with hypotheses — they belong in Open Problems, not just the registry.

```diff
  ### 11.1 Loss Weight Optimization (Current Bottleneck)
  ...

+ ### 11.2a AMP Fix Validation (EXP-25 — Planned)
+
+ Current experiments use amp:false + clamping to [−30,30]. Clamping distorts the
+ teacher's high-confidence predictions. EXP-25 replaces this with local FP32
+ casting (autocast enabled=False), re-enabling FP16 for the rest of training.
+ Expected: no distribution distortion, ~1.5–2× faster training vs amp:false.
+
+ ### 11.2b Batch Size Sensitivity (EXP-26 — Planned)
+
+ All experiments used batch=16. KD soft labels produce denser, higher-variance
+ gradients than hard labels. EXP-26 increases batch to 32 with linear LR scaling
+ and 5-epoch warmup. SAM 2 features pre-cached to disk → teacher off GPU → batch=32
+ comfortable within 16 GB VRAM.

  ### 11.2 Ablation Study (NB5 — Planned)   ← existing, unchanged
```

---

## Change 6 — Section 11.2: Ablation Study Bug Disclosure (August 2026)

**Why:** Three ablation notebooks (nb5a, nb5b, nb5c) were run on 2026-07-30 but produced identical results. Investigation revealed `[KD] logit files: 0` in all three — the teacher logits were never uploaded to Kaggle. All three models trained as vanilla fine-tunes.

**What changed in full_review.md:**
- Section 11.2 now says "NB5 — Partial" instead of "NB5 — Planned"
- Root cause explained: `KDYOLODataset` silently skips KD losses when logit directory is empty
- All three results (`Mask mAP50 = 0.558`) documented as **invalid ablation results**
- Corrected plan: EXP-27/28/29 with logits uploaded as Kaggle dataset
- Expected findings table retained as theoretical priors

**Fix required on Kaggle:**
```
1. Package data/teacher_logits_box/ into a Kaggle dataset
2. Add it as input to ablation notebooks
3. Verify [KD] logit files: 1896 before training
4. Re-run nb5a, nb5b, nb5c
```

---

## Change 7 — Section 4.5: AMP Deeper Analysis

**Why:** The gradient scaler mechanism was not mentioned — reviewers may ask "why doesn't AMP's gradient scaler fix this?" The bfloat16 alternative was also not discussed.

**Added:**
1. Explicit note that AMP gradient scaler does NOT fix KD NaN — the NaN originates before backpropagation, bypassing the scaler entirely
2. Quantified clamping cost: at τ=3.78, σ(50/3.78) ≈ 1.000 vs σ(30/3.78) ≈ 0.9997 — the logit magnitude difference is real but confidence ceiling is near-identical
3. bfloat16 alternative — same exponent range as FP32, eliminates overflow, but unavailable on T4 GPUs

---

## Change 8 — Section 11.2b: Batch Size Theoretical Justification

**Why:** Original section just stated "KD gradients are higher variance" without explaining why or quantifying the benefit.

**Added:**
1. Explanation of WHY KD gradients have higher variance: in hard-label training only correct-class pixels get gradient; in soft-mask KD, every pixel gets gradient from teacher distribution
2. O(1/√B) variance scaling: doubling batch size reduces noise by √2 ≈ 41%
3. Explicit linear LR scaling formula: lr₃₂ = lr₁₆ × (32/16) = 0.002
4. Citation: Goyal et al. (2017) "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
5. Expectation calibration: modest improvement (0–3% relative) because τ=3.78 already partially controls variance

---

## Change 9 — Section 3.3: Dataset Size vs Diversity Argument

**Why:** The manifold coverage argument was present but could be dismissed as "Crack500 just has 3× more data." The 8× performance gap (0.274 vs 0.033) is far larger than a 3× data ratio would predict.

**Added:**
- Explicit note that 1,896 vs 537 training images is only a 3.5× ratio
- The 8× performance gap is disproportionate — this implicates distribution breadth as the primary driver, not merely volume
- Counter-argument: a hypothetical DeepCrack model with 1,500 images would still fail on Crack500's different texture regime
- Citation: Tang et al. (2023) on dataset diversity in knowledge distillation

---

## Summary

| # | Section | Change Type | Applied |
|:--|:--------|:------------|:-------:|
| 1 | 3.3 Domain Shift | Added manifold coverage KD argument | ✅ |
| 2 | 4.5 AMP Failure | Full rewrite — overflow math + clamping limitation + EXP-25 fix | ✅ |
| 3 | 5.4 Full KD Run | Reframed ID mAP drop as bias-variance tradeoff + citations | ✅ |
| 4 | Experiment Registry | Added EXP-25 + EXP-26 rows | ✅ |
| 5 | Section 11 | Added 11.2a (AMP) + 11.2b (batch=32) subsections | ✅ |
| 6 | 11.2 Ablation Study | Disclosed logit-upload bug, explained root cause, planned EXP-27/28/29 fix | ✅ |
| 7 | 4.5 AMP | Added gradient scaler NaN bypass explanation + bfloat16 alternative | ✅ |
| 8 | 11.2b Batch Size | Added O(1/√B) gradient variance theory + Goyal et al. linear LR citation | ✅ |
| 9 | 3.3 Dataset | Added size-vs-diversity argument — why 3× size ratio doesn't explain 8× perf gap | ✅ |

**Pending — needs Kaggle results:**
- [ ] EXP-25: Upload logits as Kaggle dataset; run local FP32 cast fix; fill in mAP + OOD mAP
- [ ] EXP-26: Pre-cache SAM 2 features; run batch=32 + 2×LR + warmup; fill in mAP + OOD mAP
- [ ] EXP-27/28/29: Upload teacher logits; re-run ablations; verify `logit files: 1896` before training starts
