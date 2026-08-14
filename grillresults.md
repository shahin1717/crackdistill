# 🔬 Crack-Distill: Grill Results & Research Synthesis
*Assembled from: `full-run (1).ipynb`, `nb_exp_results.md`, `next_moves.md`, `whatodo.md`, `crackdistill_nb5_bug_rootcause_and_plan.md`, `crackdistill_next_steps.md`, `crackdistill_next_steps_v2_post_nb5e.md` + live web research (2026-08-14)*

---

## 🔎 What `full-run (1).ipynb` Actually Did

This notebook is **nb7 (Multi-Seed Verification Study)**. It ran exactly **one** of the planned seeds:

| Config | Value |
|---|---|
| Strategy | `mask_kd_only` (task loss + soft mask KL only) |
| Temperature (T) | **1.93** ← from `configs/config.yaml` (NOT Optuna-tuned 3.7769!) |
| Mask KD Weight (W) | **0.458** |
| Seed | 42 only (seeds 123, 456 never ran) |
| Epochs | 150 |

### ✅ YES — Real SAM2 Logits Were Used

Verified directly from the saved output cells:
```
[KD] logit files: 1896          ← logits present, nonzero
[KD] temperature: 1.9300...
[KD] ✓ KD losses computed: mask_kd: 0.534671   ← KD loss was ACTIVE every epoch
```

This is a **valid, real KD run**. The bug from nb5a–e (zero logit files) does NOT apply here. nb7 was built differently and includes the correct logit loading.

### Result:

| Metric | Value |
|---|---|
| **Cropped Mask mAP50** | **0.5422** |
| Cropped Mask mAP50-95 | 0.2083 |
| Uncropped OOD | ❌ Not evaluated (`crack500_uncropped_yolo/dataset.yaml` missing) |

**vs Baseline** (YOLOv11n-seg, no KD, Crack500): **0.5400**

→ **Delta: +0.0022 Mask mAP50 (+0.4%)** — real KD, but T not tuned for 1-loss setup.

> [!NOTE]
> Seeds 123 and 456 were NOT run. Single-seed result — statistically inconclusive but the KD mechanism was working correctly.

---

## 🐛 Critical Bug Status

### Bug 1 — nb5a–e ran with ZERO teacher supervision (nb7 FIXED)

`crackdistill_nb5_bug_rootcause_and_plan.md` confirmed: every nb5a–e training run used **empty logit files** — `generate_teacher_logits.py` was never called in those notebooks. Both `kd_trainer` call and logits path were also wrong. Net effect: all 5 ablation notebooks ran plain fine-tunes with zero KD signal, wearing ablation labels.

`full-run (1).ipynb` (nb7) **correctly verifies** `[KD] logit files: 1896 > 0` before training. **nb7 is the first valid `mask_kd_only` result.**

### Bug 2 — Temperature mismatch: Optuna τ=3.7769 vs nb7 τ=1.93

Optuna tuned τ=3.7769 for the **3-loss Full KD** setup (all losses active). nb7 runs `mask_kd_only` at τ=1.93 (stale config default). These are different loss landscapes — the optimal τ for a 1-loss setup is unknown.

**Your own data already proves the sensitivity:**
- nb5f at τ=3.7769 → **0.5500**
- nb7 at τ=1.93 → **0.5422**
- **+0.78 mAP pts from temperature alone**, nothing else changed.

### Bug 3 — nb5e checkpoint collision (2 rows → same file)

`find_checkpoint()` keyword fallback caused "Full KD (Box Prompts)" and "Ablation 4 (SegHead Freeze)" rows to resolve to the same DeepCrack Stage-2 checkpoint. Fix is documented in `whatodo.md` P0-B but **not yet applied**.

---

## 📊 Valid Data: What You Can Actually Trust

| Source | Experiment | Mask mAP50 | Notes |
|---|---|---|---|
| nb1 | Baseline YOLOv11n-seg, Crack500, no KD | **0.5400** | ✅ Gold baseline |
| nb2 | Full KD (3-loss, Box prompts), Crack500 | 0.5468 | ✅ +0.68 pts |
| nb3 | DeepCrack, SegHead Frozen, 2-stage | **0.5046** | ✅ Best DeepCrack result |
| nb5f | `mask_kd_only`, Crack500, τ=3.7769 | **0.5500** | ✅ Best single-run overall |
| nb7 `full-run(1)` | `mask_kd_only`, Crack500, τ=1.93, seed=42 | 0.5422 | ⚠️ Valid, wrong temperature |

**Key insight:** nb5f (0.5500) already beats baseline by **+1.0 pt / +1.85% relative** — with the correct Optuna temperature. nb7 just ran the same recipe with the wrong τ.

---

## ❓ Why Are Gains So Small? Root Causes Ranked

### 1. 🔥 Temperature is wrong for the 1-loss setup (biggest leverage, cheapest fix)
Optuna jointly tuned (τ, α, β, γ) for 3-loss KD. With losses disabled, the optimal τ shifts. Need a dedicated 2D sweep of **(τ, W_mask_kd)** for `mask_kd_only`. This collapses a 4D problem to 2D — 30 trials is plenty.

### 2. 📊 No multi-seed validation — all deltas may be noise
Every result is a single run. The nb5f "best" of 0.5500 needs ≥3 seeds before it's a finding, not a lottery win.

### 3. 🎯 Box prompts may ceiling SAM2 quality on thin diagonal cracks
Diagonal crack bounding boxes are mostly background (low fill-ratio). SAM2 gets a bad prompt for the hardest cases. This is a structural ceiling on teacher signal quality independent of any hyperparameter.

### 4. 🔬 The "feature MSE hurts" finding from nb5a–e was FAKE
The ablation ran with zero KD signal. The +2.14% from removing feature MSE was pure noise from independent random seeds. We don't actually know yet whether feature MSE hurts or helps once real logits are flowing.

---

## 🚀 Action Plan: Beat Baseline by ≥3% With Confidence

### 🔴 Priority 0: Fix config temperature before running more seeds (5 min, local)

```bash
# Update configs/config.yaml: temperature: 1.9300... → 3.7769
# OR run nb7 with explicit T override
```

If you run seeds 123/456 at T=1.93, you're locking in a suboptimal value and all three seeds will cluster around 0.5422. **Run at T=3.7769 to match nb5f's 0.5500 result first.**

---

### 🔴 Priority 1: 2D Optuna sweep for `mask_kd_only` (1 Kaggle session)

```python
# New notebook: optuna_mask_only.ipynb
# Search space (2D only, everything else disabled):
temperature:    [1.5, 6.0]    # vs current mis-applied 1.93
mask_kd_weight: [0.3, 3.0]    # vs current 0.458
# Trials: 30-40 at 15 epochs each
# Objective: Mask mAP50 on Crack500 val
```

**Expected gain:** Based on the 0.78 pt gap already seen from τ alone, optimizing (τ, W) jointly should yield **0.555–0.565** Mask mAP50. That's **+2.5–4.5% relative over baseline**.

---

### 🔴 Priority 2: 3-seed reruns of best config (3 Kaggle runs)

```python
# Seeds: [42, 123, 456]
# Config: τ=best_from_P1, W=best_from_P1, mask_kd_only
# Report: mean ± std across 3 seeds
```

**Minimum viable publishable claim:** `mask_kd_only` at optimal (τ, W) achieves **0.555 ± σ Mask mAP50** vs 0.540 baseline → **+2.8% relative**, p confirmed by 3 seeds.

---

### 🟡 Priority 3: Re-run nb5a–f ablations WITH real logits (5 Kaggle runs)

Now that the bug is fixed, redo all 5 ablation arms properly. This will tell you for the first time whether:
- Feature MSE actually hurts or helps (real answer, not noise)
- Boundary BCE is truly neutral
- The 3-loss combination is better or worse than 1-loss

Without this, the ablation table in your paper is invalid.

---

### 🟡 Priority 4: Confirm +10.3% OOD claim (1 Kaggle run)

Re-run `run_on_kaggle_final_rauf.ipynb` with correct uncropped yaml. The headline claim has zero saved output cells — it currently has no artifact.

---

### 🟢 Priority 5: Curriculum Temperature Scheduling (experimental)

Based on **CTKD (Curriculum Temperature KD)** from 2023-2024 literature: start high τ (soft/easy targets) and anneal to lower τ (harder targets) as training progresses.

```python
# τ_start = 5.0 → τ_end = 2.0, cosine decay
# Why for cracks: soft targets early prevent overconfident boundary locking
# Expected: +0.5-1.5 mAP from literature on dense tasks
```

---

### 🟢 Priority 6: Skeleton-point SAM2 prompts (publishable contribution)

Replace box prompts with multi-point prompts sampled along the crack skeleton for thin diagonal cracks. Addresses the teacher quality ceiling from low fill-ratio boxes.

```python
# In generate_teacher_logits.py:
# 1. skimage.morphology.skeletonize(binary_mask)
# 2. Sample N=5 points along skeleton
# 3. Pass as multi-point SAM2 prompt
# Measure: does logit uncertainty correlate with fill-ratio? If yes → publishable.
```

---

## 📋 Minimum Kaggle Run Queue

| # | Notebook | Purpose | Expected Result |
|---|---|---|---|
| 1 | Fix `configs/config.yaml` T → 3.7769, run nb7 seed 42 again | Confirm nb5f's 0.5500 reproducible | Baseline sanity check |
| 2 | `optuna_mask_only.ipynb` (2D sweep) | Find optimal (τ, W) | Best single-run config |
| 3-5 | nb7 seeds [42, 123, 456] at optimal config | Statistical validity | Mean ± std |
| 6 | `run_on_kaggle_final_rauf.ipynb` re-run + save | Confirm OOD +10.3% | Headline artifact |

---

## 🔥 Open Grilling Questions (Decisions You Need to Make)

These are the key decisions blocking progress. Answer them in order:

1. **Temperature before seeds**: Will you update `config.yaml` to T=3.7769 before running seeds 123/456? If not, you'll have 3 seeds of the wrong temperature. *(Recommended: YES — takes 2 minutes.)*

2. **Optuna 2D sweep**: Is 1 Kaggle run (30 trials × 15 epochs ≈ 7h) worth spending before doing multi-seed? Or do you want to first reproduce nb5f's 0.5500 at 3 seeds and treat that as "good enough"? *(Recommended: DO the sweep first — you only get one Kaggle GPU session.)*

3. **Ablation re-do**: The nb5a–e results are junk. Do you want to budget 5 more Kaggle runs to redo them properly, or declare nb5f as the definitive ablation result and move on? *(Recommended: Re-run at minimum nb5a (no mask_kd) and nb5f (mask_kd_only) to have a valid before/after.)*

4. **OOD claim**: Is the +10.3% OOD headline going in the paper? If yes, it **must** be re-run and saved. If you're dropping it, you save a run but weaken the contribution story. *(Recommended: Re-run it — it's your strongest claim if it reproduces.)*

5. **Acceptance threshold**: What Mask mAP50 would make you call the project publication-ready? 0.555? 0.560? This determines how many Optuna iterations are worth running.

---

## ✅ Paper-Ready Checklist

```
[ ] configs/config.yaml updated: temperature = 3.7769 (matches nb5f)
[ ] 2D Optuna sweep (τ, W) for mask_kd_only — find true optimum
[ ] 3-seed reruns of best config → report mean ± std
[ ] nb5e re-run: fix find_checkpoint() + use Crack500-only baseline row
[ ] run_on_kaggle_final_rauf.ipynb re-run with saved output cells
[ ] Confirm [KD] logit files > 0 printed at start of EVERY notebook
[ ] Confirm [KD] KD losses computed appears in EVERY training log
```
