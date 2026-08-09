# ✅ What To Do — Crack-Distill Action Plan
*Reconciled from: `crackdistill_nb5_bug_rootcause_and_plan.md`, `crackdistill_next_steps.md`, `crackdistill_next_steps_v2_post_nb5e.md`, and current code audit.*
*Last updated: 2026-08-09*

---

## 🔍 First: What Claude Got Right vs Wrong

> [!IMPORTANT]
> Claude's 3 files contain some **outdated** diagnoses. The current `build_all_notebooks.py` has already been partially fixed. Here is the ground truth after reading the actual code:

| Claude's Claim | Reality (Code-Verified) |
|---|---|
| "nb5a has no `generate_teacher_logits` call" | **Wrong** — `crack500_logits_generation_cell` IS included in nb5a/b/c (lines 494–511, 523, 562, 601). The script already generates logits if missing. |
| "nb5a calls `KDSegmentationTrainer(cfg)` with no `logits_dir`" | **Check needed** — `override_config` sets `teacher.logits_dir` but we need to verify `KDSegmentationTrainer.__init__` reads it (not env var fallback). |
| "nb5e checkpoint resolver fails via keyword fallback" | ✅ **Confirmed** — `find_checkpoint()` lines 827–851 falls through to keyword glob search, causing the collision. This IS the real bug. |
| "nb5a–e ablation results are measuring nothing" | **Partially wrong** — nb5a/b/c likely ran real KD. Collision affected only rows 1 and 6. Rows 3, 4, 5 (nb5a/b/c) appear valid. |

---

## 🚨 Priority 0 — Must do before trusting any number

### P0-A: Verify `KDSegmentationTrainer` reads `logits_dir` from config

**Why**: nb5a–d pass `teacher.logits_dir` via `override_config()`. Must confirm `KDSegmentationTrainer.__init__` reads that key, not an env var fallback.

**Action** (5 min, local):
```bash
cd /home/shahin/distill
python3 -c "
from utils.config_loader import load_config, override_config
from distillation.kd_trainer import KDSegmentationTrainer
import inspect
cfg = load_config('configs/config.yaml')
cfg = override_config(cfg, {'teacher.logits_dir': 'data/teacher_logits_box/'})
print(inspect.getsource(KDSegmentationTrainer.__init__))
"
```

**If broken**: Fix `__init__` to read from `cfg.teacher.logits_dir`. This is the "second stacked bug" from `crackdistill_nb5_bug_rootcause_and_plan.md` §1.

**Diagnostic to add to each ablation notebook BEFORE training**:
```python
from pathlib import Path
logits_dir = Path("data/teacher_logits_box")
n = len(list(logits_dir.glob("*_logits.npy")))
assert n > 0, f"[KD] FATAL: 0 logit files in {logits_dir}. Aborting."
print(f"[KD] logit files: {n}  ← must be > 0 or training is broken")
```

---

### P0-B: Fix the `find_checkpoint()` fallback in build_all_notebooks.py

**What's broken**: `find_checkpoint()` at lines 827–851 falls back to keyword glob when explicit paths don't exist. `"seghead_frozen"` matched the nb3-DeepCrack checkpoint before the nb5d checkpoint, corrupting rows 1 and 6.

**Fix** — edit `scripts/build_all_notebooks.py`:
```python
def find_checkpoint(name, keywords, candidates):
    for cand in candidates:
        if Path(cand).exists():
            return cand
        abs_cand = Path("/kaggle/working") / cand
        if abs_cand.exists():
            return str(abs_cand)
    # REMOVED: keyword glob fallback — silently picks wrong checkpoints
    print(f"WARNING: checkpoint for '{name}' not found. Searched: {candidates}")
    return None
```

**Then rebuild**:
```bash
cd /home/shahin/distill
python3 scripts/build_all_notebooks.py
```

---

### P0-C: Re-run nb5d (Full SegHead Freeze) on Kaggle

Row 6 resolved to the nb3-DeepCrack checkpoint — not nb5d's output. Re-run clean.

**Steps**:
1. `python3 scripts/build_all_notebooks.py` (after P0-B fix)
2. Upload `kaggle_notebooks/nb5d_ablation_seghead_frozen.ipynb` to Kaggle
3. Attach `distill_datasetforme` dataset
4. Run all cells — verify prints `[KD] logit files: N > 0`
5. Save output, note the Kaggle run path for nb5e

---

### P0-D: Get a valid Crack500 Baseline row

Row 0 used a DeepCrack-trained model out-of-domain — the "+156%" comparison is fake.

**Fix**: Use the nb1/nb2 no-KD Crack500 checkpoint (`crack500_baseline/weights/best.pt`).
Hardcode its explicit path into `ablation_runs[0]` candidates in `build_all_notebooks.py`.

---

## 🟡 Priority 1 — After P0 is clean

### P1-A: Run "task + mask_kd only" (nb5f) — likely your best model

**Why**: From valid ablation rows:
- Removing Feature MSE → +2.14% Mask mAP50 (0.4921 vs 0.4818)
- Removing Boundary BCE → +0.79% Mask mAP50 (0.4897 vs 0.4818)
- The implied winner — mask_kd only — has **never been run**.

**Add nb5f to `build_all_notebooks.py`** with config:
```python
"distillation.losses.mask_kd.enabled": True,
"distillation.losses.feature.enabled": False,
"distillation.losses.boundary.enabled": False,
"project.experiment": "ablation_mask_kd_only",
```

**If nb5f beats 0.4921**: your paper method is 1-loss KD. This is your headline ablation finding.

---

### P1-B: Unify the eval protocol

Three incompatible protocols exist. Pick one: **per-dataset val splits** (Crack500 val + DeepCrack val separately). Update nb5e to output multi-split results as `next_moves.md` §1.2 specifies.

---

### P1-C: Spot-verify nb5a/b/c checkpoint paths

Confirm resolved paths in the saved nb5e output point to `/kaggle/working/runs/` (training outputs), NOT `/kaggle/input/` (preloaded). Check file sizes (>10MB expected for trained YOLO weights).

---

## 🟢 Priority 2 — Needed for submission-ready

### P2-A: Null-teacher control

Replace SAM2 logits with random Gaussian noise. If OOD still improves → mechanism is "any regularizer," not "SAM2 knowledge" → abstract must change.

### P2-B: Confirm +10.3% OOD claim has an artifact

`run_on_kaggle_final_rauf.ipynb` has **zero saved output cells**. Re-run on Kaggle and save, or trace where 0.352 OOD mAP50-95 came from (log file, screenshot).

### P2-C: 3-seed reruns of 2 key comparisons

Once stable, run seeds [42, 123, 456] for:
1. Crack500 baseline vs mask_kd-only KD
2. DeepCrack 2-stage freeze vs no-freeze
Report mean ± std — these deltas are within plausible seed variance.

---

## ❄️ The Freezing Question — Deep Dive

*You asked: "do I need freezing? because without it it worked better..."*

### What the data shows

| Condition | Dataset | Result |
|---|---|---|
| No freeze (Crack500, nb2) | Crack500 ~2700 imgs | Mask mAP50 ~0.52 — no-freeze is fine |
| 2-stage progressive freeze (nb3) | DeepCrack ~537 imgs | +3.98% over no-freeze — strongest finding |
| Full-run freeze (nb5d) | DeepCrack | ⚠️ Checkpoint corrupted — unknown |
| No freeze (nb5a/b/c) | Crack500 combined | 0.4762–0.4921 — works fine |

### The honest answer: freezing is dataset-size-dependent

- **Crack500 (~2700 images)**: Enough data. No-freeze works. Feature MSE over-constrains thin cracks. → **Don't freeze. Drop Feature MSE.**
- **DeepCrack (~537 images)**: Head overfits to backbone noise on small data. 2-stage progressive freeze (30% frozen → 70% unfrozen) rescued training. → **2-stage freeze for small datasets.**

### Claude's valid point: the confound

The freeze comparison has always been across different datasets — never the same data at different sizes. The missing experiment:
- Subsample Crack500 to ~537 images → run freeze vs no-freeze
- If freeze helps → "sample size" explanation (general, citable)
- If not → "DeepCrack-specific" (narrower claim)

**Recommendation**: For finishing now, claim "2-stage progressive freeze stabilizes KD on small datasets (< ~600 images)." The nb3 finding supports this. Don't overclaim.

---

## 🤔 SAM2 + YOLO Logits — Ensemble or Per-Layer?

*You asked: "it is using logits of SAM, then logits of YOLO and ensembles at the end, or at each layer?"*

### What the code actually does

```
SAM2 runs OFFLINE (before training begins)
       ↓
Generates per-instance soft mask logit .npy files (stored on disk)
       ↓
KDYOLODataset loads them as batch targets during training
       ↓
KDSegmentationTrainer.patched_loss() during each training step:
    student_preds = YOLO forward pass (normal, no SAM2 involved)
    kd_loss = _kd_loss_from_preds(student_preds, sam2_logits):
        ├── mask_kd:   KL(YOLO_mask_logits || SAM2_mask_logits)   ← FINAL layer output
        ├── feature:   MSE(YOLO_backbone_feats, SAM2_encoder_feats) ← INTERMEDIATE layers
        └── boundary:  BCE(YOLO_edge_map, SAM2_edge_map)            ← FINAL layer (derived)
```

**NO online ensemble.** SAM2 is a frozen offline teacher. At test time, only YOLO runs.

### Per-layer or final-layer?

| Loss term | Where |
|---|---|
| **Mask KL** | Final YOLO seg head output vs SAM2 mask logits → **final output layer** |
| **Feature MSE** | Intermediate backbone features (via ActiveHook) vs SAM2 encoder features → **per intermediate layer** |
| **Boundary BCE** | Derived edge map from YOLO mask vs SAM2 mask edge → **final layer (post-processed)** |

### What would be better?

| Approach | Better for cracks? |
|---|---|
| Offline KL at final output (current mask_kd) | ✅ Works — proven by your own data |
| Per-layer feature MSE (current) | ❌ Your ablation says this HURTS (+2.14% removed) |
| Attention Transfer (AT) loss | 🔶 Better than raw MSE for large capacity gaps (SAM2=224M vs YOLO=2.84M) |
| Relational KD (RKD) | 🔶 Preserves pairwise instance structure — not capacity-limited |
| Online ensemble (SAM2 + YOLO at test time) | ❌ Defeats the purpose — SAM2 is slow |

**Bottom line**: Offline SAM2 logits → final output KL is the right design. Drop Feature MSE (your data says so). Optionally try Attention Transfer as a P2 experiment if you want to replace it with something better.

---

## 📋 Master Execution Checklist

```
LOCAL (run on your machine first):
[ ] 1. Check KDSegmentationTrainer reads logits_dir from cfg (not env var fallback)
[ ] 2. Fix find_checkpoint() in build_all_notebooks.py (remove keyword glob fallback)
[ ] 3. python3 scripts/build_all_notebooks.py  ← rebuilds all notebooks

ON KAGGLE (run in this order):
[ ] 4. Run nb5d fresh (Full SegHead Freeze on DeepCrack) — save output
[ ] 5. Run nb5f NEW (task + mask_kd only on Crack500) — most important run
[ ] 6. Re-run run_on_kaggle_final_rauf.ipynb and SAVE OUTPUT  ← confirms +10.3% OOD
[ ] 7. Attach nb2, nb3, nb5a, nb5b, nb5c, nb5d, nb5f outputs → run nb5e
[ ] 8. Verify all rows have DISTINCT checkpoint paths (no two rows identical)
[ ] 9. Verify Baseline row uses Crack500-trained no-KD checkpoint (not DeepCrack)

WRITING (after Kaggle results):
[ ] 10. Update DistillVault/nb_exp_results.md with final ablation table
[ ] 11. Update Academic Report: "mask_kd drives gains; feature MSE over-constrains thin cracks"
[ ] 12. If +10.3% reproduced: keep headline. If not: downgrade to "preliminary observation"
```

---

## 🎯 Minimum Viable Paper-Ready State

| # | Status | Item |
|---|---|---|
| nb1–nb4 | ✅ Done | Baselines + main KD results — solid |
| nb5a/b/c | ✅ Done | Valid: mask_kd matters, feature MSE hurts |
| nb5d | ❌ Re-run | Checkpoint was corrupted — 1 Kaggle run |
| **nb5f (new)** | **❌ Run** | **Task + mask_kd only — likely your best model** |
| nb5e | ❌ Re-run | After fixing resolver + after nb5d + nb5f |
| final OOD run | ❌ Re-run | Save output to confirm +10.3% headline |

**= 4 Kaggle runs + 1 local script fix** to call this paper-ready.
