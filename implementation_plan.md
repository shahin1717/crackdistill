# Crack-Distill: Notebook Separation & Experiment Expansion Plan

## Background

Current state: one monolithic `run_on_kaggle_final.ipynb` that does everything — setup, logit generation, training, and evaluation — for the combined Crack500+DeepCrack dataset with YOLOv11n-seg. The new plan separates this into **focused, purpose-built Kaggle notebooks**, adds **dataset-level comparisons** (Crack500 vs DeepCrack), **architecture comparisons** (YOLOv8n-seg vs YOLOv11n-seg), and creates **formal academic documentation** in the vault.

---

## CRITIC Questions: Addressed in the Plan

> [!IMPORTANT]
> **Q: Train on Crack500, test on DeepCrack — overfitting?**  
> This is the **cross-dataset generalization test** (Notebook 4 below). We do NOT fine-tune on DeepCrack — we train exclusively on Crack500 (or DeepCrack) and evaluate on the other, treating it as a held-out OOD test. This measures domain transfer capability without any test-set leakage. The "overfitting" concern is specifically addressed by: (a) reporting in-domain val + OOD cross-dataset test separately, and (b) showing KD gap vs baseline on OOD data.

> [!IMPORTANT]
> **Q: Segmentation head freeze?**  
> Already partially done via Stage 1 of the progressive schedule (backbone frozen, head trains). The proposed new experiment: **reverse freeze** — freeze the head and only train the backbone + neck under KD for the full run. This isolates whether feature distillation alone (without head adaptation) is sufficient for generalization. Documented as a new ablation in Notebook 3.

---

## Proposed Notebooks

### Notebook 1: `nb1_baseline_comparison.ipynb`
**Goal**: Establish clean baselines — YOLOv8n-seg vs YOLOv11n-seg — on each dataset separately.

**Experiments:**
| ID | Model | Train Data | Val Data |
|---|---|---|---|
| `v8_crack500_baseline` | YOLOv8n-seg | Crack500 | Crack500 val |
| `v11_crack500_baseline` | YOLOv11n-seg | Crack500 | Crack500 val |
| `v8_deepcrack_baseline` | YOLOv8n-seg | DeepCrack | DeepCrack val |
| `v11_deepcrack_baseline` | YOLOv11n-seg | DeepCrack | DeepCrack val |

**Outputs**: mAP50-seg, mAP50-95-seg, FPS, GFLOPs table. No SAM 2 needed — pure YOLO fine-tune.

---

### Notebook 2: `nb2_crack500_kd.ipynb`
**Goal**: Full KD pipeline on Crack500 only (no DeepCrack in training).

**Experiments:**
| ID | Model | Config |
|---|---|---|
| `v8_c500_baseline` | YOLOv8n-seg | No KD |
| `v8_c500_full_kd_box` | YOLOv8n-seg | Full KD, box prompt |
| `v11_c500_baseline` | YOLOv11n-seg | No KD |
| `v11_c500_full_kd_box` | YOLOv11n-seg | Full KD, box prompt |
| `v11_c500_full_kd_centroid` | YOLOv11n-seg | Full KD, box+centroid |

**Outputs**: In-domain Crack500 val + OOD cross-test on DeepCrack (no DeepCrack in training).

---

### Notebook 3: `nb3_deepcrack_kd.ipynb`
**Goal**: Full KD pipeline on DeepCrack only.

**Experiments:**
| ID | Model | Config |
|---|---|---|
| `v8_dc_baseline` | YOLOv8n-seg | No KD |
| `v8_dc_full_kd_box` | YOLOv8n-seg | Full KD, box prompt |
| `v11_dc_baseline` | YOLOv11n-seg | No KD |
| `v11_dc_full_kd_box` | YOLOv11n-seg | Full KD, box prompt |
| `v11_dc_seghead_frozen_kd` | YOLOv11n-seg | **NEW**: Head-frozen KD (ablation) |

**Outputs**: In-domain DeepCrack val + OOD cross-test on Crack500.

---

### Notebook 4: `nb4_cross_dataset_generalization.ipynb`
**Goal**: Evaluate all trained models from Notebooks 2 & 3 on the opposite dataset. Answers the CRITIC question directly — does KD generalize across domains better than baseline?

**Structure**: This is an **evaluation-only** notebook — loads `.pt` weights from Notebooks 2 & 3 and runs inference on cross-dataset test splits.

**Result Table Format:**
| Model | Train | Test | mAP50-seg (ID) | mAP50-seg (OOD) | Δ vs Baseline |
|---|---|---|---|---|---|
| v8 baseline | Crack500 | DeepCrack | ... | ... | — |
| v8 KD | Crack500 | DeepCrack | ... | ... | +X% |
| v11 baseline | DeepCrack | Crack500 | ... | ... | — |
| v11 KD | DeepCrack | Crack500 | ... | ... | +X% |

---

### Notebook 5: `nb5_ablation_and_results.ipynb`
**Goal**: Results aggregation and ablation study — combines metrics from all notebooks.

**Ablations** (run on Crack500 KD):
- `ablation_no_mask_kd` — remove KL divergence loss
- `ablation_no_feature` — remove feature MSE loss
- `ablation_no_boundary` — remove boundary BCE loss
- `ablation_seghead_frozen` — **NEW**: keep head frozen through all stages

---

## Vault Documentation Plan

### New Notes to Create

#### `Efforts/Experiments/Runs/` (individual run logs)
Each run gets a stub created **before** it runs (fill in hypothesis), then updated **after** with results.

#### `Atlas/Academic/Ideas Log.md` — **[NEW]**
> [!IMPORTANT]
> This is the "failed or not failed idea" documentation the academic advisor asked for.

Format per idea:
```
## Idea: [Name]
- **Hypothesis**: What we expected
- **Status**: ✅ Worked | ❌ Failed | ⚠️ Partial
- **Evidence**: Metric delta or error trace
- **Why it worked/failed**: Root cause
- **What we learned**: Takeaway for the paper
```

Ideas to document:
| Idea | Status | Key Evidence |
|---|---|---|
| Global boundary loss (Phase 2) | ✅ Partial | +2.1 mAP50, but misaligned instances |
| Per-instance KL divergence (Phase 4) | ✅ Worked | +5.3% OOD gain |
| Dynamic stride-based feature mapping | ❌ Failed | Aspect-ratio padding broke stride estimates |
| Hardcoded index-based feature mapping | ✅ Worked | Stable, no mismatch warnings |
| Raw boundary BCE without clamping | ❌ Failed | NaN crash at epoch 42–44 under AMP |
| Clamped boundary BCE `[-30, 30]` | ✅ Worked | Stable 150-epoch run |
| Pre-registered hooks (without DDP check) | ❌ Failed | Hooks stripped by DDP wrapper |
| Dynamic hook re-registration | ✅ Worked | Feature distillation works in DDP |
| Stale validation cache not cleared | ❌ Failed | Shape mismatch crash on resume |
| Explicit cache clear in `preprocess_batch` | ✅ Worked | Stable batch shapes |

#### `Atlas/Academic/Experiment Log.md` — **[NEW]**
> [!IMPORTANT]
> This is the formal "experiment log" the advisor asked for — a chronological table of all runs.

Format:
```
| Run ID | Date | Notebook | Dataset | Model | Config | Epochs | mAP50-seg (ID) | mAP50-seg (OOD) | Notes |
```

---

## Proposed Changes

### Notebooks (New Files)

#### [NEW] `nb1_baseline_comparison.ipynb`
- Setup cell (dirs, installs, dataset links)
- Dataset conversion (Crack500 + DeepCrack separately, no combine)
- 4 training cells: v8/v11 × crack500/deepcrack baseline
- Evaluation + results table cell
- **No SAM 2** — pure YOLO, fast to run

#### [NEW] `nb2_crack500_kd.ipynb`
- Full setup with SAM 2 install
- Crack500-only data conversion + logit generation (box + box_centroid)
- 5 training cells (v8 baseline, v8 KD box, v11 baseline, v11 KD box, v11 KD centroid)
- Dual eval: Crack500 val + DeepCrack cross-test

#### [NEW] `nb3_deepcrack_kd.ipynb`
- Full setup with SAM 2 install
- DeepCrack-only data conversion + logit generation
- 5 training cells (v8 baseline, v8 KD box, v11 baseline, v11 KD box, v11 head-frozen KD)
- Dual eval: DeepCrack val + Crack500 cross-test

#### [NEW] `nb4_cross_dataset_generalization.ipynb`
- Eval-only notebook (no training)
- Loads `.pt` model weights from Kaggle output datasets
- Runs inference on cross-dataset test splits
- Produces the main cross-dataset comparison table

#### [NEW] `nb5_ablation_and_results.ipynb`
- Ablation runs (3 existing + 1 new head-frozen)
- Results aggregation from all prior notebooks
- Final comparison table + plots

### Vault Notes (New Files)

#### [NEW] `Atlas/Academic/Ideas Log.md`
Structured log of every idea tried — failed or not — with hypothesis, status, evidence, and lesson.

#### [NEW] `Atlas/Academic/Experiment Log.md`
Chronological table of every training run — run ID, date, dataset, model, config, epochs, metrics, notes.

#### [MODIFY] `Atlas/Academic/Academic Report.md`
Add Section 9: Cross-Dataset Generalization (new experiment) and Section 10: Ablation Study.

---

## Open Questions

> [!WARNING]
> **Do you want `nb1_baseline_comparison.ipynb` to use 150 epochs (same as the paper run) or a shorter demo run (e.g., 50 epochs) to save Kaggle GPU time?**

> [!WARNING]
> **For the segmentation head freeze ablation in Notebook 3: should we freeze the head for the ENTIRE training run, or just apply a 3-stage schedule (backbone KD → full KD → head unfreeze)?**

> [!WARNING]
> **For cross-dataset eval (Notebook 4): should the Crack500 test split used for OOD evaluation be the "cropped" or "uncropped" variant? (Uncropped is harder and more realistic but the DeepCrack images are also cropped — recommend using cropped for apples-to-apples.)**

---

## Verification Plan

### Automated Tests
- Each notebook has a final "Results Table" cell that fails loudly if any model `.pt` weight is missing
- Cross-dataset notebook verifies that test images come from the opposite dataset (path-based assertion)

### Manual Verification
- Run `nb1_baseline_comparison.ipynb` first (no SAM 2, fast) to confirm dataset pipeline works cleanly for both datasets separately
- Confirm v8 baseline metrics on Crack500 are close to existing `0.5249 mAP50-seg` from prior run (regression check)
