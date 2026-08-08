# 🚀 Next Moves & Action Plan (`next_moves.md`)

This document outlines the strategic action plan, technical fixes, loss function refinements, and documentation updates for the **Crack-Distill** (SAM 2 $\rightarrow$ YOLOv11n-seg Knowledge Distillation) framework following the post-logit fix ablation review (`reviews-analysis2.md`).

---

## 🎯 Phase 1: Immediate Bug Fixes & Evaluation Cleanup

### 1.1 Fix Checkpoint Resolver in `nb5e` Summary Notebook
* **Problem**: `"Full KD (Box Prompts)"` and `"Ablation 4: Full SegHead Freeze (nb5d)"` both fell back to the same DeepCrack stage 2 checkpoint due to broad keyword matching in `find_checkpoint()`.
* **Action**:
  - Update `ablation_runs` in `kaggle_notebooks/nb5e_ablation_results_summary.ipynb` and `scripts/generate_self_contained_notebooks.py` to use explicit hardcoded directory paths matching Kaggle notebook outputs:
    ```python
    ablation_runs = [
        ("Baseline (No KD)", ["runs/crack500_baseline/weights/best.pt"]),
        ("Full KD (Box Prompts)", ["runs/nb2_full_kd_box/weights/best.pt"]),
        ("Full KD (Box + Centroid)", ["runs/nb2_full_kd_centroid/weights/best.pt"]),
        ("w/o Mask KL (nb5a)", ["runs/nb5a_ablation_no_mask_kd/weights/best.pt"]),
        ("w/o Feature MSE (nb5b)", ["runs/nb5b_ablation_no_feature/weights/best.pt"]),
        ("w/o Boundary BCE (nb5c)", ["runs/nb5c_ablation_no_boundary/weights/best.pt"]),
        ("Full SegHead Freeze (nb5d)", ["runs/nb5d_ablation_seghead_frozen/weights/best.pt"]),
    ]
    ```

### 1.2 Multi-Split Validation Protocol in `nb5e`
* **Problem**: Evaluating all models exclusively on the combined 408-image validation set mixes dataset-specific fine-tuned weights with multi-dataset weights, causing lower baseline numbers.
* **Action**: Update `nb5e` to output a 3-column comparative summary table:
  1. **Crack500 Val Set** (348 images)
  2. **DeepCrack Val Set** (60 images)
  3. **Combined Val Set** (408 images)

---

## 🔬 Phase 2: Loss Function Refinement & Strategy Tuning

### 2.1 Re-weighting Distillation Loss Components
Based on valid ablation evidence from `reviews-analysis2.md`:
* **Mask KL Divergence ($\beta = 1.8658$)**: **Primary Driver**. Soft mask probabilities from SAM 2 transfer essential spatial uncertainty boundaries. Keep active across all runs.
* **Feature MSE ($\gamma = 0.8055$)**: **Over-constraining**. Removing feature MSE improved Mask mAP50 by **+2.14%** (`0.4921` vs `0.4818`).
  - *Recommendation*: Reduce $\gamma$ weight from `0.8055` $\rightarrow$ `0.10 - 0.20`, or apply MSE feature matching solely to the final SPPF/bottleneck layer rather than all 3 backbone stages.
* **Boundary BCE**: Neutral to slightly noisy on complex crack edges. Retain as optional flag in `configs/config.yaml`.

### 2.2 Finalizing the Recommended Training Recipe
```yaml
distillation:
  enabled: true
  temperature: 3.7769
  losses:
    task_aligned: {enabled: true, weight: 0.9612}
    mask_kl: {enabled: true, weight: 1.8658}
    feature: {enabled: false, weight: 0.0}       # Disabled or low-weight based on nb5b finding
    boundary: {enabled: false, weight: 0.0}      # Disabled based on nb5c finding
```

---

## 📚 Phase 3: Codebase Validation & Vault Synchronization

### 3.1 Rebuild Notebooks & Verify Pipeline
* **Command**: Run notebook generation script to ensure all self-contained Kaggle notebooks have identical logic and picklable `ActiveHook` classes:
  ```bash
  python3 scripts/build_all_notebooks.py
  ```
* Verify clean build output in `/home/shahin/distill/kaggle_notebooks/`.

### 3.2 DistillVault Knowledge Synchronization
Update key notes in `/mnt/c/Vaults/DistillVault/`:
1. **`nb_exp_results.md`**: Update Section 7 with the corrected `nb5e` ablation findings table.
2. **`Home Base.md`**: Link `reviews-analysis2.md` and `next_moves.md` under Quick Links.
3. **`Atlas/Academic/Academic Report.md`**: Update Section 4 (Ablation Analysis) to highlight the key empirical takeaway: *SAM 2 Soft Mask KL divergence drives performance gains, while strict backbone Feature MSE introduces over-regularization on thin crack topologies.*

---

## 📋 Executive Action Checklist

- [ ] Modify `scripts/generate_self_contained_notebooks.py` to fix checkpoint resolution in `nb5e`.
- [ ] Run `python3 scripts/build_all_notebooks.py` to regenerate all 6 Kaggle notebooks.
- [ ] Execute `nb5e` to produce the finalized multi-split ablation table.
- [ ] Update `DistillVault/nb_exp_results.md` with final results.
- [ ] Update `DistillVault/Atlas/Academic/Academic Report.md` draft for publication.
