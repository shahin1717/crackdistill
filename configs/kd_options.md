# Knowledge Distillation Configurations Reference

Here are the different loss weight configurations referenced in the distillation tuning experiments:

## Option A: Literature-Aligned (Conservative)
Follows the most common patterns in computer vision segmentation KD literature.
* **temperature (τ)**: `3.0`
* **mask_kd (α)**: `0.5`
* **feature (β)**: `1.0`
* **boundary (γ)**: `0.5`

---

## Option B: Balanced-Aggressive
An aggressive set of loss scales balancing student task objectives and teacher soft targets.
* **temperature (τ)**: `3.0`
* **mask_kd (α)**: `1.5`
* **feature (β)**: `1.0`
* **boundary (γ)**: `1.0`

---

## Option C: Crack-Optimized Blend (Current Config)
Based directly on recent road crack and pavement segmentation distillation papers (equal logit + feature, reduced boundary weight to prevent edge-only overfitting).
* **temperature (τ)**: `2.5`
* **mask_kd (α)**: `1.0`
* **feature (β)**: `1.0`
* **boundary (γ)**: `0.5`

---

## [Reference] Original Optuna-Tuned Weights
The automated search configuration (which prioritized `mAP50-seg` exclusively and led to a boundary-heavy bias).
* **temperature (τ)**: `1.6502`
* **mask_kd (α)**: `0.1652`
* **feature (β)**: `0.1767`
* **boundary (γ)**: `2.0569`
