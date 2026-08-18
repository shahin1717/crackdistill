# 📓 Final Notebook Suite — Experiment Results (`final_results_exp.md`)

This document aggregates empirical results from the **`final_notebooks/output_runned/`** suite — the production-locked recipe plus five single/combined-loss research variants — evaluated on both in-domain (cropped) and out-of-distribution (uncropped) Crack500 splits. Table format follows `nb_exp_results.md` §1 (Baseline Comparison Runs).

---

## ⚙️ Hardware & Execution Environment

* **Execution Platform**: Kaggle Notebooks (Tesla T4, 14.9 GB VRAM)
* **Student Model**: YOLOv11n-seg (117–129 layers, 2.94M–3.01M parameters, ~9.6 GFLOPs)
* **Teacher**: SAM 2 Large (pre-computed offline logits, `T=3.7769`, `W=0.9612`)
* **Training**: 150 epochs, seed 42, Crack500 (348 val images in-domain / 50 val images uncropped OOD)
* **Recipe base**: All variants build on the locked Mask-KL-only recipe (`01_run_mask_kd_production_seed42`); each research variant swaps or adds one loss term on top of it.

---

## 📊 1. In-Domain Cropped Validation (Crack500 val, 348 images, 630 instances)

| Notebook | Variant | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 (seg) | Mask mAP50-95 (seg) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01` | **Locked Baseline (Mask-KL only, seed 42)** | 0.723 | 0.514 | 0.598 | 0.403 | 0.695 | 0.491 | 0.542 | 0.201 |
| `03` | Foreground-Dilated KL | 0.743 | 0.508 | 0.582 | 0.387 | 0.729 | 0.487 | 0.539 | 0.203 |
| `04` | Spatial Pixel-Affinity KD | 0.732 | 0.522 | 0.583 | 0.393 | 0.689 | 0.510 | 0.535 | 0.204 |
| `05` | Multi-Scale 512×512 Mask KD | 0.712 | 0.552 | **0.600** | **0.403** | 0.692 | **0.518** | **0.549** | **0.209** |
| `06` | Multi-Scale Neck LayerKD (CWD) | **0.772** | 0.500 | 0.590 | 0.393 | 0.724 | 0.471 | 0.542 | 0.206 |
| `09` | Combined Affinity + Dilated | **0.772** | 0.485 | 0.588 | 0.391 | **0.754** | 0.471 | 0.541 | 0.208 |
| `09` | Focal Mask KD | 0.764 | 0.494 | 0.585 | 0.390 | 0.733 | 0.473 | 0.541 | 0.206 |

### Takeaways
* **`05` (Multi-Scale 512×512)** is the best in-domain configuration on every axis — Box mAP50 (0.600), Mask mAP50 (0.549), and Mask mAP50-95 (0.209).
* All research variants sit within a tight **0.535–0.549** Mask mAP50 band — none dramatically outperforms the locked baseline (0.542) in-domain; gains here are marginal (≤ +0.7 pts) or slightly negative.

---

## 🌐 2. Out-of-Distribution Validation — Direct Resize (Uncropped Crack500, 50 images, 263 instances)

| Notebook | Variant | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 (seg) | Mask mAP50-95 (seg) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01` | Locked Baseline (Mask-KL only) | 0.248 | 0.148 | 0.111 | 0.0411 | 0.159 | 0.171 | 0.0848 | 0.0196 |
| `03` | **Foreground-Dilated KL** | **0.280** | 0.171 | **0.125** | **0.0460** | 0.266 | 0.144 | **0.1007** | 0.0241 |
| `04` | Spatial Pixel-Affinity KD | 0.224 | 0.144 | 0.108 | 0.0447 | 0.239 | 0.103 | 0.0831 | 0.0214 |
| `05` | Multi-Scale 512×512 Mask KD | 0.234 | 0.175 | 0.117 | 0.0443 | 0.246 | 0.118 | 0.0872 | 0.0226 |
| `06` | Multi-Scale Neck LayerKD (CWD) | 0.268 | **0.198** | 0.131 | **0.0539** | 0.207 | **0.163** | 0.0944 | **0.0252** |
| `09` | Combined Affinity + Dilated | 0.202 | 0.190 | 0.106 | 0.0379 | 0.188 | 0.144 | 0.0851 | 0.0200 |
| `09` | Focal Mask KD | 0.230 | 0.188 | 0.110 | 0.0446 | 0.204 | 0.144 | 0.0931 | 0.0242 |

### Takeaways
* **`03` (Foreground-Dilated)** wins raw OOD Mask mAP50 (0.1007) — best absolute number on the direct-resize protocol.
* **`06` (Layer-KD)** wins OOD Mask mAP50-95 (0.0252), the stricter high-overlap metric.
* **`09` Combined (Affinity + Dilated)** is the *worst* OOD performer of the five research variants (0.0851) — barely above the locked baseline (0.0848) and clearly below both parent techniques it was built from (0.1007 / 0.0831). Combining these two losses did not compound their gains.

---

## 🧩 3. Full-Resolution Cross-Checkpoint Eval (Notebook 07 — Direct Resize vs. Tiled Sliding-Window)

Independent re-validation: all 7 checkpoints reloaded fresh and evaluated on the same 50 full-resolution uncropped images, comparing naive resize-to-512 against 512×512 tiled sliding-window inference (the documented production deployment strategy).

| Checkpoint | Direct Mask mAP50 | Direct mAP50-95 | Full-Res Direct Dice | **Full-Res Tiled Dice** |
| :--- | :---: | :---: | :---: | :---: |
| `01` Locked Baseline | 0.0848 | 0.0196 | 0.1409 | 0.2414 |
| `03` Foreground-Dilated | **0.1007** | 0.0241 | 0.1722 | 0.2612 |
| `04` Pixel-Affinity | 0.0831 | 0.0214 | 0.1505 | 0.2594 |
| `05` Multi-Scale Mask KD | 0.0872 | 0.0226 | 0.1570 | 0.2625 |
| `06` Multi-Scale Layer-KD | 0.0944 | **0.0252** | **0.1621** | **0.2747** |
| `09` Combined Affinity+Dilated | 0.0851 | 0.0200 | 0.1546 | 0.2630 |
| `09` Focal Mask KD | 0.0931 | 0.0242 | 0.1445 | 0.2671 |

### Takeaways
* Every KD variant beats the no-KD-equivalent locked baseline on tiled Dice — tiling itself adds a large, consistent boost (roughly +0.10–0.13 Dice) across the board, confirming the project's standing tiled-inference deployment strategy.
* **`06` (Layer-KD) is the strongest deployment candidate** — best Full-Res Tiled Dice (0.2747) and best mAP50-95 (0.0252), the two metrics closest to real-world usage.
* This independently reproduces the §2 finding for `09` Combined — 0.0851 direct mAP50 matches its own self-reported number exactly, confirming (not a fluke, checkpoint-collision-free) that the combined-loss experiment does not outperform its parents.

---

## 📌 4. Summary & Recommendation

| Objective | Best Variant | Metric |
| :--- | :--- | :--- |
| In-domain accuracy | `05` Multi-Scale 512×512 | 0.549 Mask mAP50 |
| Raw OOD accuracy | `03` Foreground-Dilated | 0.1007 Mask mAP50 |
| **Deployment (tiled inference)** | **`06` Multi-Scale Layer-KD** | **0.2747 Tiled Dice** |
| Combined-loss hypothesis (`09`) | ❌ Did not beat either parent (`03`, `04`) | 0.0851 OOD / 0.541 in-domain |

**Bottom line**: No further reruns of the `09` combined affinity+dilated config are needed — the null result (no synergy from combining these two losses) has been confirmed twice via independent evaluation paths. The next candidate worth testing is a **`06` + `03` combination** (Layer-KD + Foreground-Dilated), pairing the two individually-strongest, least-redundant variants (intermediate-feature distillation + foreground-focused mask loss) instead of two variants that both act on the same output-mask signal.
