# Crack-Distill 🛣️

**SAM 2 (Teacher, 224M) $\rightarrow$ YOLOv11n-seg (Student, 2.84M) Knowledge Distillation for Real-Time Pavement Crack Segmentation**

Crack-Distill is a high-performance, real-time pavement crack instance segmentation framework. It uses **Knowledge Distillation (KD)** to transfer structural detail, curvature continuity, and boundary probability uncertainty ("dark knowledge") from Meta's **Segment Anything Model 2 (SAM 2 Large)** into an ultra-lightweight, edge-deployable **YOLOv11n-seg** student model.

At deployment time, you run the **YOLOv11n-seg student alone**—achieving **over 100 FPS** inference with **zero dependency on SAM 2, zero additional parameters (2.84M), and 0% runtime latency overhead**, while outperforming standard fine-tuned models on both in-domain and out-of-distribution real-world road inspections.

---

## 🚀 Key Results & Highlights

* **In-Domain (Cropped Crack500)**: **`0.5500` Mask mAP50** ($+1.85\%$ gain over plain fine-tuning baseline).
* **Out-of-Distribution (Uncropped Crack500)**: **`0.1308` Mask mAP50-95** ($+10.3\%$ relative gain over baseline).
* **Edge Deployment**: **2.84M parameters, 10.2 GFLOPs, >100 FPS throughput** on edge GPUs.
* **Tiled Sliding-Window Inference**: Restores $2000 \times 1500$ resolution fidelity without downscaling loss, boosting OOD detection by $4\times$.

```
[ SAM 2 Large Teacher (224M) ]  ───>  Offline GPU Inference  ───>  Pre-computed Soft Logits (.npy)
                                                                                  │
                                                                           Batch Preloading
                                                                                  │
                                                                                  ▼
[ YOLOv11n-seg Student (2.84M) ] ───>  Forward Pass  ───>  KDSegmentationTrainer Patched Loss
                                                               ├── Task Loss (Box, Cls, Mask)
                                                               └── Soft Mask KL Divergence (τ = 3.7769, W = 0.9612)
```

---

## 📂 The Final Production & Research Suite (`final_notebooks/`)

Every notebook is **100% self-contained and decoupled**, embedding exact assertions, training loops, in-domain validation, and OOD evaluations.

| # | Notebook | Purpose & Recipe | Expected Runtime | Output Summary |
| :-: | :--- | :--- | :---: | :--- |
| **01** | **[`01_run_mask_kd_production_seed42.ipynb`](final_notebooks/01_run_mask_kd_production_seed42.ipynb)** | **Locked Baseline (Seed 42)**: Uniform Mask-KL ($\tau=3.7769, W=0.9612$, box prompts, FP32). | ~2.5–3.0 hrs | `results/prod_mask_kd_box_only_T3.7769_W0.9612_seed42_150ep.json` |
| **02** | **[`02_run_mask_kd_production_seed123.ipynb`](final_notebooks/02_run_mask_kd_production_seed123.ipynb)** | **Multi-Seed Verification (Seed 123)**: Variance testing for statistical defense. | ~2.5–3.0 hrs | `results/prod_mask_kd_box_only_T3.7769_W0.9612_seed123_150ep.json` |
| **03** | **[`03_run_foreground_dilated_kd.ipynb`](final_notebooks/03_run_foreground_dilated_kd.ipynb)** | **Research Variant 1 (Foreground-Dilated KL)**: Focuses gradient on crack core + 8px context band (eliminates 99% asphalt background dilution). | ~2.5–3.0 hrs | `results/exp_foreground_dilated_mask_kd_T3.7769_W0.9612_seed42_150ep.json` |
| **04** | **[`04_run_pixel_affinity_kd.ipynb`](final_notebooks/04_run_pixel_affinity_kd.ipynb)** | **Research Variant 2 (Spatial Pixel Affinity)**: Captures topological crack continuity via 4-directional spatial difference matching. | ~2.5–3.0 hrs | `results/exp_pixel_affinity_kd_T3.7769_W0.9612_seed42_150ep.json` |
| **05** | **[`05_run_multiscale_mask_kd.ipynb`](final_notebooks/05_run_multiscale_mask_kd.ipynb)** | **Research Variant 3 (512x512 High-Res Matching)**: Full $512 \times 512$ sub-pixel logit alignment. | ~2.5–3.0 hrs | `results/exp_multiscale_512_mask_kd_T3.7769_W0.9612_seed42_150ep.json` |
| **06** | **[`06_run_multiscale_layer_kd.ipynb`](final_notebooks/06_run_multiscale_layer_kd.ipynb)** | **Research Variant 4 (Multi-Scale Neck LayerKD)**: Channel-Wise Distillation (CWD) on PANet Neck layers (12, 15, 18). | ~2.8–3.2 hrs | `results/exp_multiscale_layer_cwd_kd_T3.7769_W0.9612_seed42_150ep.json` |
| **07** | **[`07_eval_ood_and_tiled_inference.ipynb`](final_notebooks/07_eval_ood_and_tiled_inference.ipynb)** | **Deployment Evaluation**: Compares direct resizing vs. tiled sliding-window inference ($512 \times 512$ native patches). | ~5–10 mins | `results/ood_eval_summary.json` |
| **08** | **[`08_benchmark_speed_and_profile.ipynb`](final_notebooks/08_benchmark_speed_and_profile.ipynb)** | **Speed Benchmark**: Confirms 0% latency/parameter overhead (>100 FPS, 2.84M params, 10.2 GFLOPs). | ~2 mins | Latency & Throughput Report |

---

## 📥 Kaggle Inputs & Hardware Reference

| Notebook | Kaggle Dataset | Model Weights | Accelerator | Internet |
| :--- | :--- | :--- | :---: | :---: |
| **01 – 06 (Training)** | `distill_datasetforme` (or Crack500 raw + teacher logits) | Auto-downloads `yolo11n-seg.pt` | **GPU T4 x2** or **P100** | **ON** |
| **07 (OOD & Tiled Eval)** | `distill_datasetforme` (uncropped `valdata`/`testdata`) | Attach output `best.pt` from Notebook 01–06 | **GPU** or **CPU** | **ON** |
| **08 (Speed Benchmark)** | *None needed* | Auto-downloads `yolo11n-seg.pt` or uses `best.pt` | **GPU** or **CPU** | **ON** |

---

## 🔬 Core Technical & Architectural Reports

* **[`good_review.md`](good_review.md)** — Master Technical Review & Execution Blueprint (KD literature analysis, why other KD components were rejected, Short-Horizon Optuna dynamics, and OOD resolution).
* **[`layerKD.md`](layerKD.md)** — Deep-dive research into Layer-by-Layer feature distillation (CWD, MGD, FFD, and ViT-to-CNN inductive bias compatibility).
* **[`layerKDarch.md`](layerKDarch.md)** — Architectural system design for intermediate feature distillation, multi-scale layer pairing matrix, and Cross-Architecture Projectors.
* **[`nb_exp_results.md`](nb_exp_results.md)** — Complete empirical results table across 27 experimental runs.

---

## ⚙️ Quickstart: Running on Kaggle

1. In Kaggle, click **New Notebook** $\rightarrow$ **File** $\rightarrow$ **Import Notebook** $\rightarrow$ select desired notebook from `final_notebooks/`.
2. Set Accelerator to **GPU T4 x2** or **P100**, and set Internet **ON**.
3. Click **+ Add Data** $\rightarrow$ search and attach `distill_datasetforme`.
4. Click **Run All**.