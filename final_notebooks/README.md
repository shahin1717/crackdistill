# 🚀 Crack-Distill: Complete Production & Research Suite

This folder contains the complete, self-contained suite of Kaggle notebooks covering our **locked production recipe** and **advanced research candidates**.

---

## 📂 Notebook Suite Directory

| Notebook | Purpose & Recipe | Expected Runtime | Target Output |
| :--- | :--- | :---: | :--- |
| **`01_run_mask_kd_production_seed42.ipynb`** | **Locked Baseline (Seed 42)**: Uniform Mask-KL ($\\tau=3.7769, W=0.9612$, box prompts). | ~2.5–3.0 hrs | `results/prod_mask_kd_box_only_T3.7769_W0.9612_seed42_150ep.json` |
| **`02_run_mask_kd_production_seed123.ipynb`** | **Multi-Seed Verification (Seed 123)**: Statistical variance test. | ~2.5–3.0 hrs | `results/prod_mask_kd_box_only_T3.7769_W0.9612_seed123_150ep.json` |
| **`03_run_foreground_dilated_kd.ipynb`** | **Research Variant 1 (Foreground-Dilated KL)**: Focuses gradient on crack core + 8px context band (solves 99% asphalt background dilution). | ~2.5–3.0 hrs | `results/exp_foreground_dilated_mask_kd_T3.7769_W0.9612_seed42_150ep.json` |
| **`04_run_pixel_affinity_kd.ipynb`** | **Research Variant 2 (Spatial Pixel Affinity)**: Captures topological crack continuity via 4-directional spatial difference matching. | ~2.5–3.0 hrs | `results/exp_pixel_affinity_kd_T3.7769_W0.9612_seed42_150ep.json` |
| **`05_run_multiscale_mask_kd.ipynb`** | **Research Variant 3 (512x512 High-Res Matching)**: Full $512 \\times 512$ sub-pixel logit alignment. | ~2.5–3.0 hrs | `results/exp_multiscale_512_mask_kd_T3.7769_W0.9612_seed42_150ep.json` |
| **`06_run_multiscale_layer_kd.ipynb`** | **Research Variant 4 (Multi-Scale Neck LayerKD)**: Intermediate Channel-Wise Distillation (CWD) on PANet Neck layers (12, 15, 18). | ~2.8–3.2 hrs | `results/exp_multiscale_layer_cwd_kd_T3.7769_W0.9612_seed42_150ep.json` |
| **`07_eval_ood_and_tiled_inference.ipynb`** | **OOD & Tiled Inference Engine**: Evaluates checkpoints on uncropped images with direct resizing vs tiled sliding window ($512 \\times 512$ native patches). | ~5–10 mins | `results/ood_eval_summary.json` |
| **`08_benchmark_speed_and_profile.ipynb`** | **Speed Benchmark**: Confirms 0% latency/parameter overhead (>100 FPS, 2.84M params, 10.2 GFLOPs). | ~2 mins | Latency & FPS Report |
| **`09_run_combined_affinity_dilated_kd.ipynb`** | **Research Variant 5 (Combined Affinity + Dilated)**: Fuses `03_dilated` (focused 8px mask band) + `04_affinity` (4-way spatial continuity). | ~2.5–3.0 hrs | `results/exp_combined_affinity_dilated_kd_T3.7769_W0.9612_seed42_150ep.json` |

---

## 📥 Exact Kaggle Inputs & Hardware Mapping Table

| Notebook File | Required Kaggle Dataset | Required Model Checkpoint | Accelerator Setting | Internet | How to Run in Kaggle |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`01_run_mask_kd_production_seed42.ipynb`** | `distill_datasetforme` (Crack500 raw + teacher logits) | *None* (trains automatically from standard pre-trained YOLOv11) | **GPU T4 x2** or **P100** | **ON** | 1. Click **+ Add Data** $\rightarrow$ attach `distill_datasetforme`<br>2. Click **Run All** |
| **`02_run_mask_kd_production_seed123.ipynb`** | `distill_datasetforme` (Crack500 raw + teacher logits) | *None* (trains automatically from standard pre-trained YOLOv11) | **GPU T4 x2** or **P100** | **ON** | 1. Click **+ Add Data** $\rightarrow$ attach `distill_datasetforme`<br>2. Click **Run All** |
| **`03_run_foreground_dilated_kd.ipynb`** | `distill_datasetforme` (Crack500 raw + teacher logits) | *None* (trains automatically from standard pre-trained YOLOv11) | **GPU T4 x2** or **P100** | **ON** | 1. Click **+ Add Data** $\rightarrow$ attach `distill_datasetforme`<br>2. Click **Run All** |
| **`04_run_pixel_affinity_kd.ipynb`** | `distill_datasetforme` (Crack500 raw + teacher logits) | *None* (trains automatically from standard pre-trained YOLOv11) | **GPU T4 x2** or **P100** | **ON** | 1. Click **+ Add Data** $\rightarrow$ attach `distill_datasetforme`<br>2. Click **Run All** |
| **`05_run_multiscale_mask_kd.ipynb`** | `distill_datasetforme` (Crack500 raw + teacher logits) | *None* (trains automatically from standard pre-trained YOLOv11) | **GPU T4 x2** or **P100** | **ON** | 1. Click **+ Add Data** $\rightarrow$ attach `distill_datasetforme`<br>2. Click **Run All** |
| **`06_run_multiscale_layer_kd.ipynb`** | `distill_datasetforme` (Crack500 raw + teacher logits) | *None* (trains automatically from standard pre-trained YOLOv11) | **GPU T4 x2** or **P100** | **ON** | 1. Click **+ Add Data** $\rightarrow$ attach `distill_datasetforme`<br>2. Click **Run All** |
| **`07_eval_ood_and_tiled_inference.ipynb`** | `distill_datasetforme` (contains uncropped `valdata`/`testdata`) | **Attach Notebook 01-06 Output** (`best.pt`) via Kaggle "+ Add Data" $\rightarrow$ "Your Work / Notebook Output Files" | **GPU** (any) or **CPU** | **ON** | 1. Attach dataset + output `best.pt`<br>2. Click **Run All** |
| **`08_benchmark_speed_and_profile.ipynb`** | **None!** (benchmarks with synthetic tensors) | **None!** (auto-downloads `yolo11n-seg.pt` or uses trained `best.pt`) | **GPU** (T4 / P100) or **CPU** | **ON** | 1. No dataset needed<br>2. Click **Run All** |
| **`09_run_combined_affinity_dilated_kd.ipynb`** | `distill_datasetforme` (Crack500 raw + teacher logits) | *None* (trains automatically from standard pre-trained YOLOv11) | **GPU T4 x2** or **P100** | **ON** | 1. Click **+ Add Data** $\rightarrow$ attach `distill_datasetforme`<br>2. Click **Run All** |

---

## ⚙️ Quick Execution Instructions

1. **Upload**: In Kaggle, click **New Notebook** $\\rightarrow$ **File** $\\rightarrow$ **Import Notebook** $\\rightarrow$ select `.ipynb` file.
2. **Settings**: Set Accelerator to **GPU T4 x2** or **P100**, and set Internet to **ON**.
3. **Attach Data**: Click **+ Add Data** $\\rightarrow$ search `distill_datasetforme` (or your Crack500 dataset).
4. **Execute**: Click **Run All**. Training, validation, OOD testing, and JSON metric export run automatically.
