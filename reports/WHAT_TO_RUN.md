# 🚀 CrackDistill Execution Brief: What to Run & Required Inputs

> **Current State (as of September 10, 2026):** 
> - Baseline & Research Notebooks 01–10 (`final_notebooks/`): **All executed & verified.**
> - Mosaic Native Pipeline 01–04 (`OODimprovements/`): **Executed & verified** (All-time OOD champion: `0.1409` Mask mAP50, `0.1747` Box mAP50).
> - Two-Stage Fine-Tuning (`05_run_twostage`): **Executed & verified** (`0.1130` direct OOD mAP in 50 epochs).
> - Resolution-Preserving Mask-KL (`06_run_res_preserving`): **Executed & verified** (`0.2050` mAP50-95, `0.761/0.733` precision).
> - **Active Target:** Run **`08_eval_multiscale_tta_sahi.ipynb`** with checkpoints from **05 & 06** to benchmark megapixel TTA Dice, and train **`07_run_asymmetric_tversky_kd.ipynb`**.

---

## ⚡ 1. Top Priorities: What to Run Right Now

| Priority | Notebook | Purpose & Expected Gain | Hardware | Attached Kaggle Inputs | Runtime | Status |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: |
| **P0 (Immediate)** | **`OODimprovements/08_eval_multiscale_tta_sahi.ipynb`** | **Instant Megapixel Benchmarking**: Cross-evaluates checkpoints from **05** and **06** on raw uncropped photos via 2D Gaussian apodization & Multi-Scale TTA ($640+768$). Zero retraining. | GPU or CPU | 1. `distill_datasetforme`<br>2. Output of `05_twostage` (`best.pt`)<br>3. Output of `06_res_preserving` (`best.pt`) | **~8–10 mins** | 🟢 **RUN NOW** |
| **P1 (Next Training)** | **`OODimprovements/07_run_asymmetric_tversky_kd.ipynb`** | **Penalizes Broken Crack Branches**: Asymmetric Soft Tversky ($\beta=0.70$) + Neck CWD for continuous crack network topology. (150 epochs). | GPU T4 / P100 | 1. `distill_datasetforme`<br>2. Output of `01_mine_mosaics`<br>3. Output of `02_generate_teacher` | **~2.8 hrs** | 🟡 **NEXT RUN** |
| **P2 (Completed)** | **`OODimprovements/05_run_twostage_mosaic_native_tune.ipynb`** | **Two-Stage Transfer**: Rapid 50-epoch fine-tuning at $\text{lr}_0=0.001$. Achieved `0.1130` direct OOD Mask mAP50. | GPU T4 | Output in `output_runned/` | **~1.1 hrs** | ✅ **VERIFIED** |
| **P3 (Completed)** | **`OODimprovements/06_run_resolution_preserving_kd.ipynb`** | **Stops Thin-Crack Dilution**: Bilinearly upsamples student proto-mask before KL. Achieved `0.2050` mAP50-95 & `0.761` Box P. | GPU T4 | Output in `output_runned/` | **~5.2 hrs** | ✅ **VERIFIED** |

---

## 📋 2. Step-by-Step Kaggle Run Instructions

### Option A: Run `08_eval_multiscale_tta_sahi.ipynb` (Instant Evaluation)
1. Open Kaggle $\to$ **New Notebook** $\to$ Upload `OODimprovements/08_eval_multiscale_tta_sahi.ipynb`.
2. Click **+ Add Data**:
   - Attach dataset: `distill_datasetforme`.
   - Attach checkpoint: Click **Your Work** $\to$ **Notebook Output Files** $\to$ select the output of `03_run_mosaic_native_kd` (or `final_notebooks/06_layerkd`).
3. Settings:
   - Accelerator: **GPU T4 x2** or **P100** (or CPU).
   - Internet: **ON**.
4. Click **Run All**.
5. Output: `results/multiscale_tta_sahi_eval_summary.json` with Direct vs Tiled vs Multi-Scale TTA metrics.

---

### Option B: Run `05_run_twostage_mosaic_native_tune.ipynb` (50-Epoch Fine-Tuning)
1. Open Kaggle $\to$ **New Notebook** $\to$ Upload `OODimprovements/05_run_twostage_mosaic_native_tune.ipynb`.
2. Click **+ Add Data**:
   - Attach `distill_datasetforme`.
   - Attach Notebook 01 output: `01-mine-mosaics-and-negatives`.
   - Attach Notebook 02 output: `02-generate-native-teacher-logits`.
   - Attach Stage 1 checkpoint: Select output of `final_notebooks/04_run_pixel_affinity_kd` or `06_run_multiscale_layer_kd`.
3. Settings:
   - Accelerator: **GPU T4 x2** or **P100**.
   - Internet: **ON**.
4. Click **Run All**.
5. Output: `best.pt` in `runs/segment/exp_twostage_mosaic_native_tune_seed42_50ep/weights/`.

---

### Option C: Run `06_run_resolution_preserving_kd.ipynb` (Resolution-Preserving Mask-KL)
1. Open Kaggle $\to$ **New Notebook** $\to$ Upload `OODimprovements/06_run_resolution_preserving_kd.ipynb`.
2. Click **+ Add Data**:
   - Attach `distill_datasetforme`.
   - Attach Notebook 01 output (`crack500_yolo_augmented`).
   - Attach Notebook 02 output (`teacher_logits_box`).
3. Settings:
   - Accelerator: **GPU T4 x2** or **P100**.
   - Internet: **ON**.
4. Click **Run All** (150 epochs).

---

## 🚫 3. What Has Already Succeeded / Failed (Do Not Repeat)

| Experiment / Component | Outcome | Reason / Status |
| :--- | :---: | :--- |
| **`OODimprovements/01` & `02`** | ✅ **Succeeded** | 250 composites stitched, native logits generated. |
| **`OODimprovements/03` (Mosaic Native KD)** | ✅ **Succeeded** | SOTA `0.1409` Mask mAP50 (+66%), `0.1747` Box mAP50 (+106%). |
| **`final_notebooks/01–10`** | ✅ **Succeeded** | Baseline, Dilated, Affinity, Multiscale, LayerKD all verified. |
| **`09_run_combined_affinity_dilated`** | ❌ **Failed (mAP 0.0851)** | Dual mask losses collide. Do not stack Affinity with Dilated KL. |
| **Intermediate Feature MSE (Raw)** | ❌ **Failed (-2.14% mAP)** | 79× ViT-CNN capacity gap over-constrains backbone. |
| **Full SegHead Freeze (Crack500)** | ❌ **Failed** | Restricts multi-task head adaptation on large datasets. |
