# 📓 Notebook Experiment Results (`nb_exp_results.md`)

This document aggregates and presents all empirical results obtained from executed Kaggle training notebooks across the **Crack-Distill** knowledge distillation pipeline.

---

## ⚙️ Hardware & Execution Environment

* **Execution Platform**: Kaggle Notebooks (Dual NVIDIA Tesla T4 GPUs, 16 GB VRAM each)
* **Local Workspace**: WSL2 Ubuntu 24.04 LTS / PyTorch 2.5.1 + CUDA 12.4
* **Teacher Model**: SAM 2 Large (`sam2_hiera_large.pt`, 224M parameters)
* **Student Models**:
  * **YOLOv11n-seg**: 2.84M parameters, 10.2 GFLOPs
  * **YOLOv8n-seg**: 3.26M parameters, 12.0 GFLOPs

---

## 📊 1. Baseline Comparison Runs (`nb_1_runned.ipynb`)

Four baseline models were trained for **150 epochs** (image size = 512, batch size = 16, SGD optimizer, standard mosaic augmentations) without Knowledge Distillation (No KD):

| Model Architecture | Target Dataset | Box Precision ($P$) | Box Recall ($R$) | Box mAP50 | Box mAP50-95 | Mask Precision ($P$) | Mask Recall ($R$) | Mask mAP50 (seg) | Mask mAP50-95 (seg) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n-seg** | Crack500 | 0.6930 | 0.5110 | 0.5730 | 0.3850 | **0.7320** | 0.4890 | 0.5320 | 0.2030 |
| **YOLOv11n-seg** | Crack500 | **0.7490** | 0.5260 | **0.5970** | 0.3960 | 0.7120 | 0.4950 | **0.5400** | **0.2070** |
| **YOLOv8n-seg** | DeepCrack | 0.6930 | 0.5090 | 0.5570 | **0.4090** | 0.6610 | 0.4850 | 0.4960 | 0.1890 |
| **YOLOv11n-seg** | DeepCrack | 0.6720 | **0.5680** | 0.5700 | 0.4070 | 0.6460 | **0.5400** | **0.5380** | 0.1970 |

### Baseline Takeaways:
* **YOLOv11n-seg Efficiency**: Outperforms YOLOv8n-seg while using **13% fewer parameters** and **15% fewer GFLOPs**.
* **DeepCrack Mask Recall**: YOLOv11n-seg shows significant improvements in mask recall on thin cracks (+5.5 points: `0.5400` vs `0.4850`).

---

## 🧪 2. Optuna Hyperparameter Study (`optuna_full_runned.ipynb`)

Automated hyperparameter tuning executed over **10–12 trials** (15 epochs per trial) on Crack500 using SAM 2 guidance:

* **Optimization Objective**: Composite score $\text{Score} = 0.4 \cdot \text{mAP50}_{\text{cropped}} + 0.6 \cdot \text{mAP50}_{\text{OOD}}$
* **Best Hyperparameters Discovered**:
  * Temperature ($\tau$): **`3.7769`**
  * Task-Aligned Weight ($\alpha$): **`0.9612`**
  * Per-Instance KL Weight ($\beta$): **`1.8658`**
  * Feature MSE Weight ($\gamma$): **`0.8055`**

---

## 🚀 3. Full Production KD Run (`run_on_kaggle_final_rauf.ipynb`)

Full 150-epoch knowledge distillation run using the optimal Optuna hyperparameter set with logit clamping `[-30.0, 30.0]` to guarantee fp16 numerical stability:

| Evaluation Split | Metric | Baseline (No KD) | Full KD (SAM 2) | Relative Change |
| :--- | :--- | :---: | :---: | :---: |
| **Cropped (In-Distribution)** | Mask mAP50 | **0.5249** | 0.5152 | $-1.85\%$ (Regularized) |
| **Uncropped (Out-of-Distribution)** | Box mAP50 | 0.1488 | **0.1574** | **$+5.8\%$** |
| **Uncropped (Out-of-Distribution)** | Mask mAP50 | 0.1242 | **0.1308** | **$+5.3\%$** |
| **Uncropped (Out-of-Distribution)** | Mask mAP50-95 | 0.0319 | **0.0352** | **$+10.3\%$** |

### Key KD Insights:
1. **OOD Generalization**: SAM 2 spatial priors act as a strong regularizer against cropping artifact memorization, driving a **$+10.3\%$ relative boost** in high-overlap mAP50-95 on uncropped real-world test images.
2. **Numerical Stability**: Logit clamping completely eliminated Automatic Mixed Precision (AMP) fp16 NaN overflow crashes encountered in earlier iterations (`distill (4)_runned.ipynb`).

---

## 📋 4. Complete Experiment Registry

| Experiment ID | Notebook File | Dataset | Model | Config / Loss Strategy | Epochs | Mask mAP50 (Cropped) | Mask mAP50 (Uncropped OOD) | Status |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `EXP-01` | `run_on_kaggle_final.ipynb` | Combined | YOLOv8n-seg | Baseline (No KD) | 100 | 0.5290 | 0.1180 | ✅ Completed |
| `EXP-02` | `run_on_kaggle_final.ipynb` | Combined | YOLOv8n-seg | First-Gen Boundary Loss | 100 | 0.5500 | 0.1210 | ✅ Completed |
| `EXP-03` | `optuna_full_runned.ipynb` | Crack500 | YOLOv11n-seg | Optuna Tuning (12 Trials) | 15 | 0.4951 | 0.1054 | ✅ Best weights found |
| `EXP-04` | `run_on_kaggle_final.ipynb` | Crack500 | YOLOv11n-seg | Baseline (No KD) | 150 | 0.5249 | 0.1064 | ✅ Completed |
| `EXP-05` | `run_on_kaggle_final.ipynb` | Crack500 | YOLOv11n-seg | Full KD (Unclamped) | 42 | — | — | ❌ Crashed (NaN fp16) |
| `EXP-06` | `run_on_kaggle_final.ipynb` | Crack500 | YOLOv11n-seg | Full KD (Box+Centroid) | 44 | — | — | ❌ Crashed (NaN fp16) |
| `EXP-07` | `run_on_kaggle_final_rauf.ipynb` | Crack500 | YOLOv11n-seg | Full KD (Clamped BCE) | 10 | 0.4810 | 0.0980 | ✅ Patch Verified |
| `EXP-08` | `run_on_kaggle_final_rauf.ipynb` | Crack500 | YOLOv11n-seg | Full KD (Clamped BCE) | 150 | 0.5152 | **0.1308** | ✅ **Production Run (+10.3% OOD)** |
| `EXP-09` | `nb_1_runned.ipynb` | Crack500 | YOLOv8n-seg | Baseline (No KD) | 150 | 0.5320 | — | ✅ Script 1 Complete |
| `EXP-10` | `nb_1_runned.ipynb` | Crack500 | YOLOv11n-seg | Baseline (No KD) | 150 | **0.5400** | — | ✅ Script 1 Complete |
| `EXP-11` | `nb_1_runned.ipynb` | DeepCrack | YOLOv8n-seg | Baseline (No KD) | 150 | 0.4960 | — | ✅ Script 1 Complete |
| `EXP-12` | `nb_1_runned.ipynb` | DeepCrack | YOLOv11n-seg | Baseline (No KD) | 150 | **0.5380** | — | ✅ Script 1 Complete |
