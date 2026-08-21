# 🚀 CrackDistill Execution Brief: What to Run & Required Inputs

---

## ⚡ 1. Immediate Next Step (Evaluate Mosaic Checkpoint)

| Notebook | Purpose | GPU? | Kaggle "+ Add Data" Inputs | Runtime |
|---|---|:---:|---|:---:|
| **`OODimprovements/04_eval_ood_and_tiled_inference.ipynb`** | Evaluates the trained Mosaic checkpoint on full-resolution ($2000 \times 1500$) imagery using the 2D Gaussian Tiled Engine. | GPU / CPU | 1. `distill_datasetforme`<br>2. Output of `03_run_mosaic_native_kd` (`best.pt`) | ~5–10 mins |

---

## 🔬 2. Main Production & Research Notebooks (`final_notebooks/`)

All training notebooks auto-download `yolo11n-seg.pt` — **no prior checkpoint needed**.

| Notebook | Recipe & Target Metric | Accelerator | Attached Inputs |
|---|---|:---:|---|
| **`01_run_mask_kd_production_seed42.ipynb`** | **Locked Production Baseline**: Uniform Mask-KL ($\tau=3.78, \alpha=0.96$). | GPU T4 / P100 | `distill_datasetforme` |
| **`03_run_foreground_dilated_kd.ipynb`** | **#1 Best OOD Generalization**: Foreground-Dilated KL (**0.1007 mAP50**, +18.7%). | GPU T4 / P100 | `distill_datasetforme` |
| **`04_run_pixel_affinity_kd.ipynb`** | **#1 Best In-Domain Segmentation**: Spatial Pixel Affinity (**0.5569 mAP50**, +3.1%). | GPU T4 / P100 | `distill_datasetforme` |
| **`05_run_multiscale_mask_kd.ipynb`** | **#1 Best Bounding Box Accuracy**: 512x512 Logit Matching (**0.6001 Box mAP50**). | GPU T4 / P100 | `distill_datasetforme` |
| **`06_run_multiscale_layer_kd.ipynb`** | **#1 Best Fine-Grained & Tiled Dice**: Neck CWD on Layers 12, 15, 18 (**0.2747 Tiled Dice**). | GPU T4 / P100 | `distill_datasetforme` |
| **`07_eval_ood_and_tiled_inference.ipynb`** | Master evaluation on uncropped imagery with 2D Gaussian sliding window. | GPU / CPU | `distill_datasetforme` + `best.pt` from any training run |
| **`08_benchmark_speed_and_profile.ipynb`** | Hardware latency & FPS verification (**107.8 FPS** on T4). | GPU T4 | *None* (runs synthetic profile) |

---

## 🧩 3. Full Mosaic & Scale Pipeline (`OODimprovements/`)

Chained pipeline that breaks the $640 \times 360$ crop ceiling using 250 stitched wide composites ($1920 \times 720$):

```
[01_mine_mosaics] ──> [02_generate_teacher] ──> [03_run_mosaic_kd] ──> [04_eval_ood]
```

| Step | Notebook | Inputs to Attach in Kaggle | Output Produced |
|:---:|---|---|---|
| **1** | `01_mine_mosaics_and_negatives.ipynb` *(CPU)* | `distill_datasetforme` | `crack500_yolo_augmented/` (2,156 images) |
| **2** | `02_generate_native_teacher_logits.ipynb` *(GPU)* | `distill_datasetforme` + **Output of 01** | `teacher_logits_box/` (2,446 logits) |
| **3** | `03_run_mosaic_native_kd.ipynb` *(GPU)* | **Output of 01** + **Output of 02** | `best.pt` (Epoch 150 student) |
| **4** | `04_eval_ood_and_tiled_inference.ipynb` *(GPU)* | `distill_datasetforme` + **Output of 03** | Final OOD & Tiled Dice JSON |

---

## 🚫 4. What FAILED & Should NOT Be Re-Run

1. **`09_run_combined_affinity_dilated_kd.ipynb`** ❌: Mask mAP dropped to 0.0851 (Dilated alone is 0.1007). Dual output losses conflict with each other.
2. **Raw Feature MSE / Boundary BCE** ❌: 79× ViT-vs-CNN capacity gap + 99% asphalt flooding degrades performance.

---

## 🔮 5. Next Planned Breakthroughs to Build

1. **Negative Tile Injection (P1)**: Add ~200 background-only asphalt photos (0 instances) to eliminate false positives on raw road scans.
2. **SAM 2 / SAM 3 Multi-Prompt Ensemble (P2)**: Fuse **Box Prompts (0.6)** + **Skeleton Multi-Point Prompts (0.3)** + **Iterative Memory (0.1)** into a superior offline teacher logit tensor.
3. **clDice Skeleton Loss (P3)**: Penalize broken topological crack connectivity directly.
