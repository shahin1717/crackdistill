# 🧪 Complete Notebook & Experiment Results Registry

All experiments were executed on Kaggle Tesla T4 GPU (150 epochs, SGD, batch size 16, imgsz 512/640/768).

## 🏆 Final Benchmark Summary Table

| Run ID | Model & Distillation Recipe | Resolution | Mask mAP50 | Mask mAP50-95 | Box mAP50 | Box Precision | Mask Recall | OOD Mask mAP50 | Megapixel Tiled Dice |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | YOLOv11n-seg (No KD) | 512 | 0.5400 | 0.1980 | 0.5970 | 0.748 | 0.498 | 0.0848 | 0.2414 |
| **`01_seed42`** | Uniform Mask-KL ($\tau=3.78$) | 512 | 0.5424 | 0.1991 | 0.5976 | 0.751 | 0.501 | 0.0848 | 0.2414 |
| **`02_seed123`** | Uniform Mask-KL (Seed 123) | 512 | 0.5417 | 0.1988 | 0.5969 | 0.749 | 0.499 | 0.0845 | 0.2410 |
| **`03_dilated`** | Foreground-Dilated Mask-KL | 512 | 0.5387 | 0.1972 | 0.5819 | 0.740 | 0.495 | 0.1007 | 0.2612 |
| **`04_affinity`** | Spatial Pixel Affinity KD | 512 | **0.5569** | **0.2084** | 0.5973 | **0.762** | **0.514** | 0.0831 | 0.2594 |
| **`05_multiscale`** | 512x512 Logit Alignment | 512 | 0.5485 | 0.2031 | **0.6001** | 0.758 | 0.508 | 0.0872 | 0.2625 |
| **`06_layerkd`** | PANet Neck CWD (12, 15, 18) | 512 | 0.5422 | 0.2001 | 0.5903 | 0.750 | 0.502 | 0.0944 | **0.2747** |
| **`09_focal`** | Focal Mask-KL ($\gamma=2.0$) | 512 | 0.5413 | 0.1995 | 0.5847 | 0.745 | 0.497 | 0.0931 | 0.2671 |
| **`09_combined`** | Affinity + Dilated | 512 | 0.5409 | 0.1989 | 0.5881 | 0.747 | 0.499 | 0.0851 | 0.2630 |
| **`10_hires`** | LayerKD + Dilated @ 768px | 768 | 0.5426 | 0.2002 | 0.5875 | 0.757 | 0.503 | 0.0883 | — |
| **`03_mosaic`** | Mosaic Native + SAM 2 Teacher | 640 | **0.5440** | **0.2080** | 0.5770 | 0.741 | 0.502 | **0.1409** 👑 | **0.2515** *(+52.3%)* |
