# OODimprovements/ — Kaggle Notebook Suite

Implements the out-of-distribution (OOD) generalization roadmap from `../next_moves_forOOD.md` and `../possibleOODimprovements.md`. Run in sequence or select specific research arms; each notebook's Kaggle output is chained to subsequent stages via Kaggle "+ Add Data -> Your Work -> Notebook Output Files".

---

## 📂 The 8-Notebook Suite

| Notebook | What It Does | Hardware | Required Kaggle Inputs | Expected Output |
| :--- | :--- | :---: | :--- | :--- |
| **`01_mine_mosaics_and_negatives.ipynb`** | Stitches the 250 source photos of the `traincrop` grid into wide composites (up to $1920 \times 720$) and extracts pilot negative tiles. | **CPU** | `distill_datasetforme` | `crack500_yolo_augmented/` (2,156 images) |
| **`02_generate_native_teacher_logits.ipynb`** | Runs SAM 2 Large on mosaic composites to generate native-resolution teacher logits beyond the $640 \times 360$ single-crop ceiling. | **GPU T4 / P100** | `distill_datasetforme` + **Output of 01** | `teacher_logits_box/` (2,446 merged logits) |
| **`03_run_mosaic_native_kd.ipynb`** | Trains student from stock `yolo11n-seg.pt` on the augmented set at native `imgsz=640` with Neck CWD (12, 15, 18) + Foreground-Dilated Mask-KL. | **GPU T4 / P100** | **Output of 01** + **Output of 02** | `best.pt` (**All-time OOD Champion: 0.1409 mAP50**) |
| **`04_eval_ood_and_tiled_inference.ipynb`** | Evaluates trained checkpoints on raw $2000 \times 1500$ survey road photos using the 2D Gaussian apodization sliding-window engine. | **GPU / CPU** | `distill_datasetforme` + Checkpoint `best.pt` | Tiled Dice & Direct mAP JSON report |
| **`05_run_twostage_mosaic_native_tune.ipynb`** | **Two-Stage Transfer Fine-Tuning**: Loads a high-performing crop-trained checkpoint (e.g. `04_affinity` or `06_layerkd` `best.pt`) and fine-tunes for **50 epochs** on mosaics at $\text{lr}_0 = 0.001$. | **GPU T4 / P100** | `distill_datasetforme` + **Output of 01** + **Output of 02** + Prior `best.pt` | `best.pt` (Context-adapted student) |
| **`06_run_resolution_preserving_kd.ipynb`** | **Resolution-Preserving Mask-KL**: Bilinearly upsamples student prototype logits to teacher native dimensions ($640 \times 360$) before KL divergence, stopping thin $<2\text{px}$ crack dilution. | **GPU T4 / P100** | `distill_datasetforme` + **Output of 01** + **Output of 02** | `best.pt` (High-resolution mask student) |
| **`07_run_asymmetric_tversky_kd.ipynb`** | **Asymmetric Soft Tversky KD**: Weights false negatives heavily ($\beta=0.70, \alpha=0.30$) to penalize broken crack branches, combined with Neck CWD. | **GPU T4 / P100** | `distill_datasetforme` + **Output of 01** + **Output of 02** | `best.pt` (Continuous crack student) |
| **`08_eval_multiscale_tta_sahi.ipynb`** | **Multi-Scale Gaussian TTA Engine**: Evaluates uncropped photos across multi-scale patches ($640 \times 640$ + $768 \times 768$) with Gaussian apodization blending, benchmarking Direct vs Tiled vs TTA. | **GPU / CPU** | `distill_datasetforme` + Discovered `*.pt` checkpoints | `multiscale_tta_sahi_eval_summary.json` |

---

## 🎯 Which Notebook Should You Run?

1. **If you want the immediate SOTA test-time boost without training**:
   - Run **`08_eval_multiscale_tta_sahi.ipynb`** attaching the trained `03_mosaic_native` checkpoint.
2. **If you have a trained crop checkpoint (`04_affinity` or `06_layerkd`)**:
   - Run **`05_run_twostage_mosaic_native_tune.ipynb`** to adapt its detailed crack representations to wide mosaic contexts in just 50 epochs.
3. **If you want to eliminate thin-crack logit dilution**:
   - Run **`06_run_resolution_preserving_kd.ipynb`**.
4. **If thin cracks are disconnecting or fragmenting**:
   - Run **`07_run_asymmetric_tversky_kd.ipynb`**.

---

## 🔁 Regenerating These Notebooks

All 8 notebooks are dynamically compiled from active source files (`distillation/kd_trainer.py`, `configs/config.yaml`, `scripts/`):

```bash
cd ~/distill
python scripts/build_ood_improvement_notebooks.py
```
