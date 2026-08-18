# OODimprovements/ — Kaggle Notebook Suite

Implements the mosaic-composite + native-scale-teacher plan from `../possibleOODimprovements.md`. Run in order; each notebook's Kaggle output is an input to the next.

| Notebook | What it does | Needs GPU? | Attach as input |
| :--- | :--- | :---: | :--- |
| `01_mine_mosaics_and_negatives.ipynb` | Stitches the existing `traincrop` grid into larger real composites (verified locally: 250/250 source photos, up to 1920x720) + a small pilot negative-tile set (verified: only 10 crops, from 5 stems that happen to overlap the val/test split by filename — **treat as diagnostic only**, not a real fix). Outputs `crack500_yolo_augmented/` (1896 original + 250 mosaic + 10 pilot-negative training images). | No (CPU) | `distill_datasetforme` |
| `02_generate_native_teacher_logits.ipynb` | Runs SAM2 on the mosaic composites to get teacher supervision beyond the 640x360 ceiling every prior run was capped at. | Yes | `distill_datasetforme` + notebook 01 output |
| `03_run_mosaic_native_kd.ipynb` | Trains the CWD (layers 12/15/18) + foreground-dilated mask-KL recipe (same losses as `final_notebooks/10`) on the augmented set, `imgsz=640`. Trains from the stock pretrained `yolo11n-seg.pt`, same as every other notebook in this project — no prior `best.pt` needed. | Yes | notebook 01 output + notebook 02 output |
| `04_eval_ood_and_tiled_inference.ipynb` | Direct copy of `final_notebooks/07` (Gaussian-weighted tiled inference). Attach notebook 03's checkpoint alongside the existing 7 for a fresh, non-hardcoded cross-checkpoint comparison. | Optional | notebook 03 output (+ prior checkpoints if comparing) |

## Do you need `best.pt` from a prior experiment?

**No.** Verified against `scripts/build_final_notebooks.py` / `distillation/kd_trainer.py`: every training notebook in this project (01–10, and `03_run_mosaic_native_kd.ipynb` here) starts from the stock Ultralytics-pretrained `yolo11n-seg.pt`, auto-downloaded — never from a previous crack-distill checkpoint. This keeps every run a clean, independently-comparable ablation. The **only** notebook that needs `best.pt` as input is the eval notebook (`04`), which needs it via Kaggle's "+ Add Data → Your Work → Notebook Output Files" to know which checkpoints to score.

## Regenerating these notebooks

```bash
python scripts/mine_negative_and_mosaic_tiles.py        # local sanity check (already run, see possibleOODimprovements.md)
python scripts/build_augmented_training_set.py           # local sanity check
python scripts/build_ood_improvement_notebooks.py         # rebuilds all 4 .ipynb files from the current source
```
