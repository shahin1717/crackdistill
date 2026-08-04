import json
from pathlib import Path

# Load template cells from nb5a
with open("kaggle_notebooks/nb5a_ablation_no_mask_kd.ipynb") as f:
    nb5a = json.load(f)

cell1 = nb5a["cells"][1]["source"]
cell2 = nb5a["cells"][2]["source"]
cell4 = nb5a["cells"][4]["source"]
cell5 = nb5a["cells"][5]["source"]
cell6 = nb5a["cells"][6]["source"]
cell7 = nb5a["cells"][7]["source"]
cell8 = nb5a["cells"][11]["source"] # dataset/logits searcher
cell9 = nb5a["cells"][12]["source"] # dataset conversion & assertion

cell0_md = """# 🧪 Notebook 6: Batch Size Sensitivity Study (Batch=16 vs Batch=32) — EXP-26
This notebook runs **EXP-26**: Knowledge Distillation sensitivity analysis comparing **Batch Size 16** vs. **Batch Size 32** on Crack500.

### 📌 Research Motivation & Theoretical Justification:
1. **KD Gradient Noise**: KD soft labels produce denser, higher-variance gradients than hard one-hot labels because every pixel receives a soft probability distribution from SAM 2.
2. **Variance Reduction**: Increasing batch size from 16 to 32 reduces gradient variance by $\\approx 41\\%$ ($O(1/\\sqrt{B})$), promoting smoother convergence.
3. **Linear LR Scaling Rule**: Following Goyal et al. (2017) (*"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"*), doubling batch size requires scaling learning rate linearly ($LR_{32} = LR_{16} \\times \\frac{32}{16} = 0.002$) with a 5-epoch warmup to prevent initial instability.
"""

cell3_cfg = """%%writefile configs/config.yaml
# ============================================================
# Crack-Distill — Master Config
# ============================================================
project:
  name: crack-distill
  seed: 42
  output_dir: runs/

task:
  type: instance_seg
  num_classes: 1
  class_names:
    - crack

data:
  root: data/datasets/
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  image_size: 512
  batch_size: 16
  num_workers: 4

  datasets:
    - name: crack500
      path: data/datasets/crack500_yolo
      format: yolo

teacher:
  model: sam2
  checkpoint: checkpoints/sam2_hiera_large.pt
  config: configs/sam2.1/sam2.1_hiera_l.yaml
  device: cuda
  prompt_type: box
  save_logits: true
  logits_dir: data/teacher_logits_box/
  batch_size: 4

student:
  backbone: yolo11n-seg
  pretrained: true
  device: "0"
  imgsz: 512

distillation:
  enabled: true
  temperature: 3.7769

  progressive:
    enabled: false

  losses:
    task:
      weight: 1.0
    mask_kd:
      enabled: true
      weight: 0.9612
    feature:
      enabled: true
      weight: 1.8658
      layers: [2, 5, 8]
    boundary:
      enabled: true
      weight: 0.8055

train:
  epochs: 150
  lr: 0.001
  lr_scheduler: cosine
  warmup_epochs: 3
  optimizer: AdamW
  weight_decay: 0.0005
"""

cell10_run16 = """# ============================================================
# RUN 1: Batch Size 16 (Control - EXP-08 Baseline)
# ============================================================
import sys
from pathlib import Path

logits_dir = Path("data/teacher_logits_box")
logits_count = len(list(logits_dir.glob("*.npy"))) if logits_dir.exists() else 0
assert logits_count > 0, f"FATAL ERROR: Found {logits_count} logit files! Cannot run KD."

sys.path.insert(0, ".")
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

print("🚀 Starting RUN 1: Batch Size = 16 (lr = 0.001, warmup = 3 epochs)...")

cfg16 = load_config("configs/config.yaml")
cfg16 = override_config(cfg16, {
    "project.name": "crack_distill",
    "project.experiment": "batch16_control",
    "data.datasets": [{"name": "crack500", "path": "data/datasets/crack500_yolo", "format": "yolo"}],
    "data.batch_size": 16,
    "train.lr": 0.001,
    "train.warmup_epochs": 3,
    "distillation.enabled": True,
    "teacher.logits_dir": "data/teacher_logits_box/"
})

trainer16 = KDSegmentationTrainer(cfg16)
trainer16.train()
print("✓ RUN 1 (Batch=16) completed!")
"""

cell11_run32 = """# ============================================================
# RUN 2: Batch Size 32 (Scaled LR - EXP-26 Test)
# ============================================================
import sys
from pathlib import Path

logits_dir = Path("data/teacher_logits_box")
logits_count = len(list(logits_dir.glob("*.npy"))) if logits_dir.exists() else 0
assert logits_count > 0, f"FATAL ERROR: Found {logits_count} logit files! Cannot run KD."

sys.path.insert(0, ".")
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

print("🚀 Starting RUN 2: Batch Size = 32 (lr = 0.002 [Linear Scaling Rule], warmup = 5 epochs)...")

cfg32 = load_config("configs/config.yaml")
cfg32 = override_config(cfg32, {
    "project.name": "crack_distill",
    "project.experiment": "batch32_scaled",
    "data.datasets": [{"name": "crack500", "path": "data/datasets/crack500_yolo", "format": "yolo"}],
    "data.batch_size": 32,
    "train.lr": 0.002,         # Goyal et al. Linear LR Scaling: 0.001 * (32 / 16)
    "train.warmup_epochs": 5,  # Extended warmup for stability at larger batch size
    "distillation.enabled": True,
    "teacher.logits_dir": "data/teacher_logits_box/"
})

trainer32 = KDSegmentationTrainer(cfg32)
trainer32.train()
print("✓ RUN 2 (Batch=32) completed!")
"""

cell12_summary = """# ============================================================
# EVALUATION & COMPARATIVE RESULTS SUMMARY
# ============================================================
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

runs = [
    ("Batch Size 16 (Control)", [
        "runs/segment/crack_distill/batch16_control/weights/best.pt",
        "runs/crack_distill_batch16_control_instance_seg_yolo11n-seg/weights/best.pt",
    ]),
    ("Batch Size 32 (EXP-26)", [
        "runs/segment/crack_distill/batch32_scaled/weights/best.pt",
        "runs/crack_distill_batch32_scaled_instance_seg_yolo11n-seg/weights/best.pt",
    ]),
]

data_yaml = "data/datasets/crack500_yolo/dataset.yaml"
results = []

for name, ckpt_candidates in runs:
    found_ckpt = None
    for cand in ckpt_candidates:
        if Path(cand).exists():
            found_ckpt = cand
            break

    if found_ckpt:
        m = YOLO(found_ckpt)
        res = m.val(data=data_yaml, split="val")
        results.append({
            "Experiment Variant": name,
            "Batch Size": 16 if "16" in name else 32,
            "Learning Rate": 0.001 if "16" in name else 0.002,
            "Mask mAP50": res.seg.map50,
            "Mask mAP50-95": res.seg.map,
            "Box mAP50": res.box.map50,
            "Box mAP50-95": res.box.map,
            "Mask Precision": res.seg.mp,
            "Mask Recall": res.seg.mr
        })
    else:
        print(f"Skipping {name}: Checkpoint not found in {ckpt_candidates}")

df = pd.DataFrame(results)
print("\\n" + "="*75)
print("📊 EXP-26 BATCH SIZE SENSITIVITY STUDY RESULTS SUMMARY")
print("="*75)
if not df.empty:
    print(df.to_string(index=False))
else:
    print("No finished batch size experiment checkpoints found.")
"""

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in cell0_md.split("\n")]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell1},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell2},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell3_cfg.split("\n")]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell4},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell5},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell6},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell7},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell8},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell9},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell10_run16.split("\n")]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell11_run32.split("\n")]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in cell12_summary.split("\n")]},
]

nb_out = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

target = "kaggle_notebooks/nb6_batch_size_sensitivity.ipynb"
with open(target, "w") as f:
    json.dump(nb_out, f, indent=1)

print(f"✓ Successfully generated {target} with {len(cells)} cells.")
