import json
from pathlib import Path

out_dir = Path("kaggle_notebooks")
out_dir.mkdir(exist_ok=True)

def make_cell(cell_type, source):
    if isinstance(source, str):
        lines = [line + "\n" for line in source.split("\n")]
        if lines and lines[-1] == "\n":
            lines.pop()
        source = lines
    return {
        "cell_type": cell_type,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

# ==============================================================================
# NOTEBOOK 1: nb1_baseline_comparison.ipynb
# ==============================================================================
nb1_cells = [
    make_cell("markdown", """# 📊 Notebook 1: YOLOv8-seg vs YOLOv11-seg Baselines Comparison
This notebook establishes clean baseline fine-tuning results (without KD) comparing **YOLOv8n-seg** vs **YOLOv11n-seg** on **Crack500** and **DeepCrack** datasets separately.

### Experiments in this Notebook:
1. `v8_crack500_baseline`: YOLOv8n-seg trained on Crack500
2. `v11_crack500_baseline`: YOLOv11n-seg trained on Crack500
3. `v8_deepcrack_baseline`: YOLOv8n-seg trained on DeepCrack
4. `v11_deepcrack_baseline`: YOLOv11n-seg trained on DeepCrack"""),

    make_cell("code", """!mkdir -p configs utils scripts checkpoints data/datasets runs"""),

    make_cell("code", """!pip install -q ultralytics albumentations pycocotools thop pyyaml"""),

    make_cell("code", """import os
import shutil
from pathlib import Path

input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists():
    input_dir = Path("/kaggle/input")

datasets_dir = Path("data/datasets")
datasets_dir.mkdir(parents=True, exist_ok=True)

# Link raw datasets
linked_crack = False
linked_deep = False
for root, dirs, files in os.walk(str(input_dir)):
    root_path = Path(root)
    if "traincrop" in dirs and not linked_crack:
        dest = datasets_dir / "crack500"
        if os.path.lexists(dest): os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(root_path, dest)
        print(f"Linked raw Crack500: {root_path} -> {dest}")
        linked_crack = True
    if "train_img" in dirs and not linked_deep:
        dest = datasets_dir / "deepcrack"
        if os.path.lexists(dest): os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(root_path, dest)
        print(f"Linked raw DeepCrack: {root_path} -> {dest}")
        linked_deep = True
"""),

    make_cell("code", """# Convert datasets to YOLO format separately (no combination)
!python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo
!python scripts/convert_deepcrack.py --src data/datasets/deepcrack --dst data/datasets/deepcrack_yolo
"""),

    make_cell("markdown", "## 🚀 Train Baselines (YOLOv8 vs YOLOv11 on Crack500 & DeepCrack)"),

    make_cell("code", """from ultralytics import YOLO

epochs = 150
imgsz = 512

print("=== 1. Training YOLOv8n-seg on Crack500 ===")
model_v8_c500 = YOLO("yolov8n-seg.pt")
model_v8_c500.train(
    data="data/datasets/crack500_yolo/dataset.yaml",
    epochs=epochs,
    imgsz=imgsz,
    project="runs/baselines",
    name="v8_crack500_baseline",
    seed=42
)

print("\\n=== 2. Training YOLOv11n-seg on Crack500 ===")
model_v11_c500 = YOLO("yolo11n-seg.pt")
model_v11_c500.train(
    data="data/datasets/crack500_yolo/dataset.yaml",
    epochs=epochs,
    imgsz=imgsz,
    project="runs/baselines",
    name="v11_crack500_baseline",
    seed=42
)

print("\\n=== 3. Training YOLOv8n-seg on DeepCrack ===")
model_v8_dc = YOLO("yolov8n-seg.pt")
model_v8_dc.train(
    data="data/datasets/deepcrack_yolo/dataset.yaml",
    epochs=epochs,
    imgsz=imgsz,
    project="runs/baselines",
    name="v8_deepcrack_baseline",
    seed=42
)

print("\\n=== 4. Training YOLOv11n-seg on DeepCrack ===")
model_v11_dc = YOLO("yolo11n-seg.pt")
model_v11_dc.train(
    data="data/datasets/deepcrack_yolo/dataset.yaml",
    epochs=epochs,
    imgsz=imgsz,
    project="runs/baselines",
    name="v11_deepcrack_baseline",
    seed=42
)
"""),

    make_cell("markdown", "## 📊 Comparative Evaluation Table"),

    make_cell("code", """import pandas as pd
from ultralytics import YOLO

models = {
    "YOLOv8n-seg (Crack500)": ("runs/baselines/v8_crack500_baseline/weights/best.pt", "data/datasets/crack500_yolo/dataset.yaml"),
    "YOLOv11n-seg (Crack500)": ("runs/baselines/v11_crack500_baseline/weights/best.pt", "data/datasets/crack500_yolo/dataset.yaml"),
    "YOLOv8n-seg (DeepCrack)": ("runs/baselines/v8_deepcrack_baseline/weights/best.pt", "data/datasets/deepcrack_yolo/dataset.yaml"),
    "YOLOv11n-seg (DeepCrack)": ("runs/baselines/v11_deepcrack_baseline/weights/best.pt", "data/datasets/deepcrack_yolo/dataset.yaml"),
}

results = []
for name, (ckpt, data_yaml) in models.items():
    if Path(ckpt).exists():
        m = YOLO(ckpt)
        metrics = m.val(data=data_yaml, split="val")
        results.append({
            "Model": name,
            "mAP50-box": metrics.box.map50,
            "mAP50-95-box": metrics.box.map,
            "mAP50-seg": metrics.seg.map50,
            "mAP50-95-seg": metrics.seg.map,
            "Params (M)": sum(p.numel() for p in m.model.parameters()) / 1e6
        })

df = pd.DataFrame(results)
print("\\n=== BASELINE COMPARISON SUMMARY ===")
print(df.to_string(index=False))
""")
]

with open(out_dir / "nb1_baseline_comparison.ipynb", "w") as f:
    json.dump(make_nb(nb1_cells), f, indent=1, ensure_ascii=False)
print("Created nb1_baseline_comparison.ipynb")

# ==============================================================================
# NOTEBOOK 2: nb2_crack500_kd.ipynb
# ==============================================================================
nb2_cells = [
    make_cell("markdown", """# 🧪 Notebook 2: Full Knowledge Distillation on Crack500 Dataset
This notebook evaluates the SAM 2 to YOLO Knowledge Distillation framework trained **exclusively on Crack500**.

### Experiments in this Notebook:
1. `v8_c500_baseline`: YOLOv8n-seg baseline (no KD)
2. `v8_c500_full_kd_box`: YOLOv8n-seg with Full KD (Box prompts)
3. `v11_c500_baseline`: YOLOv11n-seg baseline (no KD)
4. `v11_c500_full_kd_box`: YOLOv11n-seg with Full KD (Box prompts)
5. `v11_c500_full_kd_centroid`: YOLOv11n-seg with Full KD (Box + Centroid prompts)"""),

    make_cell("code", """!mkdir -p configs utils distillation scripts checkpoints data/datasets data/teacher_logits_box data/teacher_logits_centroid"""),

    make_cell("code", """# Install dependencies and SAM 2
!pip install -q ultralytics albumentations pycocotools thop pyyaml
!git clone https://github.com/facebookresearch/sam2.git sam2_repo || true
%cd sam2_repo
!pip install -e .
%cd ..
!rm -rf sam2
"""),

    make_cell("code", """import os
import shutil
from pathlib import Path

input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists():
    input_dir = Path("/kaggle/input")

datasets_dir = Path("data/datasets")
datasets_dir.mkdir(parents=True, exist_ok=True)
checkpoints_dir = Path("checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)

# Link teacher logits to /tmp for I/O speed
for folder in ["teacher_logits_box", "teacher_logits_centroid", "teacher_features"]:
    p_local = Path("data") / folder
    p_tmp = Path("/tmp") / folder
    if os.path.lexists(p_local):
        os.unlink(p_local) if os.path.islink(p_local) else shutil.rmtree(p_local)
    p_tmp.mkdir(parents=True, exist_ok=True)
    os.symlink(p_tmp, p_local)

# Link SAM 2 checkpoint
for root, dirs, files in os.walk(str(input_dir)):
    if "sam2_hiera_large.pt" in files:
        dest = checkpoints_dir / "sam2_hiera_large.pt"
        if os.path.lexists(dest): os.remove(dest)
        os.symlink(Path(root) / "sam2_hiera_large.pt", dest)
        break
if not (checkpoints_dir / "sam2_hiera_large.pt").exists():
    !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2_hiera_large.pt

# Link Crack500
for root, dirs, files in os.walk(str(input_dir)):
    if "traincrop" in dirs:
        dest = datasets_dir / "crack500"
        if os.path.lexists(dest): os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(Path(root), dest)
        break
"""),

    make_cell("code", """# Convert Crack500 to YOLO format & generate teacher logits
!python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo

print("=== Generating SAM 2 Box-Only Logits ===")
!python scripts/generate_teacher_logits.py --prompt-type box --logits-dir data/teacher_logits_box --dataset data/datasets/crack500_yolo

print("=== Generating SAM 2 Box+Centroid Logits ===")
!python scripts/generate_teacher_logits.py --prompt-type box_centroid --logits-dir data/teacher_logits_centroid --dataset data/datasets/crack500_yolo
"""),

    make_cell("markdown", "## 🏋️ Train Crack500 KD Models"),

    make_cell("code", """# Train comparative models on Crack500
!python scripts/run_experiments.py --exp baseline_finetune --cfg configs/config.yaml
!python scripts/run_experiments.py --exp full_kd_box --cfg configs/config.yaml
!python scripts/run_experiments.py --exp full_kd_centroid --cfg configs/config.yaml
"""),

    make_cell("markdown", "## 📊 Evaluate: In-Domain (Crack500) vs OOD Cross-Dataset (DeepCrack)"),

    make_cell("code", """print("=== In-Domain Crack500 Evaluation ===")
!python scripts/test_model.py --val --model runs/crack_distill_full_kd_centroid_instance_seg_yolo11n-seg/weights/best.pt --data data/datasets/crack500_yolo/dataset.yaml
""")
]

with open(out_dir / "nb2_crack500_kd.ipynb", "w") as f:
    json.dump(make_nb(nb2_cells), f, indent=1, ensure_ascii=False)
print("Created nb2_crack500_kd.ipynb")

# ==============================================================================
# NOTEBOOK 3: nb3_deepcrack_kd.ipynb
# ==============================================================================
nb3_cells = [
    make_cell("markdown", """# 🧪 Notebook 3: Full Knowledge Distillation on DeepCrack Dataset
This notebook evaluates the SAM 2 to YOLO Knowledge Distillation framework trained **exclusively on DeepCrack**, featuring the **Segmentation Head Freeze Ablation**.

### Experiments in this Notebook:
1. `v8_dc_baseline`: YOLOv8n-seg baseline (no KD) on DeepCrack
2. `v8_dc_full_kd_box`: YOLOv8n-seg with Full KD (Box prompts) on DeepCrack
3. `v11_dc_baseline`: YOLOv11n-seg baseline (no KD) on DeepCrack
4. `v11_dc_full_kd_box`: YOLOv11n-seg with Full KD (Box prompts) on DeepCrack
5. `v11_dc_seghead_frozen_kd`: **NEW**: YOLOv11n-seg with Segment Head Frozen throughout full KD run"""),

    make_cell("code", """!mkdir -p configs utils distillation scripts checkpoints data/datasets data/teacher_logits_box data/teacher_logits_centroid"""),

    make_cell("code", """# Install dependencies and SAM 2
!pip install -q ultralytics albumentations pycocotools thop pyyaml
!git clone https://github.com/facebookresearch/sam2.git sam2_repo || true
%cd sam2_repo
!pip install -e .
%cd ..
!rm -rf sam2
"""),

    make_cell("code", """import os
import shutil
from pathlib import Path

input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists():
    input_dir = Path("/kaggle/input")

datasets_dir = Path("data/datasets")
datasets_dir.mkdir(parents=True, exist_ok=True)
checkpoints_dir = Path("checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)

# Link teacher logits to /tmp
for folder in ["teacher_logits_box", "teacher_logits_centroid", "teacher_features"]:
    p_local = Path("data") / folder
    p_tmp = Path("/tmp") / folder
    if os.path.lexists(p_local):
        os.unlink(p_local) if os.path.islink(p_local) else shutil.rmtree(p_local)
    p_tmp.mkdir(parents=True, exist_ok=True)
    os.symlink(p_tmp, p_local)

# Link SAM 2 checkpoint
for root, dirs, files in os.walk(str(input_dir)):
    if "sam2_hiera_large.pt" in files:
        dest = checkpoints_dir / "sam2_hiera_large.pt"
        if os.path.lexists(dest): os.remove(dest)
        os.symlink(Path(root) / "sam2_hiera_large.pt", dest)
        break
if not (checkpoints_dir / "sam2_hiera_large.pt").exists():
    !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2_hiera_large.pt

# Link DeepCrack
for root, dirs, files in os.walk(str(input_dir)):
    if "train_img" in dirs:
        dest = datasets_dir / "deepcrack"
        if os.path.lexists(dest): os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(Path(root), dest)
        break
"""),

    make_cell("code", """# Convert DeepCrack to YOLO format & generate teacher logits
!python scripts/convert_deepcrack.py --src data/datasets/deepcrack --dst data/datasets/deepcrack_yolo

print("=== Generating SAM 2 Box-Only Logits for DeepCrack ===")
!python scripts/generate_teacher_logits.py --prompt-type box --logits-dir data/teacher_logits_box --dataset data/datasets/deepcrack_yolo
"""),

    make_cell("markdown", "## 🏋️ Train DeepCrack KD Models (Including SegHead Freeze Ablation)"),

    make_cell("code", """# Train comparative models on DeepCrack
!python scripts/run_experiments.py --exp baseline_finetune --cfg configs/config.yaml
!python scripts/run_experiments.py --exp full_kd_box --cfg configs/config.yaml
!python scripts/run_experiments.py --exp full_kd_seghead_frozen --cfg configs/config.yaml
"""),

    make_cell("markdown", "## 📊 Evaluation & Head Freeze Analysis"),

    make_cell("code", """print("=== In-Domain DeepCrack Comparative Evaluation ===")
models = {
    "Baseline (No KD)": "runs/crack_distill_baseline_finetune_instance_seg_yolo11n-seg/weights/best.pt",
    "Full KD (Box Prompts)": "runs/crack_distill_full_kd_box_instance_seg_yolo11n-seg/weights/best.pt",
    "Full KD (SegHead Frozen)": "runs/crack_distill_full_kd_seghead_frozen_instance_seg_yolo11n-seg/weights/best.pt",
}

for name, path in models.items():
    print(f"\\n--- Evaluating {name} ---")
    !python scripts/test_model.py --val --model {path} --data data/datasets/deepcrack_yolo/dataset.yaml
""")
]

with open(out_dir / "nb3_deepcrack_kd.ipynb", "w") as f:
    json.dump(make_nb(nb3_cells), f, indent=1, ensure_ascii=False)
print("Created nb3_deepcrack_kd.ipynb")

# ==============================================================================
# NOTEBOOK 4: nb4_cross_dataset_generalization.ipynb
# ==============================================================================
nb4_cells = [
    make_cell("markdown", """# 🌐 Notebook 4: Cross-Dataset Generalization Evaluation
This evaluation-only notebook directly answers the CRITIC question: **"Is it better to train on Crack500 and test on DeepCrack? How do we solve cross-dataset overfitting?"**

It evaluates all baseline and KD models trained on **Crack500** directly on the **DeepCrack** test split (and vice versa) without fine-tuning, quantifying domain transfer gap and KD generalization gains."""),

    make_cell("code", """!mkdir -p data/datasets scripts"""),

    make_cell("code", """!pip install -q ultralytics pandas pyyaml"""),

    make_cell("code", """import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# Define models to evaluate cross-dataset
eval_matrix = [
    # Train: Crack500 -> Test: DeepCrack
    {"name": "v8_crack500_baseline", "train": "Crack500", "test_data": "data/datasets/deepcrack_yolo/dataset.yaml", "ckpt": "runs/baselines/v8_crack500_baseline/weights/best.pt", "type": "Baseline"},
    {"name": "v11_crack500_baseline", "train": "Crack500", "test_data": "data/datasets/deepcrack_yolo/dataset.yaml", "ckpt": "runs/baselines/v11_crack500_baseline/weights/best.pt", "type": "Baseline"},
    {"name": "v11_crack500_full_kd", "train": "Crack500", "test_data": "data/datasets/deepcrack_yolo/dataset.yaml", "ckpt": "runs/crack_distill_full_kd_box_instance_seg_yolo11n-seg/weights/best.pt", "type": "Full KD"},

    # Train: DeepCrack -> Test: Crack500
    {"name": "v8_deepcrack_baseline", "train": "DeepCrack", "test_data": "data/datasets/crack500_yolo/dataset.yaml", "ckpt": "runs/baselines/v8_deepcrack_baseline/weights/best.pt", "type": "Baseline"},
    {"name": "v11_deepcrack_baseline", "train": "DeepCrack", "test_data": "data/datasets/crack500_yolo/dataset.yaml", "ckpt": "runs/baselines/v11_deepcrack_baseline/weights/best.pt", "type": "Baseline"},
    {"name": "v11_deepcrack_full_kd", "train": "DeepCrack", "test_data": "data/datasets/crack500_yolo/dataset.yaml", "ckpt": "runs/crack_distill_full_kd_box_instance_seg_deepcrack/weights/best.pt", "type": "Full KD"},
]

results = []
for item in eval_matrix:
    ckpt_path = Path(item["ckpt"])
    if ckpt_path.exists() and Path(item["test_data"]).exists():
        model = YOLO(str(ckpt_path))
        metrics = model.val(data=item["test_data"], split="test")
        results.append({
            "Experiment": item["name"],
            "Train Domain": item["train"],
            "Test Domain": "DeepCrack" if "deepcrack" in item["test_data"] else "Crack500",
            "Model Type": item["type"],
            "mAP50-seg": metrics.seg.map50,
            "mAP50-95-seg": metrics.seg.map,
            "mAP50-box": metrics.box.map50,
        })
    else:
        print(f"Skipping {item['name']}: checkpoint or data not found.")

df = pd.DataFrame(results)
print("\\n" + "="*70)
print("🌐 CROSS-DATASET GENERALIZATION EVALUATION SUMMARY")
print("="*70)
print(df.to_string(index=False))
""")
]

with open(out_dir / "nb4_cross_dataset_generalization.ipynb", "w") as f:
    json.dump(make_nb(nb4_cells), f, indent=1, ensure_ascii=False)
print("Created nb4_cross_dataset_generalization.ipynb")

# Read live code contents from workspace files for 100% self-contained %%writefile cells
config_yaml_code = Path("configs/config.yaml").read_text(encoding="utf-8")
config_loader_code = Path("utils/config_loader.py").read_text(encoding="utf-8")
kd_trainer_code = Path("distillation/kd_trainer.py").read_text(encoding="utf-8")
convert_crack500_code = Path("scripts/convert_crack500.py").read_text(encoding="utf-8")
convert_deepcrack_code = Path("scripts/convert_deepcrack.py").read_text(encoding="utf-8")
generate_teacher_logits_code = Path("scripts/generate_teacher_logits.py").read_text(encoding="utf-8")

def get_self_contained_writefile_cells():
    return [
        make_cell("code", "!mkdir -p configs utils distillation scripts checkpoints data/datasets data/teacher_logits_box data/teacher_logits_centroid runs"),
        make_cell("code", "!pip install -q ultralytics albumentations pycocotools thop pyyaml pandas"),
        make_cell("code", f"%%writefile configs/config.yaml\n{config_yaml_code}"),
        make_cell("code", f"%%writefile utils/__init__.py\n# utils package"),
        make_cell("code", f"%%writefile utils/config_loader.py\n{config_loader_code}"),
        make_cell("code", f"%%writefile distillation/__init__.py\n# distillation package"),
        make_cell("code", f"%%writefile distillation/kd_trainer.py\n{kd_trainer_code}"),
        make_cell("code", f"%%writefile scripts/convert_crack500.py\n{convert_crack500_code}"),
        make_cell("code", f"%%writefile scripts/convert_deepcrack.py\n{convert_deepcrack_code}"),
        make_cell("code", f"%%writefile scripts/generate_teacher_logits.py\n{generate_teacher_logits_code}"),
    ]

# Shared helper cell snippet for linking/generating logits for Crack500
crack500_dataset_setup_cell = make_cell("code", """import os, shutil
from pathlib import Path
input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists(): input_dir = Path("/kaggle/input")
datasets_dir = Path("data/datasets")
datasets_dir.mkdir(parents=True, exist_ok=True)
checkpoints_dir = Path("checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)

# 1. Link Crack500 dataset
for root, dirs, files in os.walk(str(input_dir)):
    root_path = Path(root)
    if "traincrop" in dirs:
        dest = datasets_dir / "crack500"
        if os.path.lexists(dest): os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(root_path, dest)
        print(f"Linked Crack500: {root_path} -> {dest}")
        break

# 2. Link precomputed teacher logits if available
found_logits = False
for root, dirs, files in os.walk(str(input_dir)):
    root_p = Path(root)
    if "teacher_logits_box" in dirs:
        src_f = root_p / "teacher_logits_box"
        dst_f = Path("data/teacher_logits_box")
        if os.path.lexists(dst_f): os.unlink(dst_f) if os.path.islink(dst_f) else shutil.rmtree(dst_f)
        os.symlink(src_f, dst_f)
        print(f"Linked precomputed teacher_logits_box: {src_f} -> {dst_f}")
        found_logits = True
        break
    if "teacher_features" in dirs:
        src_f = root_p / "teacher_features"
        dst_f = Path("data/teacher_features")
        if os.path.lexists(dst_f): os.unlink(dst_f) if os.path.islink(dst_f) else shutil.rmtree(dst_f)
        os.symlink(src_f, dst_f)
        print(f"Linked precomputed teacher_features: {src_f} -> {dst_f}")
""")

crack500_logits_generation_cell = make_cell("code", """# Ensure YOLO dataset & SAM 2 teacher logits exist
!python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo

from pathlib import Path
logits_dir = Path("data/teacher_logits_box")
logits_count = len(list(logits_dir.glob("*_logits.npy"))) if logits_dir.exists() else 0
print(f"Found {logits_count} teacher logit files in {logits_dir}")

if logits_count == 0:
    print("=== Generating SAM 2 Box-Only Logits ===")
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = checkpoints_dir / "sam2_hiera_large.pt"
    if not ckpt_file.exists():
        !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2_hiera_large.pt
    !pip install -q SAM-2 || pip install -q git+https://github.com/facebookresearch/segment-anything-2.git
    !python scripts/generate_teacher_logits.py --prompt-type box --logits-dir data/teacher_logits_box --dataset data/datasets/crack500_yolo
""")

# ==============================================================================
# NOTEBOOK 5a: nb5a_ablation_no_mask_kd.ipynb
# ==============================================================================
nb5a_cells = [
    make_cell("markdown", """# 🔬 Notebook 5a: Ablation 1 — Remove Soft Mask KL Loss (w/o $L_{\\text{KL}}$)
This notebook runs **Ablation 1**: Knowledge Distillation on Crack500 with **Mask KL Loss disabled** ($\alpha = 0$).
* **Goal**: Measure how much performance and OOD generalization drop when soft KL logit distillation is removed.
* **Input Dataset**: Crack500 + pre-computed SAM 2 teacher logits/features."""),
] + get_self_contained_writefile_cells() + [
    crack500_dataset_setup_cell,
    crack500_logits_generation_cell,
    make_cell("code", """# Run Ablation 1 (No Mask KL)
import sys
sys.path.insert(0, ".")
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
cfg = override_config(cfg, {
    "project.name": "crack_distill",
    "project.experiment": "ablation_no_mask_kd",
    "data.datasets": [{"name": "crack500", "path": "data/datasets/crack500_yolo", "format": "yolo"}],
    "distillation.enabled": True,
    "distillation.losses.mask_kd.enabled": False,
    "distillation.losses.feature.enabled": True,
    "distillation.losses.boundary.enabled": True,
    "teacher.logits_dir": "data/teacher_logits_box/"
})

trainer = KDSegmentationTrainer(cfg)
trainer.train()
print("✓ Ablation 1 (No Mask KL) completed!")
""")
]

with open(out_dir / "nb5a_ablation_no_mask_kd.ipynb", "w") as f:
    json.dump(make_nb(nb5a_cells), f, indent=1, ensure_ascii=False)
print("Created nb5a_ablation_no_mask_kd.ipynb")

# ==============================================================================
# NOTEBOOK 5b: nb5b_ablation_no_feature_mse.ipynb
# ==============================================================================
nb5b_cells = [
    make_cell("markdown", """# 🔬 Notebook 5b: Ablation 2 — Remove Feature Alignment MSE (w/o $L_{\\text{feature}}$)
This notebook runs **Ablation 2**: Knowledge Distillation on Crack500 with **Intermediate Feature Alignment MSE disabled** ($\beta = 0$).
* **Goal**: Measure how much representation transfer relies on intermediate backbone feature MSE.
* **Input Dataset**: Crack500 + pre-computed SAM 2 teacher logits."""),
] + get_self_contained_writefile_cells() + [
    crack500_dataset_setup_cell,
    crack500_logits_generation_cell,
    make_cell("code", """# Run Ablation 2 (No Feature MSE)
import sys
sys.path.insert(0, ".")
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
cfg = override_config(cfg, {
    "project.name": "crack_distill",
    "project.experiment": "ablation_no_feature",
    "data.datasets": [{"name": "crack500", "path": "data/datasets/crack500_yolo", "format": "yolo"}],
    "distillation.enabled": True,
    "distillation.losses.mask_kd.enabled": True,
    "distillation.losses.feature.enabled": False,
    "distillation.losses.boundary.enabled": True,
    "teacher.logits_dir": "data/teacher_logits_box/"
})

trainer = KDSegmentationTrainer(cfg)
trainer.train()
print("✓ Ablation 2 (No Feature MSE) completed!")
""")
]

with open(out_dir / "nb5b_ablation_no_feature_mse.ipynb", "w") as f:
    json.dump(make_nb(nb5b_cells), f, indent=1, ensure_ascii=False)
print("Created nb5b_ablation_no_feature_mse.ipynb")

# ==============================================================================
# NOTEBOOK 5c: nb5c_ablation_no_boundary_bce.ipynb
# ==============================================================================
nb5c_cells = [
    make_cell("markdown", """# 🔬 Notebook 5c: Ablation 3 — Remove Uncertainty Boundary BCE (w/o $L_{\\text{boundary}}$)
This notebook runs **Ablation 3**: Knowledge Distillation on Crack500 with **Boundary BCE Loss disabled** ($\gamma = 0$).
* **Goal**: Measure the impact of boundary-specific pixel BCE loss on thin crack edge precision.
* **Input Dataset**: Crack500 + pre-computed SAM 2 teacher logits."""),
] + get_self_contained_writefile_cells() + [
    crack500_dataset_setup_cell,
    crack500_logits_generation_cell,
    make_cell("code", """# Run Ablation 3 (No Boundary BCE)
import sys
sys.path.insert(0, ".")
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
cfg = override_config(cfg, {
    "project.name": "crack_distill",
    "project.experiment": "ablation_no_boundary",
    "data.datasets": [{"name": "crack500", "path": "data/datasets/crack500_yolo", "format": "yolo"}],
    "distillation.enabled": True,
    "distillation.losses.mask_kd.enabled": True,
    "distillation.losses.feature.enabled": True,
    "distillation.losses.boundary.enabled": False,
    "teacher.logits_dir": "data/teacher_logits_box/"
})

trainer = KDSegmentationTrainer(cfg)
trainer.train()
print("✓ Ablation 3 (No Boundary BCE) completed!")
""")
]

with open(out_dir / "nb5c_ablation_no_boundary_bce.ipynb", "w") as f:
    json.dump(make_nb(nb5c_cells), f, indent=1, ensure_ascii=False)
print("Created nb5c_ablation_no_boundary_bce.ipynb")

# ==============================================================================
# NOTEBOOK 5d: nb5d_ablation_seghead_frozen.ipynb
# ==============================================================================
deepcrack_dataset_setup_cell = make_cell("code", """import os, shutil
from pathlib import Path
input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists(): input_dir = Path("/kaggle/input")
datasets_dir = Path("data/datasets")
datasets_dir.mkdir(parents=True, exist_ok=True)
checkpoints_dir = Path("checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)

# 1. Link DeepCrack dataset
for root, dirs, files in os.walk(str(input_dir)):
    root_path = Path(root)
    if "train_img" in dirs:
        dest = datasets_dir / "deepcrack"
        if os.path.lexists(dest): os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(root_path, dest)
        print(f"Linked DeepCrack: {root_path} -> {dest}")
        break

# 2. Link precomputed teacher logits if available
found_logits = False
for root, dirs, files in os.walk(str(input_dir)):
    root_p = Path(root)
    if "teacher_logits_box" in dirs:
        src_f = root_p / "teacher_logits_box"
        dst_f = Path("data/teacher_logits_box")
        if os.path.lexists(dst_f): os.unlink(dst_f) if os.path.islink(dst_f) else shutil.rmtree(dst_f)
        os.symlink(src_f, dst_f)
        print(f"Linked precomputed teacher_logits_box: {src_f} -> {dst_f}")
        found_logits = True
        break
    if "teacher_features" in dirs:
        src_f = root_p / "teacher_features"
        dst_f = Path("data/teacher_features")
        if os.path.lexists(dst_f): os.unlink(dst_f) if os.path.islink(dst_f) else shutil.rmtree(dst_f)
        os.symlink(src_f, dst_f)
        print(f"Linked precomputed teacher_features: {src_f} -> {dst_f}")
""")

deepcrack_logits_generation_cell = make_cell("code", """# Ensure DeepCrack YOLO dataset & SAM 2 teacher logits exist
!python scripts/convert_deepcrack.py --src data/datasets/deepcrack --dst data/datasets/deepcrack_yolo

from pathlib import Path
logits_dir = Path("data/teacher_logits_box")
logits_count = len(list(logits_dir.glob("*_logits.npy"))) if logits_dir.exists() else 0
print(f"Found {logits_count} teacher logit files in {logits_dir}")

if logits_count == 0:
    print("=== Generating SAM 2 Box-Only Logits for DeepCrack ===")
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = checkpoints_dir / "sam2_hiera_large.pt"
    if not ckpt_file.exists():
        !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2_hiera_large.pt
    !pip install -q SAM-2 || pip install -q git+https://github.com/facebookresearch/segment-anything-2.git
    !python scripts/generate_teacher_logits.py --prompt-type box --logits-dir data/teacher_logits_box --dataset data/datasets/deepcrack_yolo
""")

nb5d_cells = [
    make_cell("markdown", """# 🔬 Notebook 5d: Ablation 4 — Full-Run Segmentation Head Freezing
This notebook runs **Ablation 4**: Knowledge Distillation on DeepCrack with **Segmentation Head Frozen throughout the ENTIRE training run** (not just Stage 1).
* **Goal**: Compare full-run head freezing against 2-stage progressive unfreezing.
* **Input Dataset**: DeepCrack + pre-computed SAM 2 teacher logits."""),
] + get_self_contained_writefile_cells() + [
    deepcrack_dataset_setup_cell,
    deepcrack_logits_generation_cell,
    make_cell("code", """# Run Ablation 4 (Full-Run SegHead Frozen)
import sys
sys.path.insert(0, ".")
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
cfg = override_config(cfg, {
    "project.name": "crack_distill",
    "project.experiment": "ablation_seghead_frozen",
    "data.datasets": [{"name": "deepcrack", "path": "data/datasets/deepcrack_yolo", "format": "yolo"}],
    "distillation.enabled": True,
    "distillation.progressive.enabled": True,
    "distillation.progressive.freeze_head": True,
    "distillation.progressive.unfreeze_epoch_ratio": 1.0,  # Never unfreeze during run
    "distillation.losses.mask_kd.enabled": True,
    "distillation.losses.feature.enabled": True,
    "distillation.losses.boundary.enabled": True,
    "teacher.logits_dir": "data/teacher_logits_box/"
})

trainer = KDSegmentationTrainer(cfg)
trainer.train()
print("✓ Ablation 4 (Full-Run SegHead Frozen) completed!")
""")
]

with open(out_dir / "nb5d_ablation_seghead_frozen.ipynb", "w") as f:
    json.dump(make_nb(nb5d_cells), f, indent=1, ensure_ascii=False)
print("Created nb5d_ablation_seghead_frozen.ipynb")

# ==============================================================================
# NOTEBOOK 5f: nb5f_ablation_mask_kd_only.ipynb
# ==============================================================================
nb5f_cells = [
    make_cell("markdown", """# 🔬 Notebook 5f: Ablation 5 — Mask KL Only (Task + Mask KL, No Feature MSE, No Boundary BCE)
This notebook runs **Ablation 5**: Knowledge Distillation on Crack500 with **Mask KL Loss ONLY** ($\alpha = 1.8658$, $\beta = 0$, $\gamma = 0$).
* **Goal**: Test the optimal 1-loss KD strategy identified from nb5b/nb5c findings.
* **Input Dataset**: Crack500 + pre-computed SAM 2 teacher logits."""),
] + get_self_contained_writefile_cells() + [
    crack500_dataset_setup_cell,
    crack500_logits_generation_cell,
    make_cell("code", """# Run Ablation 5 (Mask KL Only)
import sys
sys.path.insert(0, ".")
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
cfg = override_config(cfg, {
    "project.name": "crack_distill",
    "project.experiment": "ablation_mask_kd_only",
    "data.datasets": [{"name": "crack500", "path": "data/datasets/crack500_yolo", "format": "yolo"}],
    "distillation.enabled": True,
    "distillation.losses.mask_kd.enabled": True,
    "distillation.losses.feature.enabled": False,
    "distillation.losses.boundary.enabled": False,
    "teacher.logits_dir": "data/teacher_logits_box/"
})

trainer = KDSegmentationTrainer(cfg)
trainer.train()
print("✓ Ablation 5 (Mask KL Only) completed!")
""")
]

with open(out_dir / "nb5f_ablation_mask_kd_only.ipynb", "w") as f:
    json.dump(make_nb(nb5f_cells), f, indent=1, ensure_ascii=False)
print("Created nb5f_ablation_mask_kd_only.ipynb")

# ==============================================================================
# NOTEBOOK 5e: nb5e_ablation_results_summary.ipynb
# ==============================================================================
nb5e_cells = [
    make_cell("markdown", """# 📊 Notebook 5e: Master Ablation Study & Results Summary Aggregation
This notebook evaluates all trained model checkpoints from `nb2`, `nb3`, and `nb5a-f` to generate the master academic comparison table.
* **Input Checkpoints**: `best.pt` files from all ablation and production runs.
* **Outputs**: Master Markdown table with Mask mAP50, Mask mAP50-95, Box mAP50, and Precision/Recall."""),
] + get_self_contained_writefile_cells() + [
    make_cell("code", """import glob
from pathlib import Path
import tempfile
import yaml
import pandas as pd
from ultralytics import YOLO

ablation_runs_crack500 = [
    ("Baseline (No KD Fine-tune)", [
        "runs/crack500_baseline/weights/best.pt",
        "runs/segment/crack_distill/baseline_finetune/weights/best.pt",
        "runs/crack_distill_baseline_finetune_instance_seg_yolo11n-seg/weights/best.pt",
    ]),
    ("Full KD (Box Prompts)", [
        "runs/nb2_full_kd_box/weights/best.pt",
        "runs/segment/crack_distill/full_kd_box/weights/best.pt",
        "runs/crack_distill_full_kd_box_instance_seg_yolo11n-seg/weights/best.pt",
    ]),
    ("Full KD (Box + Centroid)", [
        "runs/nb2_full_kd_centroid/weights/best.pt",
        "runs/segment/crack_distill/full_kd_centroid/weights/best.pt",
        "runs/crack_distill_full_kd_centroid_instance_seg_yolo11n-seg/weights/best.pt",
    ]),
    ("Ablation 1: w/o Mask KL (nb5a)", [
        "runs/nb5a_ablation_no_mask_kd/weights/best.pt",
        "runs/segment/crack_distill/ablation_no_mask_kd/weights/best.pt",
        "runs/crack_distill_ablation_no_mask_kd_instance_seg_yolo11n-seg/weights/best.pt",
    ]),
    ("Ablation 2: w/o Feature MSE (nb5b)", [
        "runs/nb5b_ablation_no_feature/weights/best.pt",
        "runs/segment/crack_distill/ablation_no_feature/weights/best.pt",
        "runs/crack_distill_ablation_no_feature_instance_seg_yolo11n-seg/weights/best.pt",
    ]),
    ("Ablation 3: w/o Boundary BCE (nb5c)", [
        "runs/nb5c_ablation_no_boundary/weights/best.pt",
        "runs/segment/crack_distill/ablation_no_boundary/weights/best.pt",
        "runs/crack_distill_ablation_no_boundary_instance_seg_yolo11n-seg/weights/best.pt",
    ]),
    ("Ablation 5: Mask KL Only (nb5f)", [
        "runs/nb5f_ablation_mask_kd_only/weights/best.pt",
        "runs/segment/crack_distill/ablation_mask_kd_only/weights/best.pt",
        "runs/crack_distill_ablation_mask_kd_only_instance_seg_yolo11n-seg/weights/best.pt",
    ]),
]

deepcrack_ablation_runs = [
    ("Ablation 4: Full SegHead Freeze (nb5d)", [
        "runs/nb5d_ablation_seghead_frozen/weights/best.pt",
        "runs/segment/crack_distill/ablation_seghead_frozen/weights/best.pt",
        "runs/crack_distill_ablation_seghead_frozen_instance_seg_yolo11n-seg/weights/best.pt",
    ]),
]

def find_dataset_yaml(dset_name="crack500_yolo"):
    candidates = [
        f"data/datasets/{dset_name}/dataset.yaml",
        f"/kaggle/working/data/datasets/{dset_name}/dataset.yaml",
    ]
    for cand in candidates:
        cand_p = Path(cand)
        if cand_p.exists():
            parent = cand_p.parent
            if (parent / "images" / "val").exists() or (parent / "val").exists():
                return str(cand_p)

    search_dirs = [Path("data"), Path("/kaggle/input"), Path("/kaggle/working")]
    for d in search_dirs:
        if d.exists():
            matches = list(d.glob(f"**/{dset_name}/**/dataset.yaml")) + list(d.glob(f"**/{dset_name}/dataset.yaml")) + list(d.glob("**/dataset.yaml"))
            for m in matches:
                parent = m.parent
                if (parent / "images" / "val").exists() or (parent / "val").exists():
                    return str(m)

    for cand in candidates:
        if Path(cand).exists():
            return cand

    return f"data/datasets/{dset_name}/dataset.yaml"

def prepare_dataset_yaml(raw_yaml_path):
    yaml_path = Path(raw_yaml_path).resolve()
    dataset_dir = yaml_path.parent
    
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        
    val_rel = data.get("val", "images/val")
    if not (dataset_dir / val_rel).exists():
        matches = list(dataset_dir.glob("**/images/val")) or list(dataset_dir.glob("**/val"))
        if matches:
            real_val_path = matches[0]
            if real_val_path.name == "val" and real_val_path.parent.name == "images":
                dataset_dir = real_val_path.parent.parent
            else:
                dataset_dir = real_val_path.parent
                
    data["path"] = str(dataset_dir.resolve())
    
    runtime_yaml = Path(tempfile.gettempdir()) / f"runtime_{yaml_path.parent.name}_dataset.yaml"
    with open(runtime_yaml, "w") as f:
        yaml.dump(data, f)
        
    return str(runtime_yaml)

def find_checkpoint(name, candidates):
    # 1. Strict candidate checking — explicit exact path checks
    for cand in candidates:
        p = Path(cand)
        if p.exists():
            return str(p.resolve())
        abs_cand = Path("/kaggle/working") / cand
        if abs_cand.exists():
            return str(abs_cand.resolve())
            
    # 2. Scoped directory search matching exact experiment folder name + weight filename
    search_dirs = [Path("/kaggle/input"), Path("/kaggle/working/runs"), Path("runs"), Path("checkpoints")]
    for cand in candidates:
        cand_p = Path(cand)
        if len(cand_p.parts) >= 3 and cand_p.parts[-2] == "weights":
            exp_folder = cand_p.parts[-3]
        elif len(cand_p.parts) >= 2:
            exp_folder = cand_p.parent.name
        else:
            exp_folder = None
            
        filename = cand_p.name
        
        for d in search_dirs:
            if d.exists():
                if exp_folder:
                    matches = list(d.glob(f"**/{exp_folder}/**/{filename}"))
                else:
                    matches = list(d.glob(f"**/{filename}"))
                if matches:
                    return str(matches[0].resolve())

    print(f"⚠️ [STRICT RESOLVER] Checkpoint for '{name}' NOT found in explicit candidates or scoped search.")
    return None

crack500_yaml = prepare_dataset_yaml(find_dataset_yaml("crack500_yolo"))
deepcrack_yaml = prepare_dataset_yaml(find_dataset_yaml("deepcrack_yolo"))
print(f"📁 Crack500 Dataset YAML resolved to: {crack500_yaml}")
print(f"📁 DeepCrack Dataset YAML resolved to: {deepcrack_yaml}")

results_c500 = []
missing_c500 = []

for name, ckpt_candidates in ablation_runs_crack500:
    found_ckpt = find_checkpoint(name, ckpt_candidates)
    if found_ckpt:
        print(f"✅ Found Crack500 checkpoint for '{name}': {found_ckpt}")
        try:
            m = YOLO(found_ckpt)
            res = m.val(data=crack500_yaml, split="val")
            results_c500.append({
                "Ablation / Model Variant": name,
                "Evaluation Split": "Crack500 Val (348 imgs)",
                "Mask mAP50": getattr(res.seg, "map50", 0.0),
                "Mask mAP50-95": getattr(res.seg, "map", 0.0),
                "Box mAP50": getattr(res.box, "map50", 0.0),
                "Box mAP50-95": getattr(res.box, "map", 0.0),
                "Resolved Path": found_ckpt,
            })
        except Exception as e:
            print(f"⚠️ Error evaluating '{name}' ({found_ckpt}): {e}")
            missing_c500.append(name)
    else:
        missing_c500.append(name)

results_deep = []
for name, ckpt_candidates in deepcrack_ablation_runs:
    found_ckpt = find_checkpoint(name, ckpt_candidates)
    if found_ckpt:
        print(f"✅ Found DeepCrack checkpoint for '{name}': {found_ckpt}")
        try:
            m = YOLO(found_ckpt)
            res = m.val(data=deepcrack_yaml, split="val")
            results_deep.append({
                "Ablation / Model Variant": name,
                "Evaluation Split": "DeepCrack Val (60 imgs)",
                "Mask mAP50": getattr(res.seg, "map50", 0.0),
                "Mask mAP50-95": getattr(res.seg, "map", 0.0),
                "Box mAP50": getattr(res.box, "map50", 0.0),
                "Box mAP50-95": getattr(res.box, "map", 0.0),
                "Resolved Path": found_ckpt,
            })
        except Exception as e:
            print(f"⚠️ Error evaluating '{name}' ({found_ckpt}): {e}")

df_c500 = pd.DataFrame(results_c500)
df_deep = pd.DataFrame(results_deep)

print("\\n" + "="*70)
print("🔬 MASTER ABLATION STUDY SUMMARY (CRACK500 VAL SET)")
print("="*70)
if not df_c500.empty:
    print(df_c500.to_string(index=False))
else:
    print("No finished Crack500 ablation checkpoints found.")

if not df_deep.empty:
    print("\\n" + "="*70)
    print("🔬 DEEPCRACK SEGHEAD FREEZE ABLATION SUMMARY (DEEPCRACK VAL SET)")
    print("="*70)
    print(df_deep.to_string(index=False))

if missing_c500:
    print("\\n" + "-"*70)
    print("💡 KAGGLE / LOCAL CHECKPOINT ATTACHMENT GUIDE:")
    print("-"*70)
    print(f"Missing Crack500 variants ({len(missing_c500)}): {', '.join(missing_c500)}")
""")
]

with open(out_dir / "nb5e_ablation_results_summary.ipynb", "w") as f:
    json.dump(make_nb(nb5e_cells), f, indent=1, ensure_ascii=False)
print("Created nb5e_ablation_results_summary.ipynb")

# ==============================================================================
# NOTEBOOK 7: nb7_seed_reruns.ipynb
# ==============================================================================
nb7_cells = [
    make_cell("markdown", """# 🧪 Notebook 7: Multi-Seed Verification Study (Seeds 42, 123, 456)
This notebook executes **Multi-Seed Reruns** across the 5 core Crack500 ablation arms to obtain statistically robust `mean ± std` metrics for publication readiness.

### 📌 Ablation Arms & Seeds Included:
1. **Full KD** (`full_kd`): Seeds 123, 456 (plus baseline seed 42)
2. **w/o Mask KL** (`no_mask_kd`): Seeds 123, 456 (plus baseline seed 42)
3. **w/o Feature MSE** (`no_feature`): Seeds 123, 456 (plus baseline seed 42)
4. **w/o Boundary BCE** (`no_boundary`): Seeds 123, 456 (plus baseline seed 42)
5. **Mask KL Only** (`mask_kd_only`): Seeds 123, 456 (plus baseline seed 42)

* **Verification Gate**: Every run explicitly verifies `[KD] logit files: N > 0` before starting training."""),
] + get_self_contained_writefile_cells() + [
    crack500_dataset_setup_cell,
    crack500_logits_generation_cell,
]

# Generate training cells for 5 arms x 2 seeds (123, 456)
arms_config = [
    ("Full KD", "full_kd", True, True, True),
    ("w/o Mask KL", "no_mask_kd", False, True, True),
    ("w/o Feature MSE", "no_feature", True, False, True),
    ("w/o Boundary BCE", "no_boundary", True, True, False),
    ("Mask KL Only", "mask_kd_only", True, False, False),
]

for arm_label, arm_name, mask_on, feat_on, bound_on in arms_config:
    for seed in [123, 456]:
        train_code = f"""# ============================================================
# RUN: {arm_label} (Seed {seed})
# ============================================================
import sys
from pathlib import Path

logits_dir = Path("data/teacher_logits_box")
logits_count = len(list(logits_dir.glob("*.npy"))) if logits_dir.exists() else 0
assert logits_count > 0, f"[KD FATAL ERROR] Found {{logits_count}} logit files! Cannot run KD for {arm_name}_s{seed}."

sys.path.insert(0, ".")
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

print("🚀 Starting {arm_label} (Seed {seed})...")

cfg = load_config("configs/config.yaml")
cfg = override_config(cfg, {{
    "project.name": "crack_distill",
    "project.experiment": "ablation_{arm_name}_s{seed}",
    "project.seed": {seed},
    "data.datasets": [{{"name": "crack500", "path": "data/datasets/crack500_yolo", "format": "yolo"}}],
    "distillation.enabled": True,
    "distillation.losses.mask_kd.enabled": {mask_on},
    "distillation.losses.feature.enabled": {feat_on},
    "distillation.losses.boundary.enabled": {bound_on},
    "teacher.logits_dir": "data/teacher_logits_box/"
}})

trainer = KDSegmentationTrainer(cfg)
trainer.train()
print("✓ {arm_label} (Seed {seed}) completed!")
"""
        nb7_cells.append(make_cell("code", train_code))

# Multi-seed Aggregation Cell
aggregation_code = """# ============================================================
# MULTI-SEED STATISTICAL AGGREGATION SUMMARY (SEEDS 42, 123, 456)
# ============================================================
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

arms_eval = [
    ("Full KD", "full_kd", ["runs/nb2_full_kd_box/weights/best.pt", "runs/segment/crack_distill/full_kd_box/weights/best.pt"]),
    ("w/o Mask KL", "no_mask_kd", ["runs/nb5a_ablation_no_mask_kd/weights/best.pt", "runs/segment/crack_distill/ablation_no_mask_kd/weights/best.pt"]),
    ("w/o Feature MSE", "no_feature", ["runs/nb5b_ablation_no_feature/weights/best.pt", "runs/segment/crack_distill/ablation_no_feature/weights/best.pt"]),
    ("w/o Boundary BCE", "no_boundary", ["runs/nb5c_ablation_no_boundary/weights/best.pt", "runs/segment/crack_distill/ablation_no_boundary/weights/best.pt"]),
    ("Mask KL Only", "mask_kd_only", ["runs/nb5f_ablation_mask_kd_only/weights/best.pt", "runs/segment/crack_distill/ablation_mask_kd_only/weights/best.pt"]),
]

seeds = [42, 123, 456]
data_yaml = "data/datasets/crack500_yolo/dataset.yaml"

all_rows = []

for label, key, seed42_defaults in arms_eval:
    for seed in seeds:
        found_ckpt = None
        if seed == 42:
            candidates = seed42_defaults + [f"runs/crack_distill_ablation_{key}_s42_instance_seg_yolo11n-seg/weights/best.pt"]
        else:
            candidates = [
                f"runs/crack_distill_ablation_{key}_s{seed}_instance_seg_yolo11n-seg/weights/best.pt",
                f"runs/segment/crack_distill/ablation_{key}_s{seed}/weights/best.pt",
                f"runs/ablation_{key}_s{seed}/weights/best.pt",
            ]
        
        for cand in candidates:
            if Path(cand).exists():
                found_ckpt = cand
                break
            abs_c = Path("/kaggle/working") / cand
            if abs_c.exists():
                found_ckpt = str(abs_c)
                break
            # Search /kaggle/input for exact matches
            if Path("/kaggle/input").exists():
                cand_p = Path(cand)
                matches = list(Path("/kaggle/input").glob(f"**/{cand_p.parent.name}/{cand_p.name}"))
                if matches:
                    found_ckpt = str(matches[0])
                    break

        if found_ckpt:
            try:
                m = YOLO(found_ckpt)
                res = m.val(data=data_yaml, split="val")
                all_rows.append({
                    "Arm": label,
                    "Seed": seed,
                    "Mask mAP50": getattr(res.seg, "map50", 0.0),
                    "Mask mAP50-95": getattr(res.seg, "map", 0.0),
                    "Box mAP50": getattr(res.box, "map50", 0.0),
                    "Resolved Path": found_ckpt
                })
            except Exception as e:
                print(f"⚠️ Error evaluating {label} (Seed {seed}): {e}")
        else:
            print(f"Skipping {label} Seed {seed}: Checkpoint not found.")

df_raw = pd.DataFrame(all_rows)

print("\\n" + "="*75)
print("📊 INDIVIDUAL SEED EVALUATION RESULTS")
print("="*75)
if not df_raw.empty:
    print(df_raw.to_string(index=False))
    
    # Statistical Aggregation (mean ± std)
    agg_summary = []
    for arm_name, group in df_raw.groupby("Arm"):
        m_seg50 = group["Mask mAP50"].values
        m_seg95 = group["Mask mAP50-95"].values
        b_box50 = group["Box mAP50"].values
        
        agg_summary.append({
            "Ablation Arm": arm_name,
            "Seeds Evaluated": len(group),
            "Mask mAP50 (Mean ± Std)": f"{np.mean(m_seg50):.4f} ± {np.std(m_seg50):.4f}",
            "Mask mAP50-95 (Mean ± Std)": f"{np.mean(m_seg95):.4f} ± {np.std(m_seg95):.4f}",
            "Box mAP50 (Mean ± Std)": f"{np.mean(b_box50):.4f} ± {np.std(b_box50):.4f}",
        })
        
    df_agg = pd.DataFrame(agg_summary)
    print("\\n" + "="*75)
    print("📈 FINAL MULTI-SEED STATISTICAL SUMMARY (MEAN ± STD)")
    print("="*75)
    print(df_agg.to_string(index=False))
else:
    print("No multi-seed checkpoints found.")
"""

nb7_cells.append(make_cell("code", aggregation_code))

with open(out_dir / "nb7_seed_reruns.ipynb", "w") as f:
    json.dump(make_nb(nb7_cells), f, indent=1, ensure_ascii=False)
print("Created nb7_seed_reruns.ipynb")



