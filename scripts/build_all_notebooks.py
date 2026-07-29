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
    ]

# ==============================================================================
# NOTEBOOK 5a: nb5a_ablation_no_mask_kd.ipynb
# ==============================================================================
nb5a_cells = [
    make_cell("markdown", """# 🔬 Notebook 5a: Ablation 1 — Remove Soft Mask KL Loss (w/o $L_{\\text{KL}}$)
This notebook runs **Ablation 1**: Knowledge Distillation on Crack500 with **Mask KL Loss disabled** ($\alpha = 0$).
* **Goal**: Measure how much performance and OOD generalization drop when soft KL logit distillation is removed.
* **Input Dataset**: Crack500 + pre-computed SAM 2 teacher logits/features."""),
] + get_self_contained_writefile_cells() + [
    make_cell("code", """import os, shutil
from pathlib import Path
input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists(): input_dir = Path("/kaggle/input")
datasets_dir = Path("data/datasets")
datasets_dir.mkdir(parents=True, exist_ok=True)

for root, dirs, files in os.walk(str(input_dir)):
    root_path = Path(root)
    if "traincrop" in dirs:
        dest = datasets_dir / "crack500"
        if os.path.lexists(dest): os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(root_path, dest)
        print(f"Linked Crack500: {root_path} -> {dest}")
        break
"""),
    make_cell("code", """!python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo"""),
    make_cell("code", """# Run Ablation 1 (No Mask KL)
import sys
sys.path.insert(0, ".")
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
cfg = override_config(cfg, {
    "project.name": "crack_distill",
    "project.experiment": "ablation_no_mask_kd",
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
    make_cell("code", """import os, shutil
from pathlib import Path
input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists(): input_dir = Path("/kaggle/input")
datasets_dir = Path("data/datasets")
datasets_dir.mkdir(parents=True, exist_ok=True)

for root, dirs, files in os.walk(str(input_dir)):
    root_path = Path(root)
    if "traincrop" in dirs:
        dest = datasets_dir / "crack500"
        if os.path.lexists(dest): os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(root_path, dest)
        print(f"Linked Crack500: {root_path} -> {dest}")
        break
"""),
    make_cell("code", """!python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo"""),
    make_cell("code", """# Run Ablation 2 (No Feature MSE)
import sys
sys.path.insert(0, ".")
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
cfg = override_config(cfg, {
    "project.name": "crack_distill",
    "project.experiment": "ablation_no_feature",
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
    make_cell("code", """import os, shutil
from pathlib import Path
input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists(): input_dir = Path("/kaggle/input")
datasets_dir = Path("data/datasets")
datasets_dir.mkdir(parents=True, exist_ok=True)

for root, dirs, files in os.walk(str(input_dir)):
    root_path = Path(root)
    if "traincrop" in dirs:
        dest = datasets_dir / "crack500"
        if os.path.lexists(dest): os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(root_path, dest)
        print(f"Linked Crack500: {root_path} -> {dest}")
        break
"""),
    make_cell("code", """!python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo"""),
    make_cell("code", """# Run Ablation 3 (No Boundary BCE)
import sys
sys.path.insert(0, ".")
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
cfg = override_config(cfg, {
    "project.name": "crack_distill",
    "project.experiment": "ablation_no_boundary",
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
nb5d_cells = [
    make_cell("markdown", """# 🔬 Notebook 5d: Ablation 4 — Full-Run Segmentation Head Freezing
This notebook runs **Ablation 4**: Knowledge Distillation on DeepCrack with **Segmentation Head Frozen throughout the ENTIRE training run** (not just Stage 1).
* **Goal**: Compare full-run head freezing against 2-stage progressive unfreezing.
* **Input Dataset**: DeepCrack + pre-computed SAM 2 teacher logits."""),
] + get_self_contained_writefile_cells() + [
    make_cell("code", """import os, shutil
from pathlib import Path
input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists(): input_dir = Path("/kaggle/input")
datasets_dir = Path("data/datasets")
datasets_dir.mkdir(parents=True, exist_ok=True)

for root, dirs, files in os.walk(str(input_dir)):
    root_path = Path(root)
    if "train_img" in dirs:
        dest = datasets_dir / "deepcrack"
        if os.path.lexists(dest): os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(root_path, dest)
        print(f"Linked DeepCrack: {root_path} -> {dest}")
        break
"""),
    make_cell("code", """!python scripts/convert_deepcrack.py --src data/datasets/deepcrack --dst data/datasets/deepcrack_yolo"""),
    make_cell("code", """# Run Ablation 4 (Full-Run SegHead Frozen)
import sys
sys.path.insert(0, ".")
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
cfg = override_config(cfg, {
    "project.name": "crack_distill",
    "project.experiment": "ablation_seghead_frozen",
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
# NOTEBOOK 5e: nb5e_ablation_results_summary.ipynb
# ==============================================================================
nb5e_cells = [
    make_cell("markdown", """# 📊 Notebook 5e: Master Ablation Study & Results Summary Aggregation
This notebook evaluates all trained model checkpoints from `nb2`, `nb3`, and `nb5a-d` to generate the master academic comparison table.
* **Input Checkpoints**: `best.pt` files from all ablation and production runs.
* **Outputs**: Master Markdown table with Mask mAP50, Mask mAP50-95, Box mAP50, and Precision/Recall."""),
] + get_self_contained_writefile_cells() + [
    make_cell("code", """import pandas as pd
from pathlib import Path
from ultralytics import YOLO

ablation_runs = [
    ("Baseline (No KD Fine-tune)", "runs/crack_distill_baseline_finetune_instance_seg_yolo11n-seg/weights/best.pt"),
    ("Full KD (Box Prompts)", "runs/crack_distill_full_kd_box_instance_seg_yolo11n-seg/weights/best.pt"),
    ("Full KD (Box + Centroid)", "runs/crack_distill_full_kd_centroid_instance_seg_yolo11n-seg/weights/best.pt"),
    ("Ablation 1: w/o Mask KL (nb5a)", "runs/crack_distill_ablation_no_mask_kd_instance_seg_yolo11n-seg/weights/best.pt"),
    ("Ablation 2: w/o Feature MSE (nb5b)", "runs/crack_distill_ablation_no_feature_instance_seg_yolo11n-seg/weights/best.pt"),
    ("Ablation 3: w/o Boundary BCE (nb5c)", "runs/crack_distill_ablation_no_boundary_instance_seg_yolo11n-seg/weights/best.pt"),
    ("Ablation 4: Full SegHead Freeze (nb5d)", "runs/crack_distill_ablation_seghead_frozen_instance_seg_yolo11n-seg/weights/best.pt"),
]

data_yaml = "data/datasets/crack500_yolo/dataset.yaml"
results = []
for name, ckpt in ablation_runs:
    ckpt_path = Path(ckpt)
    if ckpt_path.exists():
        m = YOLO(str(ckpt_path))
        res = m.val(data=data_yaml, split="val")
        results.append({
            "Ablation / Model Variant": name,
            "Mask mAP50": res.seg.map50,
            "Mask mAP50-95": res.seg.map,
            "Box mAP50": res.box.map50,
            "Box mAP50-95": res.box.map,
        })
    else:
        print(f"Skipping {name}: checkpoint {ckpt} not found.")

df = pd.DataFrame(results)
print("\\n" + "="*70)
print("🔬 MASTER ABLATION STUDY RESULTS SUMMARY")
print("="*70)
print(df.to_string(index=False))
""")
]

with open(out_dir / "nb5e_ablation_results_summary.ipynb", "w") as f:
    json.dump(make_nb(nb5e_cells), f, indent=1, ensure_ascii=False)
print("Created nb5e_ablation_results_summary.ipynb")


