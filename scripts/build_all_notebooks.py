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
"""),

    make_cell("markdown", "## 📊 Evaluation & Head Freeze Analysis"),

    make_cell("code", """print("=== In-Domain DeepCrack Evaluation ===")
!python scripts/test_model.py --val --model runs/crack_distill_full_kd_box_instance_seg_yolo11n-seg/weights/best.pt --data data/datasets/deepcrack_yolo/dataset.yaml
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

# ==============================================================================
# NOTEBOOK 5: nb5_ablation_and_results.ipynb
# ==============================================================================
nb5_cells = [
    make_cell("markdown", """# 🔬 Notebook 5: Loss Component Ablation & Results Summary
This notebook runs full loss component ablations to isolate the contributions of:
- Mask KD loss ($L_{\text{mask\_kd}}$)
- Intermediate Feature Alignment MSE ($L_{\text{feature}}$)
- Uncertainty Boundary BCE ($L_{\text{boundary}}$)
- Progressive Segment Head Freezing (Stage 1 vs Full-Run Freeze)"""),

    make_cell("code", """!mkdir -p configs utils distillation scripts checkpoints data/datasets"""),

    make_cell("code", """!pip install -q ultralytics albumentations pycocotools thop pyyaml pandas"""),

    make_cell("code", """# Run component ablations
!python scripts/run_experiments.py --exp ablation_no_mask_kd --cfg configs/config.yaml
!python scripts/run_experiments.py --exp ablation_no_feature --cfg configs/config.yaml
!python scripts/run_experiments.py --exp ablation_no_boundary --cfg configs/config.yaml
"""),

    make_cell("markdown", "## 📈 Master Results Summary Table"),

    make_cell("code", """import pandas as pd
from pathlib import Path
from ultralytics import YOLO

ablation_runs = [
    ("Baseline (Fine-tune)", "runs/crack_distill_baseline_finetune_instance_seg_yolo11n-seg/weights/best.pt"),
    ("Full KD (Box)", "runs/crack_distill_full_kd_box_instance_seg_yolo11n-seg/weights/best.pt"),
    ("Full KD (Box+Centroid)", "runs/crack_distill_full_kd_centroid_instance_seg_yolo11n-seg/weights/best.pt"),
    ("Ablation: w/o Mask KD", "runs/crack_distill_ablation_no_mask_kd_instance_seg_yolo11n-seg/weights/best.pt"),
    ("Ablation: w/o Feature KD", "runs/crack_distill_ablation_no_feature_instance_seg_yolo11n-seg/weights/best.pt"),
    ("Ablation: w/o Boundary KD", "runs/crack_distill_ablation_no_boundary_instance_seg_yolo11n-seg/weights/best.pt"),
]

data_yaml = "data/datasets/crack500_yolo/dataset.yaml"
results = []
for name, ckpt in ablation_runs:
    if Path(ckpt).exists():
        m = YOLO(ckpt)
        res = m.val(data=data_yaml, split="val")
        results.append({
            "Configuration": name,
            "Cropped mAP50-seg": res.seg.map50,
            "Cropped mAP50-95-seg": res.seg.map,
            "Cropped mAP50-box": res.box.map50,
        })

df = pd.DataFrame(results)
print("\\n=== ABLATION STUDY RESULTS ===")
print(df.to_string(index=False))
""")
]

with open(out_dir / "nb5_ablation_and_results.ipynb", "w") as f:
    json.dump(make_nb(nb5_cells), f, indent=1, ensure_ascii=False)
print("Created nb5_ablation_and_results.ipynb")
