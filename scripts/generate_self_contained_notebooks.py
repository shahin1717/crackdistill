import json
from pathlib import Path

# Load source notebook cells from run_on_kaggle_final_rauf.ipynb
with open("run_on_kaggle_final_rauf.ipynb", "r") as f:
    source_nb = json.load(f)

source_cells = source_nb["cells"]

# Extract code file writefile cells by target filename
code_files = {}
for cell in source_cells:
    src = "".join(cell.get("source", []))
    if src.startswith("%%writefile"):
        first_line = src.split("\n")[0]
        filepath = first_line.replace("%%writefile", "").strip()
        code_files[filepath] = cell

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

# Read fresh content directly from physical files on disk
for filepath in list(code_files.keys()):
    if Path(filepath).exists():
        content = Path(filepath).read_text(encoding="utf-8")
        code_files[filepath] = make_cell("code", f"%%writefile {filepath}\n" + content)

print("Extracted self-contained source files:")
for path in code_files:
    print(f" - {path}")

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

# Common markdown & setup header
mkdir_cell = make_cell("code", "!mkdir -p configs utils distillation scripts checkpoints data/datasets data/teacher_logits_box data/teacher_logits_centroid runs")
env_cell = make_cell("code", "!pip install -q ultralytics albumentations pycocotools thop pyyaml pandas")
sam2_env_cell = make_cell("code", """# Install required packages & SAM 2
!pip install -q ultralytics albumentations pycocotools thop pyyaml pandas
!git clone https://github.com/facebookresearch/sam2.git sam2_repo || true
%cd sam2_repo
!pip install -e .
%cd ..
!rm -rf sam2
""")

linker_cell = source_cells[16] # Dataset linker from rauf notebook

# ==============================================================================
# NOTEBOOK 1: nb1_baseline_comparison.ipynb
# ==============================================================================
nb1_cells = [
    make_cell("markdown", """# 📊 Notebook 1: YOLOv8-seg vs YOLOv11-seg Baselines Comparison
This notebook is 100% self-contained for Kaggle execution. It establishes clean baseline fine-tuning results (without KD) comparing **YOLOv8n-seg** vs **YOLOv11n-seg** on **Crack500** and **DeepCrack** datasets separately.

### Experiments in this Notebook:
1. `v8_crack500_baseline`: YOLOv8n-seg trained on Crack500
2. `v11_crack500_baseline`: YOLOv11n-seg trained on Crack500
3. `v8_deepcrack_baseline`: YOLOv8n-seg trained on DeepCrack
4. `v11_deepcrack_baseline`: YOLOv11n-seg trained on DeepCrack"""),

    mkdir_cell,
    env_cell,
    code_files["scripts/convert_crack500.py"],
    code_files["scripts/convert_deepcrack.py"],
    linker_cell,
    make_cell("code", """# Convert raw datasets to YOLO format separately
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
from pathlib import Path
from ultralytics import YOLO

models = {
    "YOLOv8n-seg (Crack500)": ("runs/baselines/v8_crack500_baseline/weights/best.pt", "data/datasets/crack500_yolo/dataset.yaml"),
    "YOLOv11n-seg (Crack500)": ("runs/baselines/v11_crack500_baseline/weights/best.pt", "data/datasets/crack500_yolo/dataset.yaml"),
    "YOLOv8n-seg (DeepCrack)": ("runs/baselines/v8_deepcrack_baseline/weights/best.pt", "data/datasets/deepcrack_yolo/dataset.yaml"),
    "YOLOv11n-seg (DeepCrack)": ("runs/baselines/v11_deepcrack_baseline/weights/best.pt", "data/datasets/deepcrack_yolo/dataset.yaml"),
}

results = []
for name, (ckpt, data_yaml) in models.items():
    if Path(ckpt).exists() and Path(data_yaml).exists():
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
print("Saved self-contained nb1_baseline_comparison.ipynb")

# ==============================================================================
# NOTEBOOK 2: nb2_crack500_kd.ipynb
# ==============================================================================
nb2_cells = [
    make_cell("markdown", """# 🧪 Notebook 2: Full Knowledge Distillation on Crack500 Dataset
This notebook is 100% self-contained for Kaggle execution. It evaluates the SAM 2 to YOLO Knowledge Distillation framework trained **exclusively on Crack500**.

### Experiments in this Notebook:
1. `v8_c500_baseline`: YOLOv8n-seg baseline (no KD)
2. `v8_c500_full_kd_box`: YOLOv8n-seg with Full KD (Box prompts)
3. `v11_c500_baseline`: YOLOv11n-seg baseline (no KD)
4. `v11_c500_full_kd_box`: YOLOv11n-seg with Full KD (Box prompts)
5. `v11_c500_full_kd_centroid`: YOLOv11n-seg with Full KD (Box + Centroid prompts)"""),

    mkdir_cell,
    code_files["configs/config.yaml"],
    code_files["utils/config_loader.py"],
    code_files["distillation/kd_trainer.py"],
    code_files["distillation/trainer.py"],
    code_files["scripts/convert_crack500.py"],
    code_files["scripts/generate_teacher_logits.py"],
    code_files["scripts/test_model.py"],
    code_files["scripts/run_experiments.py"],
    sam2_env_cell,
    linker_cell,
    make_cell("code", """# Convert Crack500 to YOLO format & generate teacher logits
!python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo
!ln -sfn crack500_yolo data/datasets/combined_yolo

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
    make_cell("markdown", "## 📊 Evaluate: In-Domain (Crack500)"),
    make_cell("code", """print("=== In-Domain Crack500 Evaluation ===")
!python scripts/test_model.py --val --model runs/crack_distill_full_kd_centroid_instance_seg_yolo11n-seg/weights/best.pt --data data/datasets/crack500_yolo/dataset.yaml
""")
]

with open(out_dir / "nb2_crack500_kd.ipynb", "w") as f:
    json.dump(make_nb(nb2_cells), f, indent=1, ensure_ascii=False)
print("Saved self-contained nb2_crack500_kd.ipynb")

# ==============================================================================
# NOTEBOOK 3: nb3_deepcrack_kd.ipynb
# ==============================================================================
nb3_cells = [
    make_cell("markdown", """# 🧪 Notebook 3: Full Knowledge Distillation on DeepCrack Dataset
This notebook is 100% self-contained for Kaggle execution. It evaluates the SAM 2 to YOLO Knowledge Distillation framework trained **exclusively on DeepCrack**, featuring the **Segmentation Head Freeze Ablation**.

### Experiments in this Notebook:
1. `v8_dc_baseline`: YOLOv8n-seg baseline (no KD) on DeepCrack
2. `v8_dc_full_kd_box`: YOLOv8n-seg with Full KD (Box prompts) on DeepCrack
3. `v11_dc_baseline`: YOLOv11n-seg baseline (no KD) on DeepCrack
4. `v11_dc_full_kd_box`: YOLOv11n-seg with Full KD (Box prompts) on DeepCrack
5. `v11_dc_seghead_frozen_kd`: **NEW**: YOLOv11n-seg with Segment Head Frozen throughout full KD run"""),

    mkdir_cell,
    code_files["configs/config.yaml"],
    code_files["utils/config_loader.py"],
    code_files["distillation/kd_trainer.py"],
    code_files["distillation/trainer.py"],
    code_files["scripts/convert_deepcrack.py"],
    code_files["scripts/generate_teacher_logits.py"],
    code_files["scripts/test_model.py"],
    code_files["scripts/run_experiments.py"],
    sam2_env_cell,
    linker_cell,
    make_cell("code", """# Convert DeepCrack to YOLO format & generate teacher logits
!python scripts/convert_deepcrack.py --src data/datasets/deepcrack --dst data/datasets/deepcrack_yolo
!ln -sfn deepcrack_yolo data/datasets/combined_yolo

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
print("Saved self-contained nb3_deepcrack_kd.ipynb")

# ==============================================================================
# NOTEBOOK 4: nb4_cross_dataset_generalization.ipynb
# ==============================================================================
nb4_cells = [
    make_cell("markdown", """# 🌐 Notebook 4: Cross-Dataset Generalization Evaluation
This 100% self-contained evaluation notebook directly answers the CRITIC question: **"Is it better to train on Crack500 and test on DeepCrack? How do we solve cross-dataset overfitting?"**

It evaluates all baseline and KD models trained on **Crack500** directly on the **DeepCrack** test split (and vice versa) without fine-tuning on the target domain, quantifying domain transfer gap and KD generalization gains."""),

    mkdir_cell,
    env_cell,
    code_files["scripts/convert_crack500.py"],
    code_files["scripts/convert_deepcrack.py"],
    linker_cell,
    make_cell("code", """# Convert datasets to YOLO format for test evaluation
!python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo
!python scripts/convert_deepcrack.py --src data/datasets/deepcrack --dst data/datasets/deepcrack_yolo
"""),
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
    test_yaml = Path(item["test_data"])
    if ckpt_path.exists() and test_yaml.exists():
        model = YOLO(str(ckpt_path))
        metrics = model.val(data=str(test_yaml), split="test")
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
        print(f"Skipping {item['name']}: checkpoint ({ckpt_path}) or test yaml ({test_yaml}) not found.")

df = pd.DataFrame(results)
print("\\n" + "="*70)
print("🌐 CROSS-DATASET GENERALIZATION EVALUATION SUMMARY")
print("="*70)
print(df.to_string(index=False))
""")
]

with open(out_dir / "nb4_cross_dataset_generalization.ipynb", "w") as f:
    json.dump(make_nb(nb4_cells), f, indent=1, ensure_ascii=False)
print("Saved self-contained nb4_cross_dataset_generalization.ipynb")

# ==============================================================================
# NOTEBOOK 5: nb5_ablation_and_results.ipynb
# ==============================================================================
nb5_cells = [
    make_cell("markdown", """# 🔬 Notebook 5: Loss Component Ablation & Results Summary
This notebook is 100% self-contained for Kaggle execution. It runs full loss component ablations to isolate the contributions of:
- Mask KD loss ($L_{\\text{mask\\_kd}}$)
- Intermediate Feature Alignment MSE ($L_{\\text{feature}}$)
- Uncertainty Boundary BCE ($L_{\\text{boundary}}$)
- Progressive Segment Head Freezing (Stage 1 vs Full-Run Freeze)"""),

    mkdir_cell,
    code_files["configs/config.yaml"],
    code_files["utils/config_loader.py"],
    code_files["distillation/kd_trainer.py"],
    code_files["distillation/trainer.py"],
    code_files["scripts/convert_crack500.py"],
    code_files["scripts/generate_teacher_logits.py"],
    code_files["scripts/test_model.py"],
    code_files["scripts/run_experiments.py"],
    sam2_env_cell,
    linker_cell,
    make_cell("code", """# Convert Crack500 dataset & link combined_yolo
!python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo
!ln -sfn crack500_yolo data/datasets/combined_yolo

# Run component ablations
!python scripts/run_experiments.py --exp ablation_no_mask_kd --cfg configs/config.yaml
!python scripts/run_experiments.py --exp ablation_no_feature --cfg configs/config.yaml
!python scripts/run_experiments.py --exp ablation_no_boundary --cfg configs/config.yaml
"""),
    make_cell("markdown", "## 📈 Master Results Summary Table"),
        make_cell("code", """import glob
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

ablation_runs = [
    ("Baseline (Fine-tune)", ["baseline_finetune", "baseline", "nb2"], [
        "runs/segment/crack_distill/baseline_finetune/weights/best.pt",
        "runs/crack_distill_baseline_finetune_instance_seg_yolo11n-seg/weights/best.pt",
        "runs/crack_distill_baseline_finetune_instance_seg_yolov8n-seg/weights/best.pt",
    ]),
    ("Full KD (Box)", ["full_kd_box", "full_kd_prompts", "full_kd"], [
        "runs/segment/crack_distill/full_kd_box/weights/best.pt",
        "runs/crack_distill_full_kd_box_instance_seg_yolo11n-seg/weights/best.pt",
        "runs/crack_distill_full_kd_instance_seg_yolov8n-seg/weights/best.pt",
    ]),
    ("Full KD (Box+Centroid)", ["full_kd_centroid", "centroid"], [
        "runs/segment/crack_distill/full_kd_centroid/weights/best.pt",
        "runs/crack_distill_full_kd_centroid_instance_seg_yolo11n-seg/weights/best.pt",
        "runs/crack_distill_full_kd_centroid_instance_seg_yolov8n-seg/weights/best.pt",
    ]),
    ("Ablation: w/o Mask KD", ["ablation_no_mask_kd", "no_mask_kd", "nb5a"], [
        "runs/segment/crack_distill/ablation_no_mask_kd/weights/best.pt",
        "runs/crack_distill_ablation_no_mask_kd_instance_seg_yolo11n-seg/weights/best.pt",
        "runs/crack_distill_ablation_no_mask_kd_instance_seg_yolov8n-seg/weights/best.pt",
    ]),
    ("Ablation: w/o Feature KD", ["ablation_no_feature", "no_feature", "nb5b"], [
        "runs/segment/crack_distill/ablation_no_feature/weights/best.pt",
        "runs/crack_distill_ablation_no_feature_instance_seg_yolo11n-seg/weights/best.pt",
        "runs/crack_distill_ablation_no_feature_instance_seg_yolov8n-seg/weights/best.pt",
    ]),
    ("Ablation: w/o Boundary KD", ["ablation_no_boundary", "no_boundary", "nb5c"], [
        "runs/segment/crack_distill/ablation_no_boundary/weights/best.pt",
        "runs/crack_distill_ablation_no_boundary_instance_seg_yolo11n-seg/weights/best.pt",
        "runs/crack_distill_ablation_no_boundary_instance_seg_yolov8n-seg/weights/best.pt",
    ]),
]

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
                
    data["path"] = str(dataset_dir)
    
    runtime_yaml = Path(tempfile.gettempdir()) / "runtime_dataset.yaml"
    with open(runtime_yaml, "w") as f:
        yaml.dump(data, f)
        
    return str(runtime_yaml)

raw_data_yaml = find_dataset_yaml()
data_yaml = prepare_dataset_yaml(raw_data_yaml)
print(f"📁 Dataset YAML resolved & patched to: {data_yaml}")

results = []
for name, keywords, ckpt_candidates in ablation_runs:
    found_ckpt = find_checkpoint(name, keywords, ckpt_candidates)
    if found_ckpt:
        try:
            m = YOLO(found_ckpt)
            res = m.val(data=data_yaml, split="val")
            results.append({
                "Configuration": name,
                "Cropped mAP50-seg": getattr(res.seg, "map50", 0.0),
                "Cropped mAP50-95-seg": getattr(res.seg, "map", 0.0),
                "Cropped mAP50-box": getattr(res.box, "map50", 0.0),
                "Resolved Path": found_ckpt,
            })
        except Exception as e:
            print(f"⚠️ Error evaluating {name} ({found_ckpt}): {e}")

df = pd.DataFrame(results)
print("\\n=== ABLATION STUDY RESULTS ===")
if not df.empty:
    print(df.to_string(index=False))
else:
    print("No finished ablation checkpoints found.")
""")
]

with open(out_dir / "nb5_ablation_and_results.ipynb", "w") as f:
    json.dump(make_nb(nb5_cells), f, indent=1, ensure_ascii=False)
print("Saved self-contained nb5_ablation_and_results.ipynb")
