import json
from pathlib import Path

# Read master codes
with open('configs/config.yaml') as f:
    config_yaml = f.read()

with open('utils/config_loader.py') as f:
    config_loader_code = f.read()

with open('distillation/kd_trainer.py') as f:
    kd_trainer_code = f.read()

with open('scripts/convert_crack500.py') as f:
    convert_crack500_code = f.read()

with open('scripts/generate_teacher_logits.py') as f:
    generate_teacher_logits_code = f.read()

def make_cell(cell_type, source):
    if isinstance(source, str):
        lines = [line + '\n' for line in source.split('\n')]
        if lines and lines[-1] == '\n':
            lines.pop()
        source = lines
    return {
        'cell_type': cell_type,
        'metadata': {},
        'outputs': [],
        'source': source
    }

def make_nb(cells):
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}
        },
        'nbformat': 4,
        'nbformat_minor': 2
    }

def generate_notebook(prompt_type: str, out_filename: str):
    is_centroid = (prompt_type == "box_centroid")
    prompt_name = "Box + Centroid Prompts" if is_centroid else "Box-Only Prompts"
    folder_name = "teacher_logits_centroid" if is_centroid else "teacher_logits_box"
    exp_name = f"prod_mask_kd_{'box_centroid' if is_centroid else 'box_only'}_T3.7769_W0.9612_150ep"
    
    step1_code = f"""# ── Step 1: Link or Prepare Crack500 Dataset and SAM 2 Logits ({folder_name}) ──
import os, shutil
from pathlib import Path

input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists():
    input_dir = Path("/kaggle/input")

datasets_dir = Path("data/datasets")
datasets_dir.mkdir(parents=True, exist_ok=True)
checkpoints_dir = Path("checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)

# 1. Link Crack500 raw images
for root, dirs, files in os.walk(str(input_dir)):
    root_path = Path(root)
    if "traincrop" in dirs:
        dest = datasets_dir / "crack500"
        if os.path.lexists(dest):
            os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(root_path, dest)
        print(f"[Dataset] Linked Crack500: {{root_path}} -> {{dest}}")
        break

# 2. Link precomputed teacher logits for {folder_name}
found_logits = False
for root, dirs, files in os.walk(str(input_dir)):
    root_p = Path(root)
    if "{folder_name}" in dirs:
        src_f = root_p / "{folder_name}"
        dst_f = Path("data/{folder_name}")
        if os.path.lexists(dst_f):
            os.unlink(dst_f) if os.path.islink(dst_f) else shutil.rmtree(dst_f)
        os.symlink(src_f, dst_f)
        print(f"[Logits] Linked precomputed {folder_name}: {{src_f}} -> {{dst_f}}")
        found_logits = True
        break
"""

    step2_code = f"""# ── Step 2: Convert Dataset & Verify Teacher Logits ({folder_name}) ──
!python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo

from pathlib import Path
logits_dir = Path("data/{folder_name}")
logits_count = len(list(logits_dir.glob("*_logits.npy"))) if logits_dir.exists() else 0
print(f"[Verification] Found {{logits_count}} teacher logit files in {{logits_dir}}")

if logits_count == 0:
    print("=== Pre-computed logits not found in input; Generating SAM 2 {prompt_name} on GPU ===")
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = checkpoints_dir / "sam2_hiera_large.pt"
    if not ckpt_file.exists():
        !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2_hiera_large.pt
    !pip install -q SAM-2 || pip install -q git+https://github.com/facebookresearch/segment-anything-2.git
    !python scripts/generate_teacher_logits.py --prompt-type {prompt_type} --logits-dir data/{folder_name} --dataset data/datasets/crack500_yolo

logits_count = len(list(logits_dir.glob("*_logits.npy")))
assert logits_count > 0, f"[FATAL] 0 logit files found in {{logits_dir}}. Fix input dataset attachment before training!"
print(f"[Verification Passed] Ready to train with {{logits_count}} real SAM 2 teacher logits from {{logits_dir}}!")
"""

    step3_code = f"""# ── Step 3: Run Production Mask KD Training ({prompt_name}) ──
import sys
sys.path.insert(0, ".")
from pathlib import Path
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")

EXPERIMENT_NAME = "{exp_name}"

cfg = override_config(cfg, {{
    "project.name": "crack_distill",
    "project.experiment": EXPERIMENT_NAME,
    "project.seed": 42,
    "data.datasets": [{{"name": "crack500", "path": "data/datasets/crack500_yolo", "format": "yolo"}}],
    "teacher.logits_dir": "data/{folder_name}/",
    "train.epochs": 150,
    "train.amp": False,
    "distillation.enabled": True,
    "distillation.temperature": 3.7769,
    "distillation.progressive.enabled": False,
    "distillation.losses.task.weight": 1.0,
    "distillation.losses.mask_kd.enabled": True,
    "distillation.losses.mask_kd.weight": 0.9612,
    "distillation.losses.feature.enabled": False,
    "distillation.losses.feature.weight": 0.0,
    "distillation.losses.boundary.enabled": False,
    "distillation.losses.boundary.weight": 0.0
}})

print(f"=== Starting Production Mask KD ({prompt_name}): {{EXPERIMENT_NAME}} ===")
print("Parameters: Temperature=3.7769, Mask_KD_Weight=0.9612, Prompt={prompt_type}, Epochs=150, AMP=False, Freezing=False")

trainer = KDSegmentationTrainer(cfg)
trainer.train()
print("✓ Training completed successfully!")
"""

    step4_code = f"""# ── Step 4: Validate Best Checkpoint & Export Results ──
import glob, json, os
from pathlib import Path
from ultralytics import YOLO

# Strict checkpoint lookup — NO silent fallback to random runs
best_pt = glob.glob(f"runs/**/{{EXPERIMENT_NAME}}*/weights/best.pt", recursive=True)
assert best_pt, f"No checkpoint found for {{EXPERIMENT_NAME}}! (not falling back to a random one)"

print(f"Evaluating best checkpoint: {{best_pt[0]}}")
model = YOLO(best_pt[0])

# 1. Validate on Crack500 In-Domain Val Set
val_metrics = model.val(data="data/datasets/crack500_yolo/dataset.yaml", split="val", verbose=True)

results = {{
    "experiment": EXPERIMENT_NAME,
    "checkpoint": best_pt[0],
    "config": {{
        "prompt_type": "{prompt_type}",
        "temperature": 3.7769,
        "mask_kd_weight": 0.9612,
        "epochs": 150,
        "freezing": False,
        "feature_mse": False,
        "boundary_bce": False
    }},
    "metrics": {{
        "box_precision": float(val_metrics.box.p[0]) if hasattr(val_metrics.box, 'p') and len(val_metrics.box.p) > 0 else float(val_metrics.box.mp),
        "box_recall": float(val_metrics.box.r[0]) if hasattr(val_metrics.box, 'r') and len(val_metrics.box.r) > 0 else float(val_metrics.box.mr),
        "box_mAP50": float(val_metrics.box.map50),
        "box_mAP50_95": float(val_metrics.box.map),
        "mask_precision": float(val_metrics.seg.p[0]) if hasattr(val_metrics.seg, 'p') and len(val_metrics.seg.p) > 0 else float(val_metrics.seg.mp),
        "mask_recall": float(val_metrics.seg.r[0]) if hasattr(val_metrics.seg, 'r') and len(val_metrics.seg.r) > 0 else float(val_metrics.seg.mr),
        "mask_mAP50": float(val_metrics.seg.map50),
        "mask_mAP50_95": float(val_metrics.seg.map)
    }}
}}

# 2. Validate on Crack500 Uncropped (OOD) Set if available
if os.path.exists("data/datasets/crack500_uncropped_yolo/dataset.yaml"):
    print("\\nEvaluating on Crack500 Uncropped (OOD) Set...")
    ood_metrics = model.val(data="data/datasets/crack500_uncropped_yolo/dataset.yaml", split="val", verbose=False)
    results["metrics"]["ood_mask_mAP50"] = float(ood_metrics.seg.map50)
    results["metrics"]["ood_mask_mAP50_95"] = float(ood_metrics.seg.map)
    results["metrics"]["ood_box_mAP50"] = float(ood_metrics.box.map50)
    results["metrics"]["ood_box_mAP50_95"] = float(ood_metrics.box.map)
else:
    print("No uncropped dataset.yaml found at data/datasets/crack500_uncropped_yolo/ — skipping OOD evaluation.")

print("\\n" + "="*50)
print(f"🎯 FINAL EVALUATION RESULTS ({prompt_name} - Crack500 Val):")
print(f"   Mask mAP50    : {{results['metrics']['mask_mAP50']:.4f}}")
print(f"   Mask mAP50-95 : {{results['metrics']['mask_mAP50_95']:.4f}}")
print(f"   Box mAP50     : {{results['metrics']['box_mAP50']:.4f}}")
print(f"   Box mAP50-95  : {{results['metrics']['box_mAP50_95']:.4f}}")
if "ood_mask_mAP50" in results["metrics"]:
    print(f"   OOD Mask mAP50    : {{results['metrics']['ood_mask_mAP50']:.4f}}")
    print(f"   OOD Mask mAP50-95 : {{results['metrics']['ood_mask_mAP50_95']:.4f}}")
print("="*50)

out_file = Path(f"/kaggle/working/results/{{EXPERIMENT_NAME}}.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved structured summary to {{out_file}}")
"""

    cells = [
        make_cell('markdown', f"""# 🚀 Production Mask-Only KD: SAM 2 → YOLOv11n-seg ({prompt_name})
This self-contained notebook trains YOLOv11n-seg using **Mask KD Only** (Task + Soft Mask KL divergence) with **{prompt_name}** on **Crack500** for **150 epochs**:
* **Architecture**: YOLOv11n-seg (2.84M params, 10.2 GFLOPs)
* **Teacher**: SAM 2 Large (`sam2_hiera_large.pt`, 224M params)
* **Prompt Type**: `{prompt_type}` (logits in `data/{folder_name}/`)
* **Hyperparameters**: Temperature $\\tau = 3.7769$, Mask KD Weight $\\alpha = 0.9612$
* **Feature MSE & Boundary BCE**: Disabled
* **Head Freezing**: Disabled (Full end-to-end training)
* **Precision**: FP32 (`amp: false`) for 100% loss stability"""),

        make_cell('code', f"!mkdir -p configs utils distillation scripts checkpoints data/datasets data/{folder_name} runs results"),
        make_cell('code', "!pip install -q ultralytics albumentations pycocotools thop pyyaml pandas"),

        make_cell('code', f"%%writefile configs/config.yaml\n{config_yaml}"),
        make_cell('code', "%%writefile utils/__init__.py\n# utils package"),
        make_cell('code', f"%%writefile utils/config_loader.py\n{config_loader_code}"),
        make_cell('code', "%%writefile distillation/__init__.py\n# distillation package"),
        make_cell('code', f"%%writefile distillation/kd_trainer.py\n{kd_trainer_code}"),
        make_cell('code', f"%%writefile scripts/convert_crack500.py\n{convert_crack500_code}"),
        make_cell('code', f"%%writefile scripts/generate_teacher_logits.py\n{generate_teacher_logits_code}"),

        make_cell('code', step1_code),
        make_cell('code', step2_code),
        make_cell('code', step3_code),
        make_cell('code', step4_code)
    ]

    out_path = Path(out_filename)
    with open(out_path, 'w') as f:
        json.dump(make_nb(cells), f, indent=1, ensure_ascii=False)
    print(f"Generated {out_path} ({prompt_type}) with {len(cells)} cells.")

# Generate both notebooks in root directory
generate_notebook("box", "run_mask_kd_box_only.ipynb")
generate_notebook("box_centroid", "run_mask_kd_box_centroid.ipynb")
