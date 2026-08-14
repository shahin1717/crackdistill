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

cells = [
    make_cell('markdown', """# 🚀 Production SAM 2 → YOLOv11n-seg: Mask KD Only (No Freezing)
This self-contained notebook runs the **optimal 1-loss Knowledge Distillation** recipe on **Crack500** for **150 epochs**:
* **Architecture**: YOLOv11n-seg (2.84M params, 10.2 GFLOPs)
* **Teacher**: SAM 2 Large (`sam2_hiera_large.pt`, 224M params)
* **Loss Configuration**: Task Loss + Soft Mask KL Divergence ($\\tau = 3.7769$, $\\alpha = 0.9612$)
* **Feature MSE**: Disabled ($\\beta = 0$, avoids over-constraining thin cracks)
* **Boundary BCE**: Disabled ($\\gamma = 0$)
* **Head Freezing**: Disabled (Full end-to-end training without freezing)
* **Precision**: FP32 (`amp: false`) for 100% loss numerical stability"""),

    make_cell('code', "!mkdir -p configs utils distillation scripts checkpoints data/datasets data/teacher_logits_box data/teacher_logits_centroid runs results"),
    make_cell('code', "!pip install -q ultralytics albumentations pycocotools thop pyyaml pandas"),

    make_cell('code', f"%%writefile configs/config.yaml\n{config_yaml}"),
    make_cell('code', "%%writefile utils/__init__.py\n# utils package"),
    make_cell('code', f"%%writefile utils/config_loader.py\n{config_loader_code}"),
    make_cell('code', "%%writefile distillation/__init__.py\n# distillation package"),
    make_cell('code', f"%%writefile distillation/kd_trainer.py\n{kd_trainer_code}"),
    make_cell('code', f"%%writefile scripts/convert_crack500.py\n{convert_crack500_code}"),
    make_cell('code', f"%%writefile scripts/generate_teacher_logits.py\n{generate_teacher_logits_code}"),

    make_cell('code', """# ── Step 1: Link or Prepare Crack500 Dataset and SAM 2 Logits ──
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
        print(f"[Dataset] Linked Crack500: {root_path} -> {dest}")
        break

# 2. Link precomputed teacher logits (box or box_centroid) if available in Kaggle input
found_logits = False
for root, dirs, files in os.walk(str(input_dir)):
    root_p = Path(root)
    for target_name in ["teacher_logits_box", "teacher_logits_centroid", "teacher_logits"]:
        if target_name in dirs:
            src_f = root_p / target_name
            dst_f = Path(f"data/{target_name}")
            if os.path.lexists(dst_f):
                os.unlink(dst_f) if os.path.islink(dst_f) else shutil.rmtree(dst_f)
            os.symlink(src_f, dst_f)
            if target_name != "teacher_logits_box":
                canon_dst = Path("data/teacher_logits_box")
                if os.path.lexists(canon_dst):
                    os.unlink(canon_dst) if os.path.islink(canon_dst) else shutil.rmtree(canon_dst)
                os.symlink(src_f, canon_dst)
            print(f"[Logits] Linked precomputed {target_name}: {src_f} -> data/{target_name}")
            found_logits = True
            break
    if found_logits:
        break
"""),

    make_cell('code', """# ── Step 2: Convert Dataset & Verify Teacher Logits ──
!python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo

from pathlib import Path
logits_dir = Path("data/teacher_logits_box")
if not logits_dir.exists() and Path("data/teacher_logits_centroid").exists():
    logits_dir = Path("data/teacher_logits_centroid")

logits_count = len(list(logits_dir.glob("*_logits.npy"))) if logits_dir.exists() else 0
print(f"[Verification] Found {logits_count} teacher logit files in {logits_dir}")

if logits_count == 0:
    print("=== Pre-computed logits not found in input; Generating SAM 2 Logits on GPU ===")
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = checkpoints_dir / "sam2_hiera_large.pt"
    if not ckpt_file.exists():
        !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2_hiera_large.pt
    !pip install -q SAM-2 || pip install -q git+https://github.com/facebookresearch/segment-anything-2.git
    !python scripts/generate_teacher_logits.py --prompt-type box --logits-dir data/teacher_logits_box --dataset data/datasets/crack500_yolo

logits_count = len(list(logits_dir.glob("*_logits.npy")))
assert logits_count > 0, f"[FATAL] 0 logit files found in {logits_dir}. Fix input dataset attachment before training!"
print(f"[Verification Passed] Ready to train with {logits_count} real SAM 2 teacher logits from {logits_dir}!")
"""),

    make_cell('code', """# ── Step 3: Run Full Production Mask-Only KD Training (150 Epochs, No Freezing) ──
import sys
sys.path.insert(0, ".")
from pathlib import Path
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")

effective_logits_dir = "data/teacher_logits_box/"
if len(list(Path("data/teacher_logits_box").glob("*_logits.npy"))) == 0:
    if Path("data/teacher_logits_centroid").exists() and len(list(Path("data/teacher_logits_centroid").glob("*_logits.npy"))) > 0:
        effective_logits_dir = "data/teacher_logits_centroid/"

EXPERIMENT_NAME = "prod_mask_kd_only_T3.7769_W0.9612_150ep"

cfg = override_config(cfg, {
    "project.name": "crack_distill",
    "project.experiment": EXPERIMENT_NAME,
    "project.seed": 42,
    "data.datasets": [{"name": "crack500", "path": "data/datasets/crack500_yolo", "format": "yolo"}],
    "teacher.logits_dir": effective_logits_dir,
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
})

print(f"=== Starting Production Mask KD Run: {EXPERIMENT_NAME} ===")
print(f"Parameters: Temperature=3.7769, Mask_KD_Weight=0.9612, Logits_Dir={effective_logits_dir}, Epochs=150, AMP=False, Freezing=False")

trainer = KDSegmentationTrainer(cfg)
trainer.train()
print("✓ Training completed successfully!")
"""),

    make_cell('code', """# ── Step 4: Validate Best Checkpoint & Export Results ──
import glob, json, os
from pathlib import Path
from ultralytics import YOLO

best_pt = glob.glob(f"runs/**/{EXPERIMENT_NAME}*/weights/best.pt", recursive=True)
assert best_pt, f"No checkpoint found for {EXPERIMENT_NAME}! (not falling back to a random one)"

print(f"Evaluating best checkpoint: {best_pt[0]}")
model = YOLO(best_pt[0])

# 1. Validate on Crack500 In-Domain Val Set
val_metrics = model.val(data="data/datasets/crack500_yolo/dataset.yaml", split="val", verbose=True)

results = {
    "experiment": EXPERIMENT_NAME,
    "checkpoint": best_pt[0],
    "config": {
        "temperature": 3.7769,
        "mask_kd_weight": 0.9612,
        "epochs": 150,
        "freezing": False,
        "feature_mse": False,
        "boundary_bce": False
    },
    "metrics": {
        "box_precision": float(val_metrics.box.p[0]) if hasattr(val_metrics.box, 'p') and len(val_metrics.box.p) > 0 else float(val_metrics.box.mp),
        "box_recall": float(val_metrics.box.r[0]) if hasattr(val_metrics.box, 'r') and len(val_metrics.box.r) > 0 else float(val_metrics.box.mr),
        "box_mAP50": float(val_metrics.box.map50),
        "box_mAP50_95": float(val_metrics.box.map),
        "mask_precision": float(val_metrics.seg.p[0]) if hasattr(val_metrics.seg, 'p') and len(val_metrics.seg.p) > 0 else float(val_metrics.seg.mp),
        "mask_recall": float(val_metrics.seg.r[0]) if hasattr(val_metrics.seg, 'r') and len(val_metrics.seg.r) > 0 else float(val_metrics.seg.mr),
        "mask_mAP50": float(val_metrics.seg.map50),
        "mask_mAP50_95": float(val_metrics.seg.map)
    }
}

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
print(f"🎯 FINAL EVALUATION RESULTS (Crack500 Val):")
print(f"   Mask mAP50    : {results['metrics']['mask_mAP50']:.4f}")
print(f"   Mask mAP50-95 : {results['metrics']['mask_mAP50_95']:.4f}")
print(f"   Box mAP50     : {results['metrics']['box_mAP50']:.4f}")
print(f"   Box mAP50-95  : {results['metrics']['box_mAP50_95']:.4f}")
if "ood_mask_mAP50" in results["metrics"]:
    print(f"   OOD Mask mAP50    : {results['metrics']['ood_mask_mAP50']:.4f}")
    print(f"   OOD Mask mAP50-95 : {results['metrics']['ood_mask_mAP50_95']:.4f}")
print("="*50)

out_file = Path(f"/kaggle/working/results/{EXPERIMENT_NAME}.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved structured summary to {out_file}")
""")
]

nb_path = Path('run_mask_kd_production.ipynb')
with open(nb_path, 'w') as f:
    json.dump(make_nb(cells), f, indent=1, ensure_ascii=False)

print(f"Successfully generated {nb_path} with {len(cells)} cells.")
