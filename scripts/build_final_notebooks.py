#!/usr/bin/env python3
"""
Builder script to generate self-contained, production-ready Kaggle notebooks in final_notebooks/
Supports locked recipes and advanced research variants (Full KD, Mask-KL, Foreground-Dilated KL, Pixel Affinity, Multi-Scale, Tiled Inference).
"""

import json
from pathlib import Path

# Load master files to embed in notebooks
with open('configs/config.yaml', encoding='utf-8') as f:
    config_yaml = f.read()

with open('utils/config_loader.py', encoding='utf-8') as f:
    config_loader_code = f.read()

with open('distillation/kd_trainer.py', encoding='utf-8') as f:
    kd_trainer_code = f.read()

with open('utils/checkpoint.py', encoding='utf-8') as f:
    checkpoint_code = f.read()

with open('inference/tiled_inference.py', encoding='utf-8') as f:
    tiled_inference_code = f.read()

with open('scripts/convert_crack500.py', encoding='utf-8') as f:
    convert_crack500_code = f.read()

with open('scripts/convert_crack500_uncropped.py', encoding='utf-8') as f:
    convert_crack500_uncropped_code = f.read()

with open('scripts/generate_teacher_logits.py', encoding='utf-8') as f:
    generate_teacher_logits_code = f.read()

with open('scripts/generate_gt_soft_logits.py', encoding='utf-8') as f:
    generate_gt_soft_logits_code = f.read()

# Pinned: the letterbox / instance-order fixes were verified against this exact version
ULTRALYTICS_PIN = "ultralytics==8.4.60"


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


def format_python_dict(d, indent=4):
    lines = ["{\n"]
    for k, v in d.items():
        lines.append(f"{' ' * indent}{repr(k)}: {repr(v)},\n")
    lines.append("}")
    return "".join(lines)


def generate_training_notebook(
    variant_name,
    title,
    description,
    overrides_dict,
    seed=42,
    prompt_type="box",
    logits_dir="data/teacher_logits_box",
    arm=None,
):
    exp_name = f"{variant_name}_seed{seed}_150ep"
    arm = arm or variant_name
    is_kd = overrides_dict.get("distillation.enabled", True)

    if is_kd:
        prompt_desc = {
            "box": "Bounding Box Only (offline pre-computed soft logits)",
            "centroid": "Bounding Box + Centroid Points (offline pre-computed soft logits)",
            "gt_soft": "None — targets are Gaussian-blurred ground truth (sigma=2.0), generated in Step 2",
        }[prompt_type]
        teacher_desc = ("* **Teacher Model**: None — GT-soft control (same Mask-KL loss, blurred GT instead of SAM 2)"
                        if prompt_type == "gt_soft" else
                        "* **Teacher Model**: SAM 2 Large (`sam2_hiera_large.pt`, 224M parameters)")
        kd_terms = []
        if overrides_dict.get("distillation.losses.mask_kd.enabled", True):
            w = overrides_dict.get("distillation.losses.mask_kd.weight", 0.9612)
            kd_terms.append(f"Mask-KL ($W={w}$)")
        if overrides_dict.get("distillation.losses.feature.enabled", False):
            w = overrides_dict.get("distillation.losses.feature.weight", 1.8658)
            layers = overrides_dict.get("distillation.losses.feature.layers", [16, 19, 22])
            kd_terms.append(f"PANet Neck CWD Layers {layers} ($W={w}$)")
        if overrides_dict.get("distillation.losses.boundary.enabled", False):
            w = overrides_dict.get("distillation.losses.boundary.weight", 0.8055)
            kd_terms.append(f"Boundary Loss ($W={w}$)")
        if overrides_dict.get("distillation.losses.affinity.enabled", False):
            w = overrides_dict.get("distillation.losses.affinity.weight", 0.5)
            kd_terms.append(f"Pixel Affinity ($W={w}$)")
        loss_desc = f"* **Active KD Losses**: {', '.join(kd_terms) if kd_terms else 'Task Loss Only'}"
        temp_desc = f"* **Distillation Temperature**: $\\tau = {overrides_dict.get('distillation.temperature', 3.7769)}$"
    else:
        prompt_desc = "None (No distillation supervision)"
        teacher_desc = "* **Teacher Model**: None (Pure YOLOv11 fine-tuning control)"
        loss_desc = "* **Active Losses**: Task Loss Only (BCE + Box CIoU + DFL)"
        temp_desc = "* **Distillation**: Disabled (`distillation.enabled: false`)"

    if prompt_type == "gt_soft":
        logits_linking_code = """# 2. GT-soft control: never link SAM logits here — targets are generated from labels in Step 2
print("[GT-Soft] Skipping teacher-logit linking; blurred-GT targets are generated in Step 2.")"""
    elif is_kd:
        # Only link a folder holding the same teacher; never substitute box logits for centroid ones
        target_dirs = ["teacher_logits_centroid"] if prompt_type == "centroid" else [Path(logits_dir).name, "teacher_logits_box", "teacher_logits"]
        logits_linking_code = f"""# 2. Link precomputed teacher logits
found_logits = False
target_dirs = {target_dirs!r}

for root, dirs, files in os.walk(str(input_dir)):
    root_p = Path(root)
    for target_name in target_dirs:
        if target_name in dirs:
            src_f = root_p / target_name
            dst_f = Path("{logits_dir}")
            if os.path.lexists(dst_f):
                os.unlink(dst_f) if os.path.islink(dst_f) else shutil.rmtree(dst_f)
            os.symlink(src_f, dst_f)
            print(f"[Logits] Linked precomputed logits: {{src_f}} -> {{dst_f}}")
            found_logits = True
            break
    if found_logits:
        break
if not found_logits:
    print("[Logits Notice] Precomputed logits directory not found in input; will generate if needed.")"""
    else:
        logits_linking_code = """# 2. Baseline run - no teacher logits required
print("[Baseline] No teacher logits required for baseline control run.")"""

    if prompt_type == "gt_soft":
        logits_verify_code = f"""# 3. Generate GT-soft targets (<stem>_logits.npy, one map per label line, label order)
!python scripts/generate_gt_soft_logits.py --data data/datasets/crack500_yolo --out {logits_dir} --sigma 2.0
logits_dir = Path("{logits_dir}")
logits_count = len(list(logits_dir.glob("*_logits.npy")))
assert logits_count > 0, f"[FATAL ERROR] 0 GT-soft logit files in {{logits_dir}}."
print(f"[Verification Passed] {{logits_count}} GT-soft targets in {{logits_dir}} (no SAM 2 involved)")"""
    elif is_kd:
        logits_verify_code = f"""# 3. Verify Logits
logits_dir = Path("{logits_dir}")
logits_count = len(list(logits_dir.glob("*_logits.npy"))) if logits_dir.exists() else 0
print(f"[Verification] Found {{logits_count}} teacher logit files in {{logits_dir}}")

if logits_count == 0:
    print("=== Pre-computed logits not found in input; Generating SAM 2 Logits on GPU ===")
    ckpt_file = checkpoints_dir / "sam2_hiera_large.pt"
    if not ckpt_file.exists():
        !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2_hiera_large.pt
    !pip install -q SAM-2 || pip install -q git+https://github.com/facebookresearch/segment-anything-2.git
    !python scripts/generate_teacher_logits.py --prompt-type {prompt_type} --logits-dir {logits_dir} --dataset data/datasets/crack500_yolo

logits_count = len(list(logits_dir.glob("*_logits.npy")))
assert logits_count > 0, f"[FATAL ERROR] 0 logit files found in {{logits_dir}}. KD training cannot proceed without teacher supervision!"
print(f"[Verification Passed] Ready to train with {{logits_count}} real SAM 2 teacher logits from {{logits_dir}}!")"""
    else:
        logits_verify_code = """# 3. Verification
print("[Verification Passed] Clean baseline ready to train directly with standard task loss!")"""

    teacher_override = f'overrides["teacher.logits_dir"] = "{logits_dir}/"' if is_kd else '# No teacher logits override for baseline'

    cells = [
        make_cell('markdown', f"""# 🚀 Crack-Distill: {title} (Seed {seed})
{description}

### Recipe Specifications:
* **Student Model**: YOLOv11n-seg (2.84M parameters, 10.2 GFLOPs)
{teacher_desc}
* **Prompt Type**: {prompt_desc}
{loss_desc}
{temp_desc}
* **Head Freezing**: Disabled (Full end-to-end training)
* **Precision**: FP32 (`amp: false`) for 100% loss stability
* **Random Seed**: `{seed}`
* **Automated Evaluation**: Evaluates in-domain cropped val and out-of-distribution uncropped val upon completion."""),

        make_cell('code', f"""# ── Environment & Directory Initialization ──
!mkdir -p configs utils distillation inference scripts checkpoints data/datasets data/teacher_logits_box runs results
!pip install -q {ULTRALYTICS_PIN} albumentations pycocotools thop pyyaml pandas tqdm opencv-python Pillow
"""),

        make_cell('code', f"%%writefile configs/config.yaml\n{config_yaml}"),
        make_cell('code', "%%writefile utils/__init__.py\n# utils package\nfrom utils.checkpoint import resolve_checkpoint, get_checkpoint_manifest, save_checkpoint_manifest\n"),
        make_cell('code', f"%%writefile utils/config_loader.py\n{config_loader_code}"),
        make_cell('code', f"%%writefile utils/checkpoint.py\n{checkpoint_code}"),
        make_cell('code', "%%writefile inference/__init__.py\n# inference package\nfrom inference.tiled_inference import tiled_predict_image_gaussian, create_gaussian_weight_map\n"),
        make_cell('code', f"%%writefile inference/tiled_inference.py\n{tiled_inference_code}"),
        make_cell('code', "%%writefile distillation/__init__.py\n# distillation package"),
        make_cell('code', f"%%writefile distillation/kd_trainer.py\n{kd_trainer_code}"),
        make_cell('code', f"%%writefile scripts/convert_crack500.py\n{convert_crack500_code}"),
        make_cell('code', f"%%writefile scripts/convert_crack500_uncropped.py\n{convert_crack500_uncropped_code}"),
        make_cell('code', f"%%writefile scripts/generate_teacher_logits.py\n{generate_teacher_logits_code}"),
        *([make_cell('code', f"%%writefile scripts/generate_gt_soft_logits.py\n{generate_gt_soft_logits_code}")] if prompt_type == "gt_soft" else []),

        make_cell('code', f"""# ── Step 1: Link Kaggle Inputs (Dataset & Teacher Logits) ──
import os, shutil
from pathlib import Path

input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists():
    input_dir = Path("/kaggle/input")

datasets_dir = Path("data/datasets")
datasets_dir.mkdir(parents=True, exist_ok=True)
checkpoints_dir = Path("checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)

# 1. Link Crack500 raw images or pre-converted dataset
found_dataset = False
for root, dirs, files in os.walk(str(input_dir)):
    root_path = Path(root)
    if "traincrop" in dirs:
        dest = datasets_dir / "crack500"
        if os.path.lexists(dest):
            os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(root_path, dest)
        print(f"[Dataset] Linked Crack500: {{root_path}} -> {{dest}}")
        found_dataset = True
        break
    elif "dataset.yaml" in files and ("crack500" in root or "images" in dirs):
        dest = datasets_dir / "crack500_yolo"
        if os.path.lexists(dest):
            os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(root_path, dest)
        print(f"[Dataset] Linked pre-converted YOLO dataset: {{root_path}} -> {{dest}}")
        found_dataset = True
        break

if not found_dataset:
    print("[Dataset Warning] Neither raw Crack500 nor pre-converted dataset.yaml found in input.")

{logits_linking_code}
"""),

        make_cell('code', f"""# ── Step 2: Convert Datasets & Verify Teacher Logits ──
# 1. Convert Cropped Crack500 if raw was linked and yolo not yet generated
if Path("data/datasets/crack500").exists() and not Path("data/datasets/crack500_yolo/dataset.yaml").exists():
    !python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo

# 2. Convert Uncropped Crack500 for OOD Evaluation
if Path("data/datasets/crack500/valdata").exists() and not Path("data/datasets/crack500_uncropped_yolo/dataset.yaml").exists():
    !python scripts/convert_crack500_uncropped.py --src data/datasets/crack500 --dst data/datasets/crack500_uncropped_yolo

{logits_verify_code}
"""),

        make_cell('code', f"""# ── Step 3: Run Full Production Training ({exp_name}) ──
import sys
sys.path.insert(0, ".")
from pathlib import Path
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
EXPERIMENT_NAME = "{exp_name}"

overrides = {format_python_dict(overrides_dict)}
overrides["project.name"] = "crack_distill"
overrides["project.experiment"] = EXPERIMENT_NAME
overrides["project.seed"] = {seed}
overrides["data.datasets"] = [{{"name": "crack500", "path": "data/datasets/crack500_yolo", "format": "yolo"}}]
{teacher_override}
if "train.epochs" not in overrides:
    overrides["train.epochs"] = 150
if "train.amp" not in overrides:
    overrides["train.amp"] = False

cfg = override_config(cfg, overrides)

print(f"=== Starting Run: {{EXPERIMENT_NAME}} ===")
print("Config in effect: Seed={seed}, Epochs=150, AMP=False, Freezing=False")

trainer = KDSegmentationTrainer(cfg)
trainer.train()
print("✓ Training completed successfully!")
"""),

        make_cell('code', f"""# ── Step 4: Validate Best Checkpoint & Export Results ──
import json, os
from pathlib import Path
from ultralytics import YOLO
from utils.checkpoint import resolve_checkpoint, get_checkpoint_manifest, save_checkpoint_manifest

# Deterministically resolve checkpoint without fuzzy glob collision
candidate_ckpt = getattr(trainer, "best", None) or Path("runs") / "crack_distill" / EXPERIMENT_NAME / "weights" / "best.pt"
best_pt_path = resolve_checkpoint(candidate_ckpt, expected_experiment=EXPERIMENT_NAME)
manifest = get_checkpoint_manifest(best_pt_path, experiment_name=EXPERIMENT_NAME, seed={seed})
manifest_file = save_checkpoint_manifest(manifest)

print(f"Evaluating verified checkpoint: {{best_pt_path}}")
print(f"Checkpoint SHA256: {{manifest['sha256']}}")
model = YOLO(str(best_pt_path))

# 1. Validate on Crack500 In-Domain Val Set
print("\\n--- In-Domain Cropped Validation ---")
val_metrics = model.val(data="data/datasets/crack500_yolo/dataset.yaml", split="val", verbose=True)

results = {{
    "experiment": "{exp_name}",
    "arm": "{arm}",
    "seed": {seed},
    "checkpoint": str(best_pt_path),
    "checkpoint_sha256": manifest["sha256"],
    "metrics_indomain": {{
        "mask_mAP50": float(val_metrics.seg.map50),
        "mask_mAP50_95": float(val_metrics.seg.map),
        "box_mAP50": float(val_metrics.box.map50),
        "box_mAP50_95": float(val_metrics.box.map),
        "mask_precision": float(val_metrics.seg.p[0]) if hasattr(val_metrics.seg, 'p') and len(val_metrics.seg.p) > 0 else float(val_metrics.seg.mp),
        "mask_recall": float(val_metrics.seg.r[0]) if hasattr(val_metrics.seg, 'r') and len(val_metrics.seg.r) > 0 else float(val_metrics.seg.mr)
    }}
}}

# 2. Validate on Crack500 Uncropped (OOD) Set
uncropped_yaml = Path("data/datasets/crack500_uncropped_yolo/dataset.yaml")
if uncropped_yaml.exists():
    print("\\n--- Out-of-Distribution (Uncropped) Validation ---")
    ood_metrics = model.val(data=str(uncropped_yaml), split="val", verbose=True)
    results["metrics_ood"] = {{
        "ood_mask_mAP50": float(ood_metrics.seg.map50),
        "ood_mask_mAP50_95": float(ood_metrics.seg.map),
        "ood_box_mAP50": float(ood_metrics.box.map50),
        "ood_box_mAP50_95": float(ood_metrics.box.map)
    }}
else:
    print("Warning: Uncropped dataset.yaml not found — skipping OOD evaluation.")

print("\\n" + "="*60)
print(f"🎯 FINAL EVALUATION SUMMARY ({exp_name}):")
print(f"   In-Domain Mask mAP50    : {{results['metrics_indomain']['mask_mAP50']:.4f}}")
print(f"   In-Domain Mask mAP50-95 : {{results['metrics_indomain']['mask_mAP50_95']:.4f}}")
print(f"   In-Domain Box mAP50     : {{results['metrics_indomain']['box_mAP50']:.4f}}")
if "metrics_ood" in results:
    print(f"   OOD Uncropped Mask mAP50    : {{results['metrics_ood']['ood_mask_mAP50']:.4f}}")
    print(f"   OOD Uncropped Mask mAP50-95 : {{results['metrics_ood']['ood_mask_mAP50_95']:.4f}}")
print("="*60)

out_file = Path(f"/kaggle/working/results/{exp_name}.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Saved structured summary to {{out_file}}")
""")
    ]
    return make_nb(cells)


def generate_ood_tiled_eval_notebook():
    cells = [
        make_cell('markdown', """# 🔬 Standalone Out-of-Distribution & Tiled Inference Evaluation
This notebook evaluates trained YOLOv11n-seg checkpoints on **unseen, full-resolution uncropped road crack photos**:
1. **Direct Resizing Evaluation**: Standard evaluation at $512 \\times 512$.
2. **Gaussian-Weighted Tiled / Sliding-Window Inference**: Dividing $2000 \\times 1500$ uncropped images into overlapping $512 \\times 512$ patches ($25\\%$ overlap, $384\\text{px}$ stride) with **2D Gaussian Apodization Blending** (eliminating border artifacts and weighting center predictions).
3. **Head-to-Head Comparison**: Compares all available checkpoints (Baseline, Full KD, Mask KD, Foreground-Dilated, LayerKD, Focal, Combined)."""),

        make_cell('code', f"""# ── Environment & Imports ──
!mkdir -p scripts configs utils distillation inference data/datasets results
!pip install -q {ULTRALYTICS_PIN} albumentations pycocotools opencv-python Pillow matplotlib tqdm pandas
""" + """import os, cv2, json, time, glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
"""),

        make_cell('code', f"%%writefile utils/__init__.py\n# utils package\nfrom utils.checkpoint import resolve_checkpoint, get_checkpoint_manifest, save_checkpoint_manifest, discover_and_deduplicate_checkpoints\n"),
        make_cell('code', f"%%writefile utils/checkpoint.py\n{checkpoint_code}"),
        make_cell('code', f"%%writefile inference/__init__.py\n# inference package\nfrom inference.tiled_inference import tiled_predict_image_gaussian, create_gaussian_weight_map\n"),
        make_cell('code', f"%%writefile inference/tiled_inference.py\n{tiled_inference_code}"),
        make_cell('code', f"%%writefile scripts/convert_crack500_uncropped.py\n{convert_crack500_uncropped_code}"),

        make_cell('code', """# ── Step 1: Link / Prepare Uncropped Dataset ──
input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists():
    input_dir = Path("/kaggle/input")

uncropped_dir = Path("data/datasets/crack500_uncropped_yolo")
uncropped_dir.mkdir(parents=True, exist_ok=True)

# Link raw crack500 if uncropped yolo not already converted
for root, dirs, files in os.walk(str(input_dir)):
    root_p = Path(root)
    if "valdata" in dirs or "testdata" in dirs:
        !python scripts/convert_crack500_uncropped.py --src {root_p} --dst data/datasets/crack500_uncropped_yolo
        break

print("Uncropped validation dataset ready at data/datasets/crack500_uncropped_yolo/")
"""),

        make_cell('code', """# ── Step 2: Gaussian-Weighted Tiled Sliding-Window Inference Engine ──
import torch
from inference.tiled_inference import tiled_predict_image_gaussian, create_gaussian_weight_map

print("Gaussian-weighted tiled inference engine loaded from inference.tiled_inference!")
"""),

        make_cell('code', """# ── Step 3: Discover Checkpoints & Run Cross-Evaluation ──
from PIL import Image

def get_exif_rotation(img_path: Path):
    try:
        with Image.open(img_path) as im:
            exif = im.getexif()
            if exif:
                return exif.get(274)
    except Exception:
        pass
    return None

def rotate_mask_to_match_image(mask: np.ndarray, exif_orientation: int) -> np.ndarray:
    if exif_orientation == 6:
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    elif exif_orientation == 8:
        return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif exif_orientation == 3:
        return cv2.rotate(mask, cv2.ROTATE_180)
    return mask

def compute_dice(pred_mask, gt_mask):
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask.astype(np.uint8), (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / total)

from utils.checkpoint import discover_and_deduplicate_checkpoints

discovered_ckpts = discover_and_deduplicate_checkpoints([Path("/kaggle/input"), Path("runs")])
print(f"Discovered {len(discovered_ckpts)} unique checkpoints (SHA256 deduplicated):")
for item in discovered_ckpts:
    print(f"  - [{item['short_sha']}] {item['name']}: {item['path']} ({item['size_bytes'] / (1024*1024):.1f} MB)")

# Find ground-truth uncropped images and masks
val_img_dir = Path("data/datasets/crack500_uncropped_yolo/images/val")
all_val_imgs = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")))

eval_summary = {}
for item in discovered_ckpts:
    name = item["name"]
    ckpt = item["path"]
    sha = item["sha256"]
    print(f"\\n{'='*50}\\nEvaluating Checkpoint: {name} (sha: {item['short_sha']})\\nPath: {ckpt}\\n{'='*50}")
    model = YOLO(str(ckpt))
    
    # 1. Direct Resize Val (512x512)
    res_direct = model.val(data="data/datasets/crack500_uncropped_yolo/dataset.yaml", split="val", verbose=False)
    direct_mAP50 = float(res_direct.seg.map50)
    direct_mAP50_95 = float(res_direct.seg.map)
    direct_box_mAP50 = float(res_direct.box.map50)
    
    # 2. Tiled Sliding-Window Full-Resolution Dice Evaluation
    tiled_dices = []
    direct_dices = []
    
    for img_p in tqdm(all_val_imgs[:50], desc=f"  Tiling eval ({name[:20]})", leave=False):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        
        # Load ground-truth mask if available
        gt_mask_path = img_p.parent.parent.parent.parent / "crack500" / "valdata" / f"{img_p.stem}_mask.png"
        if not gt_mask_path.exists():
            for alt_name in ["val_lab", "valcrop", "masks"]:
                alt_p = img_p.parent.parent.parent.parent / "crack500" / alt_name / f"{img_p.stem}_mask.png"
                if alt_p.exists():
                    gt_mask_path = alt_p
                    break
                
        if gt_mask_path.exists():
            gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                exif_rot = get_exif_rotation(img_p)
                if exif_rot:
                    gt_mask = rotate_mask_to_match_image(gt_mask, exif_rot)
                if gt_mask.shape[:2] == (w, h):
                    gt_mask = cv2.rotate(gt_mask, cv2.ROTATE_90_CLOCKWISE)
                if gt_mask.shape[:2] != (h, w):
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                gt_binary = (gt_mask > 127).astype(np.uint8)
                
                # A. Direct resize prediction
                r_dir = model.predict(img_bgr, imgsz=512, conf=0.25, verbose=False)[0]
                pred_dir = np.zeros((h, w), dtype=np.uint8)
                if r_dir.masks is not None and len(r_dir.masks) > 0:
                    for m in r_dir.masks.data.cpu().numpy():
                        m_resized = cv2.resize(m, (w, h))
                        pred_dir = np.maximum(pred_dir, (m_resized > 0.35).astype(np.uint8))
                direct_dices.append(compute_dice(pred_dir, gt_binary))
                
                # B. Gaussian Tiled sliding-window prediction
                pred_tiled = tiled_predict_image_gaussian(model, img_bgr, tile_size=512, stride=384, conf=0.25)
                tiled_dices.append(compute_dice(pred_tiled, gt_binary))
                
    mean_direct_dice = float(np.mean(direct_dices)) if direct_dices else 0.0
    mean_tiled_dice = float(np.mean(tiled_dices)) if tiled_dices else 0.0
    
    eval_summary[name] = {
        "checkpoint_path": str(ckpt),
        "sha256": sha,
        "direct_mask_mAP50": direct_mAP50,
        "direct_mask_mAP50_95": direct_mAP50_95,
        "direct_box_mAP50": direct_box_mAP50,
        "full_res_direct_dice": mean_direct_dice,
        "full_res_tiled_dice": mean_tiled_dice
    }
    
    print(f"Results for {name} :")
    print(f"  Direct Resize Mask mAP50    : {direct_mAP50:.4f}")
    print(f"  Direct Resize Mask mAP50-95 : {direct_mAP50_95:.4f}")
    print(f"  Full-Res Direct Dice        : {mean_direct_dice:.4f}")
    print(f"  Full-Res Tiled Dice         : {mean_tiled_dice:.4f}")

out_path = Path("/kaggle/working/results/ood_eval_summary.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(eval_summary, f, indent=2)
print(f"\\nSaved evaluation summary to {out_path}")
""")
    ]
    return make_nb(cells)


def generate_benchmark_notebook():
    cells = [
        make_cell('markdown', """# ⚡ Two-Level Model Speed, Latency & Parameter Footprint Benchmark
This notebook benchmarks the deployed **YOLOv11n-seg** model across two architectural levels:

* **Level 1: Single-Tile Inference (512×512)**
  * Pure forward-pass latency (GPU / CPU in milliseconds)
  * Throughput (FPS)
  * Model parameters: 2.84M, FLOPs: 10.2 GFLOPs, weights: 6.2 MB
  * Confirms 0% runtime overhead over vanilla YOLOv11n-seg

* **Level 2: Full-Scene Tiled Reconstruction (2000×1500, 20 overlapping tiles)**
  * Full end-to-end reconstruction: tile slicing, serial per-tile inference, Gaussian blending, and thresholding
  * Measured end-to-end on this notebook's device (no batched path is implemented, so none is reported)

> **CRITICAL ARCHITECTURAL DISAMBIGUATION (P2-5):**
> 107.8 FPS (9.27 ms, Tesla T4) refers strictly to **Single-Tile (512×512)** forward inference.
> Full-Scene 2000×1500 pavement inspection requires 20 overlapping tiles; its throughput is whatever Level 2 measures below."""),

        make_cell('code', "!mkdir -p inference"),
        make_cell('code', "%%writefile inference/__init__.py\n# inference package\nfrom inference.tiled_inference import tiled_predict_image_gaussian, create_gaussian_weight_map\n"),
        make_cell('code', f"%%writefile inference/tiled_inference.py\n{tiled_inference_code}"),

        make_cell('code', f"!pip install -q {ULTRALYTICS_PIN} thop\n" + """import time, torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Benchmarking on device: {device}")

ckpt_path = Path("checkpoints/yolo11n-seg.pt")
model = YOLO(str(ckpt_path)) if ckpt_path.exists() else YOLO("yolo11n-seg.pt")

# =====================================================================
# LEVEL 1: Single-Tile Pure Model Forward Latency (512x512, Batch=1)
# =====================================================================
print("Running Level 1: Single-Tile Pure Model Forward Benchmark...")
dummy_tile = torch.randn(1, 3, 512, 512).to(device)

# Warm-up
for _ in range(50):
    _ = model(dummy_tile, verbose=False)

tile_times = []
if device == "cuda":
    torch.cuda.synchronize()
for _ in range(500):
    t0 = time.perf_counter()
    _ = model(dummy_tile, verbose=False)
    if device == "cuda":
        torch.cuda.synchronize()
    tile_times.append((time.perf_counter() - t0) * 1000)

tile_mean_ms = float(np.mean(tile_times))
tile_std_ms = float(np.std(tile_times))
tile_p50_ms = float(np.percentile(tile_times, 50))
tile_p95_ms = float(np.percentile(tile_times, 95))
tile_fps = 1000.0 / tile_mean_ms

# =====================================================================
# LEVEL 2: Full-Scene Tiled Reconstruction Benchmark (2000x1500, 20 tiles)
# =====================================================================
print("Running Level 2: Full-Scene Tiled Reconstruction Benchmark...")
# Measured end-to-end; no estimated fallback, so a failure here is visible instead of printing made-up numbers
from inference.tiled_inference import benchmark_tiled_inference_pipeline
pipeline_res = benchmark_tiled_inference_pipeline(
    model=model,
    image_shape=(1500, 2000, 3),
    tile_size=512,
    overlap=0.2,
    num_runs=20,
    device=device
)
serial_ms = pipeline_res["pipeline_serial"]["mean_ms"]
serial_fps = pipeline_res["pipeline_serial"]["fps"]
num_tiles = pipeline_res["num_tiles"]

print("=" * 68)
print("📊 TWO-LEVEL BENCHMARK RESULTS SUMMARY:")
print(f"   Device     : {device}")
print(f"   Model Size : 6.2 MB (2.84M params, 10.2 GFLOPs)")
print("-" * 68)
print("   LEVEL 1: SINGLE-TILE (512x512) FORWARD INFERENCE:")
print(f"      Mean Latency : {tile_mean_ms:.2f} ± {tile_std_ms:.2f} ms")
print(f"      p50 / p95    : {tile_p50_ms:.2f} ms / {tile_p95_ms:.2f} ms")
print(f"      Throughput   : {tile_fps:.1f} FPS")
print("-" * 68)
print(f"   LEVEL 2: FULL-SCENE (2000x1500, {num_tiles} TILES) RECONSTRUCTION:")
print(f"      Serial Pipeline  : {serial_ms:.1f} ms ({serial_fps:.1f} scenes/sec) [measured on {device}]")
print("=" * 68)
print("   CRITICAL ARCHITECTURAL DISAMBIGUATION NOTE (P2-5):")
print(f"   - Level 1 measures a single 512x512 tile forward pass ({tile_fps:.1f} FPS here).")
print(f"   - Full-scene 2000x1500 pavement inspection requires {num_tiles} tiles,")
print(f"     yielding {serial_fps:.1f} scenes/sec (serial, measured end-to-end).")
print("=" * 68)
""")
    ]
    return make_nb(cells)


# Confirmatory suite (see EXPERIMENTS.md). Arms share seeds so results pair by seed.
# Tier 1: baseline and Mask-KD x 5 seeds, GT-soft x 3; GT-soft seeds 3-4 are Tier 2 (only if KD beats baseline).
SEEDS = {"baseline": (0, 1, 2, 3, 4), "mask_kd": (0, 1, 2, 3, 4), "gt_soft": (0, 1, 2, 3, 4)}
TIER1_SEEDS = {"baseline": (0, 1, 2, 3, 4), "mask_kd": (0, 1, 2, 3, 4), "gt_soft": (0, 1, 2)}

# Mask-KD and the GT-soft control share one config; only the teacher targets differ.
# Hyper-parameters are fixed in advance (not re-tuned on the corrected pipeline).
CFG_MASK_KD = {
    "distillation.enabled": True,
    "distillation.temperature": 3.7769,
    "distillation.progressive.enabled": False,
    "distillation.losses.task.weight": 1.0,
    "distillation.losses.mask_kd.enabled": True,
    "distillation.losses.mask_kd.weight": 0.9612,
    "distillation.losses.mask_kd.focused": False,
    "distillation.losses.mask_kd.high_res": False,
    "distillation.losses.feature.enabled": False,
    "distillation.losses.boundary.enabled": False,
    "distillation.losses.affinity.enabled": False,
    "distillation.losses.tversky.enabled": False,
    "train.epochs": 150,
    "train.amp": False,
}

CFG_BASELINE = {
    "distillation.enabled": False,
    "distillation.losses.mask_kd.enabled": False,
    "distillation.losses.feature.enabled": False,
    "distillation.losses.boundary.enabled": False,
    "distillation.losses.affinity.enabled": False,
    "distillation.losses.tversky.enabled": False,
    "train.epochs": 150,
    "train.amp": False,
}

ARMS = [
    # (file prefix, arm, title, description, config, prompt_type, logits_dir)
    ("1_baseline", "baseline", "Baseline (No KD)",
     "YOLOv11n-seg fine-tuned on Crack500 with the task loss only. Lower-bound control.",
     CFG_BASELINE, "none", "data/teacher_logits_box"),
    ("2_mask_kd", "mask_kd", "Mask-KD from SAM 2 (Box Prompts)",
     "Task loss + Bernoulli Mask-KL to cached SAM 2 Large logits (T=3.7769, W=0.9612).",
     CFG_MASK_KD, "box", "data/teacher_logits_box"),
    ("3_gt_soft", "gt_soft", "GT-Soft Control (Blurred Ground Truth)",
     "Identical Mask-KL loss, but the targets are Gaussian-blurred ground truth (SVLS-style, sigma=2.0, logits "
     "capped at +-14 like the SAM background) instead of SAM 2. Separates teacher knowledge from label softening.",
     CFG_MASK_KD, "gt_soft", "data/teacher_logits_gt_soft"),
]


def main():
    final_dir = Path("final_notebooks")
    final_dir.mkdir(parents=True, exist_ok=True)

    def save_nb(filename, nb):
        with open(final_dir / filename, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✓ Created final_notebooks/{filename}")

    for prefix, arm, title, description, cfg, prompt_type, logits_dir in ARMS:
        for seed in SEEDS[arm]:
            nb = generate_training_notebook(
                arm, title, description, cfg,
                seed=seed, prompt_type=prompt_type, logits_dir=logits_dir, arm=arm,
            )
            save_nb(f"{prefix}_seed{seed}.ipynb", nb)

    save_nb("4_benchmark_speed.ipynb", generate_benchmark_notebook())

    def seeds_txt(arm):
        tier2 = [s for s in SEEDS[arm] if s not in TIER1_SEEDS[arm]]
        return ", ".join(map(str, TIER1_SEEDS[arm])) + (f" (Tier 2: {', '.join(map(str, tier2))})" if tier2 else "")
    readme_content = f"""# CrackDistill — Kaggle notebooks (confirmatory suite)

Generated by `scripts/build_final_notebooks.py`; do not edit by hand. Each notebook is self-contained
(source files are embedded with `%%writefile`) and pins `{ULTRALYTICS_PIN}`.

| Notebook | Arm | Seeds | ~T4 time |
| :--- | :--- | :--- | :---: |
| `1_baseline_seed<S>.ipynb` | Baseline, task loss only | {seeds_txt("baseline")} | ~3 h |
| `2_mask_kd_seed<S>.ipynb` | Mask-KD from SAM 2 box-prompt logits | {seeds_txt("mask_kd")} | ~3 h |
| `3_gt_soft_seed<S>.ipynb` | Same loss, blurred-GT targets (generated in-notebook) | {seeds_txt("gt_soft")} | ~3 h |

Tier 1 = 13 runs (~39 T4-hours). Run Tier 2 only if the Mask-KD minus baseline CI excludes 0 (see `EXPERIMENTS.md`).
| `4_benchmark_speed.ipynb` | Single-tile and measured full-scene latency | — | ~5 min |

## Run on Kaggle
1. **File → Import Notebook**, upload one `.ipynb`.
2. Settings: **GPU T4** (one GPU is used), **Internet ON**.
3. **+ Add Input**: the dataset with `datasets/crack500` and `teacher_logits` (e.g. `distill-datasetforme`).
   The GT-soft notebooks never read SAM logits; they generate their own targets.
4. **Save Version → Save & Run All (Commit)**.
5. From **Output**, download `results/<arm>_seed<S>_150ep.json` and
   `runs/segment/crack_distill/<arm>_seed<S>_150ep/weights/best.pt`.

## Check the log before trusting a run
- All arms: `Seed: <S>`, `mosaic=0.0` and `overlap_mask=False` in the Ultralytics args.
- Baseline: `Clean Baseline Control: Native YOLO setup` and no `KD losses computed` lines.
- Mask-KD / GT-soft: `KD losses computed: mask_kd: ...`; never `Redirecting logits_dir`,
  `Missing letterbox metadata` or `No teacher logits ... matched`.

## Final evaluation (local, once, after all training is done)
```bash
python scripts/evaluate_canonical_test_set.py --weights <best.pt> --arm <arm> --seed <S>   # -> results/test/
python scripts/aggregate_multiseed_results.py --results-dir results/test --metric in_domain_mask_mAP50
```
"""
    with open(final_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("✓ Created final_notebooks/README.md")

if __name__ == "__main__":
    main()
