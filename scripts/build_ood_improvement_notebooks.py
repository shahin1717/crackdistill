#!/usr/bin/env python3
"""
Builds the self-contained Kaggle notebook suite in OODimprovements/, implementing
the complete out-of-distribution generalization plan from possibleOODimprovements.md
and next_moves_forOOD.md:

  01 -> mine mosaic composites (+ pilot negative tiles) from the existing traincrop grid
  02 -> generate native-resolution SAM2 teacher logits on those mosaics
  03 -> train the CWD+dilated recipe (from final_notebooks/10) on the mosaic-augmented
        training set with native-scale teacher supervision
  04 -> Gaussian-tiled OOD evaluation (cross-checkpoint comparison on raw megapixel imagery)
  05 -> Two-Stage Transfer Fine-Tuning (Crop-trained backbone -> Mosaic set, 50 epochs)
  06 -> Resolution-Preserving Mask Distillation (Upsampled student proto-mask before KL)
  07 -> Asymmetric Soft Tversky Loss (beta=0.70 false-negative penalty for thin cracks)
  08 -> Multi-Scale Gaussian TTA & Sliced Inference Engine (640+768 Gaussian blending)

Run (from ~/distill):
  python scripts/build_ood_improvement_notebooks.py
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
OUT_DIR = ROOT / "OODimprovements"

with open(ROOT / "configs/config.yaml") as f:
    config_yaml = f.read()
with open(ROOT / "utils/config_loader.py") as f:
    config_loader_code = f.read()
with open(ROOT / "distillation/kd_trainer.py") as f:
    kd_trainer_code = f.read()
with open(ROOT / "scripts/convert_crack500.py") as f:
    convert_crack500_code = f.read()
with open(ROOT / "scripts/convert_crack500_uncropped.py") as f:
    convert_crack500_uncropped_code = f.read()
with open(ROOT / "scripts/generate_teacher_logits.py") as f:
    generate_teacher_logits_code = f.read()
with open(ROOT / "scripts/mine_negative_and_mosaic_tiles.py") as f:
    mine_code = f.read()
with open(ROOT / "scripts/build_augmented_training_set.py") as f:
    build_augmented_code = f.read()


def make_cell(cell_type, source):
    if isinstance(source, str):
        lines = [line + "\n" for line in source.split("\n")]
        if lines and lines[-1] == "\n":
            lines.pop()
        source = lines
    return {"cell_type": cell_type, "metadata": {}, "outputs": [], "source": source}


def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }


LINK_INPUTS_CELL = """# -- Step 1: Link Kaggle Inputs (Dataset & Teacher Logits) --
import os, shutil
from pathlib import Path

input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists():
    input_dir = Path("/kaggle/input")

datasets_dir = Path("data/datasets")
datasets_dir.mkdir(parents=True, exist_ok=True)

found_dataset = False
for root, dirs, files in os.walk(str(input_dir)):
    root_path = Path(root)
    if "traincrop" in dirs:
        dest = datasets_dir / "crack500"
        if os.path.lexists(dest):
            os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
        os.symlink(root_path, dest)
        print(f"[Dataset] Linked Crack500: {root_path} -> {dest}")
        found_dataset = True
        break
assert found_dataset, "Could not find traincrop/ inside the attached dataset."
"""


def make_training_setup_cells():
    return [
        make_cell("code", f"%%writefile configs/config.yaml\n{config_yaml}"),
        make_cell("code", "%%writefile utils/__init__.py\n# utils package"),
        make_cell("code", f"%%writefile utils/config_loader.py\n{config_loader_code}"),
        make_cell("code", "%%writefile distillation/__init__.py\n# distillation package"),
        make_cell("code", f"%%writefile distillation/kd_trainer.py\n{kd_trainer_code}"),
        make_cell("code", f"%%writefile scripts/convert_crack500.py\n{convert_crack500_code}"),
        make_cell("code", f"%%writefile scripts/convert_crack500_uncropped.py\n{convert_crack500_uncropped_code}"),
        make_cell("code", f"%%writefile scripts/mine_negative_and_mosaic_tiles.py\n{mine_code}"),
        make_cell("code", f"%%writefile scripts/build_augmented_training_set.py\n{build_augmented_code}"),
    ]


LINK_OR_BUILD_INPUTS_CELL = """# -- Step 1: Link or Auto-Build Augmented Dataset & Teacher Logits --
import os, shutil, zipfile
from pathlib import Path

def find_and_link(marker_dir_names, dest_path):
    if isinstance(marker_dir_names, str):
        marker_dir_names = [marker_dir_names]
    for root, dirs, files in os.walk("/kaggle/input"):
        for m in marker_dir_names:
            if m in dirs:
                src = Path(root) / m
                dest = Path(dest_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if os.path.lexists(dest):
                    os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
                os.symlink(src, dest)
                print(f"[Link] {src} -> {dest}")
                return True
    return False

# 1. Look for pre-attached crack500_yolo_augmented (from notebook 01)
ok1 = find_and_link("crack500_yolo_augmented", "data/datasets/crack500_yolo_augmented")

# If not pre-attached, auto-build on-the-fly from raw crack500 dataset (~60s)
if not ok1:
    print("[Dataset] 'crack500_yolo_augmented' not pre-attached.")
    print("  Searching for raw Crack500 dataset in /kaggle/input to auto-build...")
    has_yolo = find_and_link(["crack500_yolo"], "data/datasets/crack500_yolo")
    if not has_yolo:
        for root, dirs, files in os.walk("/kaggle/input"):
            if "traincrop" in dirs:
                raw_c500 = Path(root)
                dest_c500 = Path("data/datasets/crack500")
                dest_c500.parent.mkdir(parents=True, exist_ok=True)
                if os.path.lexists(dest_c500):
                    os.unlink(dest_c500) if os.path.islink(dest_c500) else shutil.rmtree(dest_c500)
                os.symlink(raw_c500, dest_c500)
                print(f"[Dataset] Found raw Crack500 at {raw_c500}, converting to YOLO format...")
                !python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo
                has_yolo = True
                break

    if has_yolo:
        print("[Dataset] Mining 250 mosaic composites & building augmented training set (~30s)...")
        !python scripts/mine_negative_and_mosaic_tiles.py
        !python scripts/build_augmented_training_set.py
        ok1 = Path("data/datasets/crack500_yolo_augmented").exists()

assert ok1, ("Could not find 'crack500_yolo_augmented' or raw 'distill_datasetforme' (with traincrop/) in /kaggle/input. "
             "Please click '+ Add Data' and attach 'distill_datasetforme' (or Notebook 01 output).")

# 2. Look for teacher logits (notebook 02 output or raw teacher_logits from distill_datasetforme)
ok2 = find_and_link(["teacher_logits_box", "teacher_logits"], "data/teacher_logits_box")
if not ok2:
    print("[Logits] 'teacher_logits_box' not found; searching for any teacher_logits directory in /kaggle/input...")
    for root, dirs, files in os.walk("/kaggle/input"):
        for d in dirs:
            if "teacher_logits" in d or "logits" in d:
                src = Path(root) / d
                dest = Path("data/teacher_logits_box")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if os.path.lexists(dest):
                    os.unlink(dest) if os.path.islink(dest) else shutil.rmtree(dest)
                os.symlink(src, dest)
                print(f"[Link] Linked fallback logits {src} -> {dest}")
                ok2 = True
                break
        if ok2:
            break

if not ok2:
    print("[Logits Warning] No teacher logits found in /kaggle/input.")
    print("  Creating empty logits directory: trainer will use student ground-truth supervision for missing logits.")
    Path("data/teacher_logits_box").mkdir(parents=True, exist_ok=True)

# 3. Discover Stage 1 Checkpoint (for twostage fine-tuning, handles .pt and unzipped folders)
repack_dir = Path("/kaggle/working/repacked_ckpts")
repack_dir.mkdir(parents=True, exist_ok=True)
stage1_ckpt = None
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "data.pkl" in files:
            folder_name = os.path.basename(root)
            if folder_name in [".", ""]:
                folder_name = os.path.basename(os.path.dirname(root))
            repacked_path = repack_dir / f"{folder_name}.pt"
            with zipfile.ZipFile(repacked_path, 'w', compression=zipfile.ZIP_STORED) as zf:
                for r, d, f_list in os.walk(root):
                    for file_name in f_list:
                        full_file = os.path.join(r, file_name)
                        rel_file = os.path.relpath(full_file, root)
                        arc_name = os.path.join("archive", rel_file).replace("\\\\", "/")
                        zf.write(full_file, arc_name)
            stage1_ckpt = str(repacked_path)
            print(f"[Stage 1] Repacked unzipped folder checkpoint: {stage1_ckpt}")
            break

if not stage1_ckpt:
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith(".pt") and "sam" not in f.lower() and "hiera" not in f.lower():
                stage1_ckpt = os.path.join(root, f)
                print(f"[Stage 1] Found .pt checkpoint: {stage1_ckpt}")
                break
        if stage1_ckpt:
            break

if not stage1_ckpt:
    print("[Stage 1] No prior checkpoint attached; falling back to stock yolo11n-seg.pt")
    stage1_ckpt = "yolo11n-seg.pt"

# 4. Uncropped OOD val set, from the raw dataset attachment (for post-training validation)
input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists():
    input_dir = Path("/kaggle/input")
for root, dirs, files in os.walk(str(input_dir)):
    if "valdata" in dirs:
        !python scripts/convert_crack500_uncropped.py --src {root} --dst data/datasets/crack500_uncropped_yolo
        break
"""


def notebook_01():
    cells = [
        make_cell("markdown", """# OODimprovements 01: Mine Mosaic Composites & Pilot Negative Tiles
Builds two new training assets purely from the existing Crack500 `traincrop` grid — no new data collection:
1. **Mosaic composites**: wherever 2+ crop tiles for the same source photo are grid-adjacent, stitches them (image + mask) into a larger real composite. Verified locally: **250/250 source photos** produce a composite, up to 1920x720 (vs the 640x360 ceiling of individual crops).
2. **Pilot negative tiles**: background-only crops, extracted from unused grid cells of source photos whose *exact filename* also appears in `valdata`/`testdata`. Verified locally: only **5/250 stems match, yielding 10 crops** — too small to be a real fix, and those 5 stems overlap the OOD eval set by photo, so treat this as a pilot/diagnostic only, not a production negative-tile set. See `possibleOODimprovements.md` for the full writeup of why this path is limited.

Output: `data/datasets/crack500_ood_mined/` (mosaics + pilot negatives) and `data/datasets/crack500_yolo_augmented/` (base 1896-image training set + the above merged in, val/test untouched)."""),
        make_cell("code", """# -- Environment & Directory Initialization --
!mkdir -p scripts data/datasets configs utils distillation
!pip install -q opencv-python numpy tqdm
"""),
        make_cell("code", LINK_INPUTS_CELL),
        make_cell("code", f"%%writefile scripts/convert_crack500.py\n{convert_crack500_code}"),
        make_cell("code", "!mkdir -p data/datasets/crack500_yolo\n"
                           "# crack500_yolo must already exist (converted cropped dataset) before augmenting it.\n"
                           "!python scripts/convert_crack500.py --src data/datasets/crack500 --dst data/datasets/crack500_yolo"),
        make_cell("code", f"%%writefile scripts/mine_negative_and_mosaic_tiles.py\n{mine_code}"),
        make_cell("code", "!python scripts/mine_negative_and_mosaic_tiles.py"),
        make_cell("code", f"%%writefile scripts/build_augmented_training_set.py\n{build_augmented_code}"),
        make_cell("code", "!python scripts/build_augmented_training_set.py"),
        make_cell("code", """# -- Verify output (this becomes the Kaggle Notebook Output for 02 and 03 to attach) --
from pathlib import Path
mined = Path("data/datasets/crack500_ood_mined")
aug = Path("data/datasets/crack500_yolo_augmented")
print("mosaic composites:", len(list((mined / "mosaic_images").glob("*.jpg"))))
print("pilot negatives   :", len(list((mined / "negative_images").glob("*.jpg"))))
print("augmented train set images:", len(list((aug / "images/train").glob("*.jpg"))))
print("\\nSave this notebook's version so 02 and 03 can attach it via '+ Add Data -> Your Work -> Notebook Output Files'.")
"""),
    ]
    return make_nb(cells)


def notebook_02():
    cells = [
        make_cell("markdown", """# OODimprovements 02: Native-Resolution SAM2 Teacher Logits
Runs SAM2 against the mosaic composites from notebook 01 (up to 1920x720, i.e. genuinely beyond the 640x360 ceiling every prior teacher-logit set was capped at — confirmed by inspecting `generate_teacher_logits.py`'s `DATASET_DIR`, which only ever pointed at `crack500` crops until now). Output logit filenames use the existing `crack500_` prefix convention so `KDSegmentationTrainer`'s stem-matching in `kd_trainer.py` resolves them with no code changes.

**Requires**: attach notebook 01's output (`+ Add Data -> Your Work -> Notebook Output Files`)."""),
        make_cell("code", """# -- Environment & Directory Initialization --
!mkdir -p scripts checkpoints data/teacher_logits_box data/datasets configs utils distillation
!pip install -q ultralytics opencv-python numpy
"""),
        make_cell("code", """# -- Link notebook 01 output (mosaics only) --
import os, shutil
from pathlib import Path

mosaic_img_src = None
mosaic_mask_src = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "mosaic_images" in dirs:
        mosaic_img_src = Path(root) / "mosaic_images"
    if "mosaic_masks" in dirs:
        mosaic_mask_src = Path(root) / "mosaic_masks"

assert mosaic_img_src is not None and mosaic_mask_src is not None, (
    "Could not find 'mosaic_images' or 'mosaic_masks' in /kaggle/input! "
    "Please attach Notebook 01's output before running."
)

dest_img = Path("data/datasets/crack500_ood_mined/mosaic_images")
dest_mask = Path("data/datasets/crack500_ood_mined/mosaic_masks")
dest_img.parent.mkdir(parents=True, exist_ok=True)

if os.path.lexists(dest_img):
    os.unlink(dest_img) if os.path.islink(dest_img) else shutil.rmtree(dest_img)
os.symlink(mosaic_img_src, dest_img)

if os.path.lexists(dest_mask):
    os.unlink(dest_mask) if os.path.islink(dest_mask) else shutil.rmtree(dest_mask)
os.symlink(mosaic_mask_src, dest_mask)

img_count = len(list(dest_img.glob("*.jpg"))) + len(list(dest_img.glob("*.png")))
print(f"[Link] Successfully linked {img_count} mosaic images from {mosaic_img_src} -> {dest_img}")
"""),
        make_cell("code", f"%%writefile scripts/generate_teacher_logits.py\n{generate_teacher_logits_code}"),
        make_cell("code", """# -- Generate native-scale logits for the mosaic composites --
import subprocess, sys
from pathlib import Path

ckpt_file = Path("checkpoints/sam2_hiera_large.pt")
ckpt_file.parent.mkdir(parents=True, exist_ok=True)
if not ckpt_file.exists():
    !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O {ckpt_file}
!pip install -q git+https://github.com/facebookresearch/segment-anything-2.git || pip install -q SAM-2

cmd = [
    sys.executable, "scripts/generate_teacher_logits.py",
    "--img-dir", "data/datasets/crack500_ood_mined/mosaic_images",
    "--mask-dir", "data/datasets/crack500_ood_mined/mosaic_masks",
    "--prefix", "crack500_",
    "--logits-dir", "data/teacher_logits_box",
    "--sam-ckpt", str(ckpt_file),
    "--resume"
]
print("Running command:", " ".join(cmd))
res = subprocess.run(cmd)
if res.returncode != 0:
    raise RuntimeError(f"generate_teacher_logits.py failed with exit code {res.returncode}")

def count_mosaic_logits():
    found = {}
    for c in [Path("data/teacher_logits_box"), Path("/tmp/teacher_logits_box")]:
        if c.exists():
            found[str(c)] = len(list(c.glob("*_mosaic_logits.npy")))
    return found

counts = count_mosaic_logits()
print(f"[Verify] Mosaic logit counts by location: {counts}")
n = max(counts.values(), default=0)
assert n > 0, f"0 mosaic logits generated anywhere (checked {list(counts.keys())}) — check the generation log above for per-image errors."

best_loc = max(counts, key=counts.get)
if best_loc != "data/teacher_logits_box" and counts[best_loc] > counts.get("data/teacher_logits_box", 0):
    dest = Path("data/teacher_logits_box")
    dest.mkdir(parents=True, exist_ok=True)
    for f in Path(best_loc).glob("*_mosaic_logits.npy"):
        link = dest / f.name
        if not link.exists():
            os.symlink(f.resolve(), link)
    print(f"[Fix] Linked {counts[best_loc]} files from {best_loc} into data/teacher_logits_box")
"""),
        make_cell("code", """# -- Merge in the ORIGINAL crop-scale teacher logits (after generation, not before) --
import os
from pathlib import Path

input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists():
    input_dir = Path("/kaggle/input")

dest = Path("data/teacher_logits_box")
dest.mkdir(parents=True, exist_ok=True)
for root, dirs, files in os.walk(str(input_dir)):
    if "teacher_logits_box" in dirs or "teacher_logits" in dirs:
        src = Path(root) / ("teacher_logits_box" if "teacher_logits_box" in dirs else "teacher_logits")
        n = 0
        for f in src.glob("*.npy"):
            link = dest / f.name
            if not link.exists():
                os.symlink(f, link)
                n += 1
        print(f"[Link] Merged {n} existing crop-scale logits from {src} -> {dest}")
        break

total = len(list(dest.glob("*_logits.npy")))
mosaic_total = len(list(dest.glob("*_mosaic_logits.npy")))
print(f"[Final] {dest}: {total} total logit files ({mosaic_total} native-scale mosaic + {total - mosaic_total} crop-scale)")
"""),
        make_cell("code", """print("Save this notebook's version — 03 needs data/teacher_logits_box (crop-scale + mosaic-scale merged) "
      "and notebook 01's crack500_yolo_augmented attached.")"""),
    ]
    return make_nb(cells)


def notebook_03():
    overrides_repr = """{
    'distillation.enabled': True,
    'distillation.temperature': 3.7769,
    'distillation.progressive.enabled': False,
    'distillation.losses.task.weight': 1.0,
    'distillation.losses.mask_kd.enabled': True,
    'distillation.losses.mask_kd.weight': 0.9612,
    'distillation.losses.mask_kd.focused': True,
    'distillation.losses.mask_kd.high_res': False,
    'distillation.losses.affinity.enabled': False,
    'distillation.losses.feature.enabled': True,
    'distillation.losses.feature.method': 'cwd',
    'distillation.losses.feature.weight': 0.25,
    'distillation.losses.feature.temperature': 4.0,
    'distillation.losses.feature.layers': [12, 15, 18],
    'distillation.losses.boundary.enabled': False,
    'student.imgsz': 640,
    'data.image_size': 640,
    'data.batch_size': 12,
    'train.lr': 0.001,
    'train.epochs': 150,
    'train.amp': False,
}"""
    cells = [
        make_cell("markdown", """# OODimprovements 03: CWD + Foreground-Dilated KD on the Mosaic-Augmented Set
Same loss recipe as `final_notebooks/10_run_layerkd_dilated_hires` (Neck CWD layers 12/15/18 + 8px foreground-dilated mask-KL), but:
- Trains on `crack500_yolo_augmented` (1896 original crops + 250 mosaic composites, up to 1920x720) instead of only the 640x360-capped crops.
- `imgsz=640`: matches the native single-tile width so the majority of the set isn't force-upsampled.
- Teacher logits merge the original crop-scale set with the native-scale mosaic logits generated in notebook 02.

**Trains from the stock pretrained `yolo11n-seg.pt` backbone, like every other notebook in this project** — no prior experiment's `best.pt` is used as a starting point.

**Requires**: attach BOTH notebook 01's output (`crack500_yolo_augmented`) and notebook 02's output (`data/teacher_logits_box` with mosaic logits merged in)."""),
        make_cell("code", """# -- Environment & Directory Initialization --
!mkdir -p configs utils distillation scripts checkpoints data/datasets data/teacher_logits_box runs results
!pip install -q ultralytics albumentations pycocotools thop pyyaml pandas tqdm opencv-python Pillow
"""),
        *make_training_setup_cells(),
        make_cell("code", LINK_OR_BUILD_INPUTS_CELL),
        make_cell("code", f"""# -- Step 2: Run Training (exp_mosaic_augmented_hires_layerkd_dilated_T3.7769_W0.9612_seed42_150ep) --
import sys
sys.path.insert(0, ".")
from pathlib import Path
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
EXPERIMENT_NAME = "exp_mosaic_augmented_hires_layerkd_dilated_T3.7769_W0.9612_seed42_150ep"

overrides = {overrides_repr}
overrides["project.name"] = "crack_distill"
overrides["project.experiment"] = EXPERIMENT_NAME
overrides["project.seed"] = 42
overrides["data.datasets"] = [{{"name": "crack500_augmented", "path": "data/datasets/crack500_yolo_augmented", "format": "yolo"}}]
overrides["teacher.logits_dir"] = "data/teacher_logits_box/"

cfg = override_config(cfg, overrides)

print(f"=== Starting Run: {{EXPERIMENT_NAME}} ===")
trainer = KDSegmentationTrainer(cfg)
trainer.train()
print("Training completed.")
"""),
        make_cell("code", """# -- Step 3: Validate Best Checkpoint & Export Results --
import glob, json
from pathlib import Path
from ultralytics import YOLO

EXPERIMENT_NAME = "exp_mosaic_augmented_hires_layerkd_dilated_T3.7769_W0.9612_seed42_150ep"
best_pt = glob.glob(f"runs/**/{EXPERIMENT_NAME}*/weights/best.pt", recursive=True)
assert best_pt, f"No checkpoint found for {EXPERIMENT_NAME}!"

model = YOLO(best_pt[0])
print("\\n--- In-Domain Cropped Validation ---")
val_yaml = "data/datasets/crack500_yolo_augmented/dataset.yaml"
for cand in [
    "data/datasets_yaml/crack500_yolo_augmented/dataset.yaml",
    "data/datasets_yaml/crack500_yolo/dataset.yaml",
]:
    if Path(cand).exists():
        val_yaml = cand
        break

dset_root = Path("data/datasets/crack500_yolo_augmented").resolve()
if not (dset_root / "images/val").exists():
    dset_root = Path("data/datasets/crack500_yolo").resolve()

if (dset_root / "images/val").exists():
    fixed_val_yaml = Path("/kaggle/working/data/datasets_yaml/indomain_val.yaml")
    fixed_val_yaml.parent.mkdir(parents=True, exist_ok=True)
    fixed_val_yaml.write_text(f"path: {dset_root}\\ntrain: images/train\\nval: images/val\\ntest: images/test\\nnc: 1\\nnames:\\n  0: crack\\n")
    val_yaml = str(fixed_val_yaml)

val_metrics = model.val(data=val_yaml, split="val", verbose=True)

results = {
    "experiment": EXPERIMENT_NAME,
    "checkpoint": best_pt[0],
    "metrics_indomain": {
        "mask_mAP50": float(val_metrics.seg.map50),
        "mask_mAP50_95": float(val_metrics.seg.map),
        "box_mAP50": float(val_metrics.box.map50),
        "box_mAP50_95": float(val_metrics.box.map),
    },
}

uncropped_yaml = Path("data/datasets/crack500_uncropped_yolo/dataset.yaml")
if uncropped_yaml.exists():
    print("\\n--- OOD Uncropped Validation (direct resize) ---")
    ood_root = uncropped_yaml.parent.resolve()
    fixed_ood_yaml = Path("/kaggle/working/data/datasets_yaml/ood_val.yaml")
    fixed_ood_yaml.parent.mkdir(parents=True, exist_ok=True)
    fixed_ood_yaml.write_text(f"path: {ood_root}\\ntrain: images/val\\nval: images/val\\ntest: images/test\\nnc: 1\\nnames:\\n  0: crack\\n")
    ood_metrics = model.val(data=str(fixed_ood_yaml), split="val", verbose=True)
    results["metrics_ood"] = {
        "ood_mask_mAP50": float(ood_metrics.seg.map50),
        "ood_mask_mAP50_95": float(ood_metrics.seg.map),
    }

print("\\n" + "=" * 60)
print(f"RESULT ({EXPERIMENT_NAME}):")
print(f"  In-Domain Mask mAP50: {results['metrics_indomain']['mask_mAP50']:.4f}")
if "metrics_ood" in results:
    print(f"  OOD Direct Mask mAP50: {results['metrics_ood']['ood_mask_mAP50']:.4f}")
print("=" * 60)

out_file = Path(f"/kaggle/working/results/{EXPERIMENT_NAME}.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_file}. Attach this notebook's output to 04 for the tiled/Gaussian OOD eval.")
"""),
    ]
    return make_nb(cells)


def notebook_05():
    overrides_repr = """{
    'distillation.enabled': True,
    'distillation.temperature': 3.7769,
    'distillation.progressive.enabled': False,
    'distillation.losses.task.weight': 1.0,
    'distillation.losses.mask_kd.enabled': True,
    'distillation.losses.mask_kd.weight': 0.9612,
    'distillation.losses.mask_kd.focused': True,
    'distillation.losses.mask_kd.high_res': False,
    'distillation.losses.affinity.enabled': False,
    'distillation.losses.feature.enabled': True,
    'distillation.losses.feature.method': 'cwd',
    'distillation.losses.feature.weight': 0.25,
    'distillation.losses.feature.temperature': 4.0,
    'distillation.losses.feature.layers': [12, 15, 18],
    'distillation.losses.boundary.enabled': False,
    'student.imgsz': 640,
    'data.image_size': 640,
    'data.batch_size': 12,
    'train.lr': 0.001,
    'train.epochs': 50,
    'train.warmup_epochs': 3,
    'train.amp': False,
}"""
    cells = [
        make_cell("markdown", """# OODimprovements 05: Two-Stage Transfer Fine-Tuning (Crop Baseline -> Mosaic Native Set)
Implements the Two-Stage Transfer hypothesis from `next_moves_forOOD.md` §5 (Phase 1):
- **Stage 1 (Representation Pre-training)**: Detailed thin-crack feature representations learned on crops (e.g. `04_affinity` or `06_layerkd` or `03_mosaic_native`).
- **Stage 2 (Distribution & Context Adaptation)**: Fine-tunes the Stage 1 checkpoint on `crack500_yolo_augmented` (250 mosaic composites + crops) for **50 epochs** at native `imgsz=640` with a $10\\times$ reduced learning rate (`lr0: 0.001`, `warmup_epochs: 3`).
- **Checkpoints**: Automatically discovers and loads any attached `*.pt` checkpoint from Kaggle input (`+ Add Data -> Your Work -> Notebook Output Files`). If no checkpoint is attached, cleanly falls back to stock `yolo11n-seg.pt`.
- **Loss Recipe**: Neck CWD (layers 12, 15, 18) + 8px Foreground-Dilated Mask-KL ($\tau=3.7769, W=0.9612$).

**Requires**:
1. `distill_datasetforme` (or uncropped val/test data)
2. Notebook 01 output (`crack500_yolo_augmented`)
3. Notebook 02 output (`data/teacher_logits_box` with mosaic logits merged)
4. *(Optional but Recommended)*: Stage 1 checkpoint `best.pt` from `final_notebooks/04`, `06`, or `OODimprovements/03`."""),
        make_cell("code", """# -- Environment & Directory Initialization --
!mkdir -p configs utils distillation scripts checkpoints data/datasets data/teacher_logits_box runs results
!pip install -q ultralytics albumentations pycocotools thop pyyaml pandas tqdm opencv-python Pillow
"""),
        *make_training_setup_cells(),
        make_cell("code", LINK_OR_BUILD_INPUTS_CELL),
        make_cell("code", f"""# -- Step 2: Run Two-Stage Fine-Tuning (50 Epochs) --
import sys
sys.path.insert(0, ".")
from pathlib import Path
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
EXPERIMENT_NAME = "exp_twostage_mosaic_native_tune_seed42_50ep"

overrides = {overrides_repr}
overrides["project.name"] = "crack_distill"
overrides["project.experiment"] = EXPERIMENT_NAME
overrides["project.seed"] = 42
overrides["student.backbone"] = stage1_ckpt
overrides["data.datasets"] = [{{"name": "crack500_augmented", "path": "data/datasets/crack500_yolo_augmented", "format": "yolo"}}]
overrides["teacher.logits_dir"] = "data/teacher_logits_box/"

cfg = override_config(cfg, overrides)

print(f"=== Starting Two-Stage Fine-Tuning: {{EXPERIMENT_NAME}} (Starting from: {{stage1_ckpt}}) ===")
trainer = KDSegmentationTrainer(cfg)
trainer.train()
print("Training completed.")
"""),
        make_cell("code", """# -- Step 3: Validate Best Checkpoint & Export Results --
import glob, json
from pathlib import Path
from ultralytics import YOLO

EXPERIMENT_NAME = "exp_twostage_mosaic_native_tune_seed42_50ep"
best_pt = glob.glob(f"runs/**/{EXPERIMENT_NAME}*/weights/best.pt", recursive=True)
assert best_pt, f"No checkpoint found for {EXPERIMENT_NAME}!"

model = YOLO(best_pt[0])
print("\\n--- In-Domain Cropped Validation ---")
val_yaml = "data/datasets/crack500_yolo_augmented/dataset.yaml"
for cand in [
    "data/datasets_yaml/crack500_yolo_augmented/dataset.yaml",
    "data/datasets_yaml/crack500_yolo/dataset.yaml",
]:
    if Path(cand).exists():
        val_yaml = cand
        break

dset_root = Path("data/datasets/crack500_yolo_augmented").resolve()
if not (dset_root / "images/val").exists():
    dset_root = Path("data/datasets/crack500_yolo").resolve()

if (dset_root / "images/val").exists():
    fixed_val_yaml = Path("/kaggle/working/data/datasets_yaml/indomain_val.yaml")
    fixed_val_yaml.parent.mkdir(parents=True, exist_ok=True)
    fixed_val_yaml.write_text(f"path: {dset_root}\\ntrain: images/train\\nval: images/val\\ntest: images/test\\nnc: 1\\nnames:\\n  0: crack\\n")
    val_yaml = str(fixed_val_yaml)

val_metrics = model.val(data=val_yaml, split="val", verbose=True)

results = {
    "experiment": EXPERIMENT_NAME,
    "initial_checkpoint": stage1_ckpt,
    "checkpoint": best_pt[0],
    "metrics_indomain": {
        "mask_mAP50": float(val_metrics.seg.map50),
        "mask_mAP50_95": float(val_metrics.seg.map),
        "box_mAP50": float(val_metrics.box.map50),
        "box_mAP50_95": float(val_metrics.box.map),
    },
}

uncropped_yaml = Path("data/datasets/crack500_uncropped_yolo/dataset.yaml")
if uncropped_yaml.exists():
    print("\\n--- OOD Uncropped Validation (direct resize) ---")
    ood_root = uncropped_yaml.parent.resolve()
    fixed_ood_yaml = Path("/kaggle/working/data/datasets_yaml/ood_val.yaml")
    fixed_ood_yaml.parent.mkdir(parents=True, exist_ok=True)
    fixed_ood_yaml.write_text(f"path: {ood_root}\\ntrain: images/val\\nval: images/val\\ntest: images/test\\nnc: 1\\nnames:\\n  0: crack\\n")
    ood_metrics = model.val(data=str(fixed_ood_yaml), split="val", verbose=True)
    results["metrics_ood"] = {
        "ood_mask_mAP50": float(ood_metrics.seg.map50),
        "ood_mask_mAP50_95": float(ood_metrics.seg.map),
    }

print("\\n" + "=" * 60)
print(f"RESULT ({EXPERIMENT_NAME}):")
print(f"  In-Domain Mask mAP50: {results['metrics_indomain']['mask_mAP50']:.4f}")
if "metrics_ood" in results:
    print(f"  OOD Direct Mask mAP50: {results['metrics_ood']['ood_mask_mAP50']:.4f}")
print("=" * 60)

out_file = Path(f"/kaggle/working/results/{EXPERIMENT_NAME}.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_file}. Attach this notebook's output to 04 or 08 for the tiled/Gaussian OOD eval.")
"""),
    ]
    return make_nb(cells)


def notebook_06():
    overrides_repr = """{
    'distillation.enabled': True,
    'distillation.temperature': 3.7769,
    'distillation.progressive.enabled': False,
    'distillation.losses.task.weight': 1.0,
    'distillation.losses.mask_kd.enabled': True,
    'distillation.losses.mask_kd.weight': 0.9612,
    'distillation.losses.mask_kd.focused': True,
    'distillation.losses.mask_kd.native_res': True,
    'distillation.losses.mask_kd.high_res': False,
    'distillation.losses.affinity.enabled': False,
    'distillation.losses.feature.enabled': True,
    'distillation.losses.feature.method': 'cwd',
    'distillation.losses.feature.weight': 0.25,
    'distillation.losses.feature.temperature': 4.0,
    'distillation.losses.feature.layers': [12, 15, 18],
    'distillation.losses.boundary.enabled': False,
    'student.imgsz': 640,
    'data.image_size': 640,
    'data.batch_size': 12,
    'train.lr': 0.001,
    'train.epochs': 150,
    'train.amp': False,
}"""
    cells = [
        make_cell("markdown", """# OODimprovements 06: Resolution-Preserving Mask Distillation (Upsampled Student Proto-Mask)
Solves the sub-pixel thin crack dilution defect detailed in `next_moves_forOOD.md` §4.1:
- **The Defect**: Standard YOLOv11-seg mask prototypes are coarse ($160 \times 160$ for $640 \times 640$ inputs). Downsampling SAM 2's native teacher logits to $160 \times 160$ washes out 1-pixel cracks ($15$ background pixels overwhelm $1$ positive pixel during pooling).
- **The Solution**: **Resolution-Preserving Mask-KL (`native_res: True`)**—bilinearly upsamples student instance mask logits to the teacher's native resolution ($640 \times 360$ / $256 \times 256$) before Temperature-scaled Bernoulli KL divergence ($\tau=3.7769$). Gradients propagate backwards through the bilinear interpolation operator directly into the 32 prototype maps and coefficients.
- **Loss Recipe**: Native-Res Foreground-Dilated Mask-KL ($\tau=3.7769, W=0.9612$) + Neck CWD (layers 12, 15, 18, $W=0.25$).
- **Dataset**: Trains on `crack500_yolo_augmented` at native `imgsz=640` for 150 epochs.

**Requires**:
1. `distill_datasetforme`
2. Notebook 01 output (`crack500_yolo_augmented`)
3. Notebook 02 output (`data/teacher_logits_box` with mosaic logits merged)"""),
        make_cell("code", """# -- Environment & Directory Initialization --
!mkdir -p configs utils distillation scripts checkpoints data/datasets data/teacher_logits_box runs results
!pip install -q ultralytics albumentations pycocotools thop pyyaml pandas tqdm opencv-python Pillow
"""),
        *make_training_setup_cells(),
        make_cell("code", LINK_OR_BUILD_INPUTS_CELL),
        make_cell("code", f"""# -- Step 2: Run Training (exp_resolution_preserving_mask_kd_T3.7769_W0.9612_seed42_150ep) --
import sys
sys.path.insert(0, ".")
from pathlib import Path
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
EXPERIMENT_NAME = "exp_resolution_preserving_mask_kd_T3.7769_W0.9612_seed42_150ep"

overrides = {overrides_repr}
overrides["project.name"] = "crack_distill"
overrides["project.experiment"] = EXPERIMENT_NAME
overrides["project.seed"] = 42
overrides["data.datasets"] = [{{"name": "crack500_augmented", "path": "data/datasets/crack500_yolo_augmented", "format": "yolo"}}]
overrides["teacher.logits_dir"] = "data/teacher_logits_box/"

cfg = override_config(cfg, overrides)

print(f"=== Starting Run: {{EXPERIMENT_NAME}} ===")
trainer = KDSegmentationTrainer(cfg)
trainer.train()
print("Training completed.")
"""),
        make_cell("code", """# -- Step 3: Validate Best Checkpoint & Export Results --
import glob, json
from pathlib import Path
from ultralytics import YOLO

EXPERIMENT_NAME = "exp_resolution_preserving_mask_kd_T3.7769_W0.9612_seed42_150ep"
best_pt = glob.glob(f"runs/**/{EXPERIMENT_NAME}*/weights/best.pt", recursive=True)
assert best_pt, f"No checkpoint found for {EXPERIMENT_NAME}!"

model = YOLO(best_pt[0])
print("\\n--- In-Domain Cropped Validation ---")
val_yaml = "data/datasets/crack500_yolo_augmented/dataset.yaml"
for cand in [
    "data/datasets_yaml/crack500_yolo_augmented/dataset.yaml",
    "data/datasets_yaml/crack500_yolo/dataset.yaml",
]:
    if Path(cand).exists():
        val_yaml = cand
        break

dset_root = Path("data/datasets/crack500_yolo_augmented").resolve()
if not (dset_root / "images/val").exists():
    dset_root = Path("data/datasets/crack500_yolo").resolve()

if (dset_root / "images/val").exists():
    fixed_val_yaml = Path("/kaggle/working/data/datasets_yaml/indomain_val.yaml")
    fixed_val_yaml.parent.mkdir(parents=True, exist_ok=True)
    fixed_val_yaml.write_text(f"path: {dset_root}\\ntrain: images/train\\nval: images/val\\ntest: images/test\\nnc: 1\\nnames:\\n  0: crack\\n")
    val_yaml = str(fixed_val_yaml)

val_metrics = model.val(data=val_yaml, split="val", verbose=True)

results = {
    "experiment": EXPERIMENT_NAME,
    "checkpoint": best_pt[0],
    "metrics_indomain": {
        "mask_mAP50": float(val_metrics.seg.map50),
        "mask_mAP50_95": float(val_metrics.seg.map),
        "box_mAP50": float(val_metrics.box.map50),
        "box_mAP50_95": float(val_metrics.box.map),
    },
}

uncropped_yaml = Path("data/datasets/crack500_uncropped_yolo/dataset.yaml")
if uncropped_yaml.exists():
    print("\\n--- OOD Uncropped Validation (direct resize) ---")
    ood_root = uncropped_yaml.parent.resolve()
    fixed_ood_yaml = Path("/kaggle/working/data/datasets_yaml/ood_val.yaml")
    fixed_ood_yaml.parent.mkdir(parents=True, exist_ok=True)
    fixed_ood_yaml.write_text(f"path: {ood_root}\\ntrain: images/val\\nval: images/val\\ntest: images/test\\nnc: 1\\nnames:\\n  0: crack\\n")
    ood_metrics = model.val(data=str(fixed_ood_yaml), split="val", verbose=True)
    results["metrics_ood"] = {
        "ood_mask_mAP50": float(ood_metrics.seg.map50),
        "ood_mask_mAP50_95": float(ood_metrics.seg.map),
    }

print("\\n" + "=" * 60)
print(f"RESULT ({EXPERIMENT_NAME}):")
print(f"  In-Domain Mask mAP50: {results['metrics_indomain']['mask_mAP50']:.4f}")
if "metrics_ood" in results:
    print(f"  OOD Direct Mask mAP50: {results['metrics_ood']['ood_mask_mAP50']:.4f}")
print("=" * 60)

out_file = Path(f"/kaggle/working/results/{EXPERIMENT_NAME}.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_file}. Attach this notebook's output to 04 or 08 for the tiled/Gaussian OOD eval.")
"""),
    ]
    return make_nb(cells)


def notebook_07():
    overrides_repr = """{
    'distillation.enabled': True,
    'distillation.temperature': 3.7769,
    'distillation.progressive.enabled': False,
    'distillation.losses.task.weight': 1.0,
    'distillation.losses.mask_kd.enabled': True,
    'distillation.losses.mask_kd.weight': 0.9612,
    'distillation.losses.mask_kd.focused': True,
    'distillation.losses.mask_kd.high_res': False,
    'distillation.losses.tversky.enabled': True,
    'distillation.losses.tversky.alpha': 0.30,
    'distillation.losses.tversky.beta': 0.70,
    'distillation.losses.tversky.weight': 1.0,
    'distillation.losses.affinity.enabled': False,
    'distillation.losses.feature.enabled': True,
    'distillation.losses.feature.method': 'cwd',
    'distillation.losses.feature.weight': 0.25,
    'distillation.losses.feature.temperature': 4.0,
    'distillation.losses.feature.layers': [12, 15, 18],
    'distillation.losses.boundary.enabled': False,
    'student.imgsz': 640,
    'data.image_size': 640,
    'data.batch_size': 12,
    'train.lr': 0.001,
    'train.epochs': 150,
    'train.amp': False,
}"""
    cells = [
        make_cell("markdown", r"""# OODimprovements 07: Asymmetric Soft Tversky & Structural Continuity KD
Implements the Asymmetric Soft Tversky Loss formulated in `next_moves_forOOD.md` §4.2:
- **The Problem**: Standard BCE / Dice symmetrically weights false positives and false negatives. Because road cracks occupy $<1\%$ of pixels, missing thin branches (FN) incurs negligible loss penalty, leading to broken and fragmented crack predictions.
- **The Solution**: **Asymmetric Tversky Loss** with $\alpha=0.30$ (False Positive weight) and $\beta=0.70$ (False Negative weight). This heavily penalizes missed thin-crack pixels while preserving pavement background precision.
- **Loss Recipe**: Task Loss + Asymmetric Tversky ($\alpha=0.30, \beta=0.70, W=1.0$) + Foreground-Dilated Mask-KL ($\tau=3.7769, W=0.9612$) + Neck CWD (layers 12, 15, 18, $W=0.25$).
- **Dataset**: Trains on `crack500_yolo_augmented` at native `imgsz=640` for 150 epochs.

**Requires**:
1. `distill_datasetforme`
2. Notebook 01 output (`crack500_yolo_augmented`)
3. Notebook 02 output (`data/teacher_logits_box` with mosaic logits merged)"""),
        make_cell("code", """# -- Environment & Directory Initialization --
!mkdir -p configs utils distillation scripts checkpoints data/datasets data/teacher_logits_box runs results
!pip install -q ultralytics albumentations pycocotools thop pyyaml pandas tqdm opencv-python Pillow
"""),
        *make_training_setup_cells(),
        make_cell("code", LINK_OR_BUILD_INPUTS_CELL),
        make_cell("code", f"""# -- Step 2: Run Training (exp_asymmetric_tversky_kd_T3.7769_W0.9612_seed42_150ep) --
import sys
sys.path.insert(0, ".")
from pathlib import Path
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

cfg = load_config("configs/config.yaml")
EXPERIMENT_NAME = "exp_asymmetric_tversky_kd_T3.7769_W0.9612_seed42_150ep"

overrides = {overrides_repr}
overrides["project.name"] = "crack_distill"
overrides["project.experiment"] = EXPERIMENT_NAME
overrides["project.seed"] = 42
overrides["data.datasets"] = [{{"name": "crack500_augmented", "path": "data/datasets/crack500_yolo_augmented", "format": "yolo"}}]
overrides["teacher.logits_dir"] = "data/teacher_logits_box/"

cfg = override_config(cfg, overrides)

print(f"=== Starting Run: {{EXPERIMENT_NAME}} ===")
trainer = KDSegmentationTrainer(cfg)
trainer.train()
print("Training completed.")
"""),
        make_cell("code", """# -- Step 3: Validate Best Checkpoint & Export Results --
import glob, json
from pathlib import Path
from ultralytics import YOLO

EXPERIMENT_NAME = "exp_asymmetric_tversky_kd_T3.7769_W0.9612_seed42_150ep"
best_pt = glob.glob(f"runs/**/{EXPERIMENT_NAME}*/weights/best.pt", recursive=True)
assert best_pt, f"No checkpoint found for {EXPERIMENT_NAME}!"

model = YOLO(best_pt[0])
print("\\n--- In-Domain Cropped Validation ---")
val_yaml = "data/datasets/crack500_yolo_augmented/dataset.yaml"
for cand in [
    "data/datasets_yaml/crack500_yolo_augmented/dataset.yaml",
    "data/datasets_yaml/crack500_yolo/dataset.yaml",
]:
    if Path(cand).exists():
        val_yaml = cand
        break

dset_root = Path("data/datasets/crack500_yolo_augmented").resolve()
if not (dset_root / "images/val").exists():
    dset_root = Path("data/datasets/crack500_yolo").resolve()

if (dset_root / "images/val").exists():
    fixed_val_yaml = Path("/kaggle/working/data/datasets_yaml/indomain_val.yaml")
    fixed_val_yaml.parent.mkdir(parents=True, exist_ok=True)
    fixed_val_yaml.write_text(f"path: {dset_root}\\ntrain: images/train\\nval: images/val\\ntest: images/test\\nnc: 1\\nnames:\\n  0: crack\\n")
    val_yaml = str(fixed_val_yaml)

val_metrics = model.val(data=val_yaml, split="val", verbose=True)

results = {
    "experiment": EXPERIMENT_NAME,
    "checkpoint": best_pt[0],
    "metrics_indomain": {
        "mask_mAP50": float(val_metrics.seg.map50),
        "mask_mAP50_95": float(val_metrics.seg.map),
        "box_mAP50": float(val_metrics.box.map50),
        "box_mAP50_95": float(val_metrics.box.map),
    },
}

uncropped_yaml = Path("data/datasets/crack500_uncropped_yolo/dataset.yaml")
if uncropped_yaml.exists():
    print("\\n--- OOD Uncropped Validation (direct resize) ---")
    ood_root = uncropped_yaml.parent.resolve()
    fixed_ood_yaml = Path("/kaggle/working/data/datasets_yaml/ood_val.yaml")
    fixed_ood_yaml.parent.mkdir(parents=True, exist_ok=True)
    fixed_ood_yaml.write_text(f"path: {ood_root}\\ntrain: images/val\\nval: images/val\\ntest: images/test\\nnc: 1\\nnames:\\n  0: crack\\n")
    ood_metrics = model.val(data=str(fixed_ood_yaml), split="val", verbose=True)
    results["metrics_ood"] = {
        "ood_mask_mAP50": float(ood_metrics.seg.map50),
        "ood_mask_mAP50_95": float(ood_metrics.seg.map),
    }

print("\\n" + "=" * 60)
print(f"RESULT ({EXPERIMENT_NAME}):")
print(f"  In-Domain Mask mAP50: {results['metrics_indomain']['mask_mAP50']:.4f}")
if "metrics_ood" in results:
    print(f"  OOD Direct Mask mAP50: {results['metrics_ood']['ood_mask_mAP50']:.4f}")
print("=" * 60)

out_file = Path(f"/kaggle/working/results/{EXPERIMENT_NAME}.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_file}. Attach this notebook's output to 04 or 08 for the tiled/Gaussian OOD eval.")
"""),
    ]
    return make_nb(cells)


def notebook_08():
    cells = [
        make_cell("markdown", """# OODimprovements 08: Multi-Scale Gaussian TTA & Sliced Inference Engine
Test-Time Deployment Engine implementing Phase 0 from `next_moves_forOOD.md`:
- **Multi-Scale Gaussian TTA**: Evaluates uncropped megapixel photos ($2000 \\times 1500$) across multi-scale patches ($640 \\times 640$ and $768 \\times 768$, $25\\%$ overlap) with **2D Gaussian Apodization Blending**.
- **Head-to-Head Comparison Across 3 Paradigms**:
  1. Direct Resizing ($2000 \\times 1500 \\to 640 \\times 640$)
  2. Single-Scale Gaussian Tiling ($640 \\times 640$ patches, 25% overlap, Gaussian apodization)
  3. Multi-Scale Gaussian TTA ($640 \\times 640$ + $768 \\times 768$ fused Gaussian apodization)
- **Checkpoints**: Automatically discovers and benchmarks all available checkpoints in `/kaggle/input` (e.g. `03_mosaic_native`, `05_twostage`, `06_res_preserving`, `07_tversky`, or prior models).
- **Outputs**: Comprehensive metrics table (Dice, Precision, Recall, IoU) and JSON export."""),
        make_cell("code", """# -- Environment & Imports --
!mkdir -p scripts configs utils distillation data/datasets results repacked_ckpts
!pip install -q ultralytics albumentations pycocotools opencv-python Pillow matplotlib tqdm pandas
import os, cv2, json, time, glob, zipfile, shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import torch
"""),
        make_cell("code", f"%%writefile scripts/convert_crack500_uncropped.py\n{convert_crack500_uncropped_code}"),
        make_cell("code", """# -- Step 1: Link or Convert Uncropped Validation Dataset --
input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists():
    input_dir = Path("/kaggle/input")

for root, dirs, files in os.walk(str(input_dir)):
    root_p = Path(root)
    if "valdata" in dirs or "testdata" in dirs:
        !python scripts/convert_crack500_uncropped.py --src {root_p} --dst data/datasets/crack500_uncropped_yolo
        break

print("Uncropped validation dataset ready at data/datasets/crack500_uncropped_yolo/")
"""),
        make_cell("code", """# -- Step 2: Multi-Scale Gaussian TTA Inference Engine --
def create_gaussian_weight_map(tile_size=640, sigma=0.35):
    \"\"\"Generates a 2D Gaussian window to smoothly blend overlapping tiles.\"\"\"
    ax = np.linspace(-1, 1, tile_size)
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    gauss_2d = np.outer(gauss_1d, gauss_1d).astype(np.float32)
    gauss_2d = np.maximum(gauss_2d, 0.05)  # baseline floor for borders
    return gauss_2d / gauss_2d.max()


def predict_sliding_window_gaussian(model, img_bgr, tile_size=640, overlap=0.25, conf=0.25, sigma=0.35):
    \"\"\"Single-scale tiled inference with 2D Gaussian apodization blending.\"\"\"
    h, w = img_bgr.shape[:2]
    stride = int(tile_size * (1.0 - overlap))
    full_prob_map = np.zeros((h, w), dtype=np.float32)
    weight_accum_map = np.zeros((h, w), dtype=np.float32)
    weight_window = create_gaussian_weight_map(tile_size, sigma)

    y_steps = list(range(0, max(1, h - tile_size + 1), stride))
    if len(y_steps) == 0 or (y_steps[-1] + tile_size < h):
        y_steps.append(max(0, h - tile_size))
        
    x_steps = list(range(0, max(1, w - tile_size + 1), stride))
    if len(x_steps) == 0 or (x_steps[-1] + tile_size < w):
        x_steps.append(max(0, w - tile_size))

    for y0 in y_steps:
        for x0 in x_steps:
            tile = img_bgr[y0:y0+tile_size, x0:x0+tile_size]
            actual_th, actual_tw = tile.shape[:2]
            if actual_th != tile_size or actual_tw != tile_size:
                tile_padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                tile_padded[:actual_th, :actual_tw] = tile
                tile = tile_padded

            results = model.predict(tile, imgsz=tile_size, conf=conf, verbose=False, device="cuda" if torch.cuda.is_available() else "cpu")[0]
            tile_prob = np.zeros((tile_size, tile_size), dtype=np.float32)

            if results.masks is not None and len(results.masks) > 0:
                for m in results.masks.data.cpu().numpy():
                    m_resized = cv2.resize(m, (tile_size, tile_size))
                    tile_prob = np.maximum(tile_prob, m_resized)

            actual_h = min(tile_size, h - y0)
            actual_w = min(tile_size, w - x0)

            full_prob_map[y0:y0+actual_h, x0:x0+actual_w] += tile_prob[:actual_h, :actual_w] * weight_window[:actual_h, :actual_w]
            weight_accum_map[y0:y0+actual_h, x0:x0+actual_w] += weight_window[:actual_h, :actual_w]

    weight_accum_map = np.maximum(weight_accum_map, 1e-5)
    full_prob_map = full_prob_map / weight_accum_map
    return (full_prob_map > 0.35).astype(np.uint8)


def predict_multiscale_gaussian_tta(model, img_bgr, tile_sizes=(640, 768), overlap=0.25, conf=0.25, sigma=0.35):
    \"\"\"Multi-scale test-time augmentation across tile sizes (e.g. 640 + 768).\"\"\"
    h, w = img_bgr.shape[:2]
    fused_prob = np.zeros((h, w), dtype=np.float32)

    for ts in tile_sizes:
        prob_map = np.zeros((h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)
        weight_window = create_gaussian_weight_map(ts, sigma)
        stride = int(ts * (1.0 - overlap))

        y_steps = list(range(0, max(1, h - ts + 1), stride))
        if len(y_steps) == 0 or (y_steps[-1] + ts < h):
            y_steps.append(max(0, h - ts))
        x_steps = list(range(0, max(1, w - ts + 1), stride))
        if len(x_steps) == 0 or (x_steps[-1] + ts < w):
            x_steps.append(max(0, w - ts))

        for y0 in y_steps:
            for x0 in x_steps:
                tile = img_bgr[y0:y0+ts, x0:x0+ts]
                actual_th, actual_tw = tile.shape[:2]
                if actual_th != ts or actual_tw != ts:
                    tile_padded = np.zeros((ts, ts, 3), dtype=np.uint8)
                    tile_padded[:actual_th, :actual_tw] = tile
                    tile = tile_padded

                results = model.predict(tile, imgsz=ts, conf=conf, verbose=False, device="cuda" if torch.cuda.is_available() else "cpu")[0]
                tile_prob = np.zeros((ts, ts), dtype=np.float32)

                if results.masks is not None and len(results.masks) > 0:
                    for m in results.masks.data.cpu().numpy():
                        m_resized = cv2.resize(m, (ts, ts))
                        tile_prob = np.maximum(tile_prob, m_resized)

                actual_h = min(ts, h - y0)
                actual_w = min(ts, w - x0)
                prob_map[y0:y0+actual_h, x0:x0+actual_w] += tile_prob[:actual_h, :actual_w] * weight_window[:actual_h, :actual_w]
                weight_accum[y0:y0+actual_h, x0:x0+actual_w] += weight_window[:actual_h, :actual_w]

        prob_map = prob_map / np.maximum(weight_accum, 1e-5)
        fused_prob += prob_map

    fused_prob = fused_prob / len(tile_sizes)
    return (fused_prob > 0.35).astype(np.uint8)

print("Multi-Scale Gaussian TTA Inference Engine ready!")
"""),
        make_cell("code", """# -- Step 3: Discover Checkpoints & Run Cross-Evaluation --
from PIL import Image

def get_exif_rotation(img_path: Path):
    \"\"\"Retrieve EXIF orientation tag from image.\"\"\"
    try:
        with Image.open(img_path) as im:
            exif = im.getexif()
            if exif:
                return exif.get(274)  # 274 is the Orientation tag
    except Exception:
        pass
    return None

def rotate_mask_to_match_image(mask: np.ndarray, exif_orientation: int) -> np.ndarray:
    \"\"\"Rotate mask array to match image rotation applied by cv2.imread based on EXIF.\"\"\"
    if exif_orientation == 6:
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    elif exif_orientation == 8:
        return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif exif_orientation == 3:
        return cv2.rotate(mask, cv2.ROTATE_180)
    return mask

def compute_metrics(pred_mask, gt_mask):
    # Defensive shape safeguard: guarantee identical broadcast dimensions under any condition
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask.astype(np.uint8), (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
    total = (pred_mask > 0).sum() + (gt_mask > 0).sum()
    pred_sum = (pred_mask > 0).sum()
    gt_sum = (gt_mask > 0).sum()

    dice = float(2.0 * intersection / total) if total > 0 else (1.0 if intersection == 0 else 0.0)
    precision = float(intersection / pred_sum) if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
    recall = float(intersection / gt_sum) if gt_sum > 0 else (1.0 if pred_sum == 0 else 0.0)
    union = pred_sum + gt_sum - intersection
    iou = float(intersection / union) if union > 0 else 1.0

    return {"dice": dice, "precision": precision, "recall": recall, "iou": iou}

repack_dir = Path("/kaggle/working/repacked_ckpts")
repack_dir.mkdir(parents=True, exist_ok=True)

# A. Handle unzipped PyTorch model folders (contains data.pkl like bestmosaic/best)
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "data.pkl" in files:
            parts = [p for p in Path(root).parts if p not in [".", ""]]
            folder_name = Path(root).name
            if folder_name in ["best", "weights", "model", "archive", "default", "1", ".", ""]:
                skip_set = {"kaggle", "input", "models", "pytorch", "default", "1", "best", "weights", "model", "archive"}
                meaningful = [p for p in parts if p not in skip_set]
                folder_name = meaningful[-1] if meaningful else Path(root).parent.name
            repacked_path = repack_dir / f"{folder_name}.pt"
            print(f"  -> Detected unzipped PyTorch folder: {root}")
            print(f"  -> Repacking into valid PyTorch container: {repacked_path}...")
            with zipfile.ZipFile(repacked_path, 'w', compression=zipfile.ZIP_STORED) as zf:
                for r, d, f_list in os.walk(root):
                    for file_name in f_list:
                        full_file = os.path.join(r, file_name)
                        rel_file = os.path.relpath(full_file, root)
                        arc_name = os.path.join("archive", rel_file).replace("\\\\", "/")
                        zf.write(full_file, arc_name)
            print(f"  -> Successfully created valid model: {repacked_path}")

# B. Scan checkpoints across repacked, input, working, and runs directories
checkpoints = {}
search_roots = ["/kaggle/working/repacked_ckpts", "/kaggle/input", "/kaggle/working", "runs"]
stock_weights = {"yolo11n-seg.pt", "yolov8n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt"}

for s_root in search_roots:
    if os.path.exists(s_root):
        for root, dirs, files in os.walk(s_root):
            for f in files:
                if (f.endswith(".pt") or f.endswith(".pth")) and "sam" not in f.lower() and "hiera" not in f.lower():
                    if f in stock_weights and "runs" not in root:
                        continue
                    resolved = str(Path(os.path.join(root, f)).resolve())
                    p = Path(resolved)
                    if p.parent.name == "weights":
                        exp_name = p.parent.parent.name
                        tag = f"{exp_name}_{p.stem}"
                    elif p.parent.name == "repacked_ckpts":
                        tag = p.stem
                    else:
                        parent_name = p.parent.name
                        tag = f"{parent_name}_{p.stem}" if parent_name not in ["working", "input"] else p.stem
                    
                    base_tag = tag
                    c_idx = 1
                    while tag in checkpoints and checkpoints[tag] != resolved:
                        tag = f"{base_tag}_{c_idx}"
                        c_idx += 1

                    if tag not in checkpoints:
                        checkpoints[tag] = resolved

if not checkpoints:
    print("[Warning] No finetuned checkpoints found; downloading stock yolo11n-seg.pt as fallback")
    checkpoints["yolo11n-seg_baseline"] = "yolo11n-seg.pt"

print(f"Discovered {len(checkpoints)} checkpoints to evaluate:")
for k, v in checkpoints.items():
    print(f"  * {k}: {v}")

# C. Pre-index ground-truth masks across /kaggle/input, /kaggle/working, and data/
print("\\n[Step 3] Indexing ground-truth mask database...")
mask_database = {}
for search_dir in ["/kaggle/input", "/kaggle/working", "data"]:
    if os.path.exists(search_dir):
        for root, dirs, files in os.walk(search_dir):
            if "teacher_logits" in root:
                continue
            for f in files:
                if f.endswith("_mask.png") or (f.endswith(".png") and not f.endswith("_prob.png")):
                    clean_stem = f.replace("_mask.png", "").replace(".png", "")
                    mask_database[clean_stem] = os.path.join(root, f)

print(f"  -> Indexed {len(mask_database)} ground-truth mask files.")

# D. Load test pairs
val_dir = Path("data/datasets/crack500_uncropped_yolo")
img_files = sorted(list((val_dir / "images/val").glob("*.jpg")) + list((val_dir / "images/val").glob("*.png")))
if not img_files:
    img_files = sorted(list((val_dir / "images/test").glob("*.jpg")) + list((val_dir / "images/test").glob("*.png")))

test_pairs = []
for img_p in img_files:
    stem = img_p.stem
    if stem in mask_database:
        test_pairs.append((img_p, Path(mask_database[stem])))

print(f"Prepared {len(test_pairs)} uncropped evaluation photo-mask pairs.")

results = {}
for ckpt_name, ckpt_path in checkpoints.items():
    print(f"\\nEvaluating checkpoint: {ckpt_name}...")
    try:
        model = YOLO(ckpt_path)
    except Exception as e:
        print(f"  Failed to load {ckpt_name}: {e}")
        continue

    dices_direct = []
    dices_tiled = []
    dices_tta = []
    precisions_tta = []
    recalls_tta = []

    for img_p, mask_p in tqdm(test_pairs[:50], desc=f"Eval {ckpt_name}"):  # evaluate first 50 for speed
        img = cv2.imread(str(img_p))
        gt = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        h, w = img.shape[:2]

        # 1. Align EXIF orientation between auto-oriented JPEG and raw PNG mask
        exif_rot = get_exif_rotation(img_p)
        if exif_rot:
            gt = rotate_mask_to_match_image(gt, exif_rot)

        # 2. Geometric transpose check (e.g. 1440x2560 vs 2560x1440)
        if gt.shape[:2] == (w, h):
            gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)

        # 3. Final dimension safety resize
        if gt.shape[:2] != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

        gt_bin = (gt > 127).astype(np.uint8)

        # 1. Direct Resize (640)
        res_dir = model.predict(img, imgsz=640, conf=0.25, verbose=False, device="cuda" if torch.cuda.is_available() else "cpu")[0]
        pred_direct = np.zeros((h, w), dtype=np.uint8)
        if res_dir.masks is not None and len(res_dir.masks) > 0:
            pred_dir_mask = (res_dir.masks.data.cpu().numpy().max(axis=0) > 0.5).astype(np.uint8)
            pred_direct = cv2.resize(pred_dir_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        dices_direct.append(compute_metrics(pred_direct, gt_bin)["dice"])

        # 2. Gaussian Tiled (640)
        pred_tiled = predict_sliding_window_gaussian(model, img, tile_size=640, overlap=0.25, conf=0.25)
        dices_tiled.append(compute_metrics(pred_tiled, gt_bin)["dice"])

        # 3. Multi-Scale Gaussian TTA (640 + 768)
        pred_tta = predict_multiscale_gaussian_tta(model, img, tile_sizes=(640, 768), overlap=0.25, conf=0.25)
        m_tta = compute_metrics(pred_tta, gt_bin)
        dices_tta.append(m_tta["dice"])
        precisions_tta.append(m_tta["precision"])
        recalls_tta.append(m_tta["recall"])

    results[ckpt_name] = {
        "mean_dice_direct": float(np.mean(dices_direct)) if dices_direct else 0.0,
        "mean_dice_tiled_640": float(np.mean(dices_tiled)) if dices_tiled else 0.0,
        "mean_dice_multiscale_tta": float(np.mean(dices_tta)) if dices_tta else 0.0,
        "mean_precision_tta": float(np.mean(precisions_tta)) if precisions_tta else 0.0,
        "mean_recall_tta": float(np.mean(recalls_tta)) if recalls_tta else 0.0,
    }
    print(f"  -> Direct Dice: {results[ckpt_name]['mean_dice_direct']:.4f}")
    print(f"  -> Tiled 640 Dice: {results[ckpt_name]['mean_dice_tiled_640']:.4f}")
    print(f"  -> Multi-Scale TTA Dice: {results[ckpt_name]['mean_dice_multiscale_tta']:.4f}")
    print(f"  -> TTA Precision: {results[ckpt_name]['mean_precision_tta']:.4f} | Recall: {results[ckpt_name]['mean_recall_tta']:.4f}")

# Save results JSON
out_path = Path("/kaggle/working/results/multiscale_tta_sahi_eval_summary.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\\nEvaluation summary saved to {out_path}!")
"""),
    ]
    return make_nb(cells)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    with open(OUT_DIR / "01_mine_mosaics_and_negatives.ipynb", "w") as f:
        json.dump(notebook_01(), f, indent=1, ensure_ascii=False)
    print("Created OODimprovements/01_mine_mosaics_and_negatives.ipynb")

    with open(OUT_DIR / "02_generate_native_teacher_logits.ipynb", "w") as f:
        json.dump(notebook_02(), f, indent=1, ensure_ascii=False)
    print("Created OODimprovements/02_generate_native_teacher_logits.ipynb")

    with open(OUT_DIR / "03_run_mosaic_native_kd.ipynb", "w") as f:
        json.dump(notebook_03(), f, indent=1, ensure_ascii=False)
    print("Created OODimprovements/03_run_mosaic_native_kd.ipynb")

    src_07 = ROOT / "final_notebooks/07_eval_ood_and_tiled_inference.ipynb"
    dst_04 = OUT_DIR / "04_eval_ood_and_tiled_inference.ipynb"
    shutil.copy(src_07, dst_04)
    print("Copied final_notebooks/07 -> OODimprovements/04_eval_ood_and_tiled_inference.ipynb")

    with open(OUT_DIR / "05_run_twostage_mosaic_native_tune.ipynb", "w") as f:
        json.dump(notebook_05(), f, indent=1, ensure_ascii=False)
    print("Created OODimprovements/05_run_twostage_mosaic_native_tune.ipynb")

    with open(OUT_DIR / "06_run_resolution_preserving_kd.ipynb", "w") as f:
        json.dump(notebook_06(), f, indent=1, ensure_ascii=False)
    print("Created OODimprovements/06_run_resolution_preserving_kd.ipynb")

    with open(OUT_DIR / "07_run_asymmetric_tversky_kd.ipynb", "w") as f:
        json.dump(notebook_07(), f, indent=1, ensure_ascii=False)
    print("Created OODimprovements/07_run_asymmetric_tversky_kd.ipynb")

    with open(OUT_DIR / "08_eval_multiscale_tta_sahi.ipynb", "w") as f:
        json.dump(notebook_08(), f, indent=1, ensure_ascii=False)
    print("Created OODimprovements/08_eval_multiscale_tta_sahi.ipynb")


if __name__ == "__main__":
    main()
