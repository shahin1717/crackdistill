#!/usr/bin/env python3
"""
Builds the self-contained Kaggle notebook suite in OODimprovements/, implementing
the plan in possibleOODimprovements.md:

  01 -> mine mosaic composites (+ pilot negative tiles) from the existing traincrop grid
  02 -> generate native-resolution SAM2 teacher logits on those mosaics
  03 -> train the CWD+dilated recipe (from final_notebooks/10) on the mosaic-augmented
        training set with native-scale teacher supervision
  04 -> Gaussian-tiled OOD evaluation (copy of final_notebooks/07, includes the new checkpoint)

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
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
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
        make_cell("code", """# -- Link notebook 01 output + original teacher logits --
import os, shutil
from pathlib import Path

# Search for mosaic_images and mosaic_masks anywhere under /kaggle/input
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

# Original box-prompt teacher logits, if attached, get merged into the same output dir
input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists():
    input_dir = Path("/kaggle/input")
Path("data/teacher_logits_box").mkdir(parents=True, exist_ok=True)
for root, dirs, files in os.walk(str(input_dir)):
    if "teacher_logits_box" in dirs or "teacher_logits" in dirs:
        src = Path(root) / ("teacher_logits_box" if "teacher_logits_box" in dirs else "teacher_logits")
        for f in src.glob("*.npy"):
            link = Path("data/teacher_logits_box") / f.name
            if not link.exists():
                os.symlink(f, link)
        print(f"[Link] Merged {len(list(src.glob('*.npy')))} existing crop-scale logits from {src}")
        break
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

n = len(list(Path("data/teacher_logits_box").glob("*_mosaic_logits.npy")))
print(f"[Verify] {n} native-scale mosaic teacher logit files written.")
assert n > 0, "0 mosaic logits generated — check mosaic_images/mosaic_masks from notebook 01."
"""),
        make_cell("code", """print("Save this notebook's version — 03 needs BOTH data/teacher_logits_box (crop-scale + mosaic-scale merged) "
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
- `imgsz=640` (not 768): matches the native single-tile width so the majority of the set isn't force-upsampled the way notebook 10's 768px runs the original-only set through — see `possibleOODimprovements.md` §1.2 for why 768 alone was only a partial fix.
- Teacher logits merge the original crop-scale set with the native-scale mosaic logits generated in notebook 02.

**Trains from the stock pretrained `yolo11n-seg.pt` backbone, like every other notebook in this project** — no prior experiment's `best.pt` is used as a starting point, so this stays a clean, comparable ablation.

**Requires**: attach BOTH notebook 01's output (`crack500_yolo_augmented`) and notebook 02's output (`data/teacher_logits_box` with mosaic logits merged in)."""),
        make_cell("code", """# -- Environment & Directory Initialization --
!mkdir -p configs utils distillation scripts checkpoints data/datasets data/teacher_logits_box runs results
!pip install -q ultralytics albumentations pycocotools thop pyyaml pandas tqdm opencv-python Pillow
"""),
        make_cell("code", f"%%writefile configs/config.yaml\n{config_yaml}"),
        make_cell("code", "%%writefile utils/__init__.py\n# utils package"),
        make_cell("code", f"%%writefile utils/config_loader.py\n{config_loader_code}"),
        make_cell("code", "%%writefile distillation/__init__.py\n# distillation package"),
        make_cell("code", f"%%writefile distillation/kd_trainer.py\n{kd_trainer_code}"),
        make_cell("code", f"%%writefile scripts/convert_crack500_uncropped.py\n{convert_crack500_uncropped_code}"),
        make_cell("code", """# -- Step 1: Link notebook 01 (augmented dataset) + notebook 02 (native teacher logits) outputs --
import os, shutil
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

ok1 = find_and_link("crack500_yolo_augmented", "data/datasets/crack500_yolo_augmented")
assert ok1, "Attach notebook 01's output (crack500_yolo_augmented) before running."

ok2 = find_and_link(["teacher_logits_box", "teacher_logits"], "data/teacher_logits_box")
assert ok2, "Attach notebook 02's output (data/teacher_logits_box with mosaic logits merged) before running."

# Uncropped OOD val set, from the raw dataset attachment (for post-training validation)
input_dir = Path("/kaggle/input/distill_datasetforme")
if not input_dir.exists():
    input_dir = Path("/kaggle/input")
for root, dirs, files in os.walk(str(input_dir)):
    if "valdata" in dirs:
        !python scripts/convert_crack500_uncropped.py --src {root} --dst data/datasets/crack500_uncropped_yolo
        break
"""),
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
val_metrics = model.val(data="data/datasets/crack500_yolo_augmented/dataset.yaml", split="val", verbose=True)

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
    ood_metrics = model.val(data=str(uncropped_yaml), split="val", verbose=True)
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
    print("Copied final_notebooks/07 -> OODimprovements/04_eval_ood_and_tiled_inference.ipynb "
          "(reuse as-is: attach the new 03 checkpoint alongside the existing 7 for a fresh cross-checkpoint comparison)")


if __name__ == "__main__":
    main()
