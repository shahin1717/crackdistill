import json
import glob
from pathlib import Path

# Load updated kd_trainer.py content from disk
with open("distillation/kd_trainer.py") as f:
    kd_trainer_code_raw = f.read()

kd_trainer_cell7_source = ["%%writefile distillation/kd_trainer.py\n"] + [line + "\n" for line in kd_trainer_code_raw.split("\n")]

def get_cell_11_code(dset_name="crack500"):
    raw_folder = "traincrop" if dset_name == "crack500" else "train_img"
    return f"""import os, shutil
from pathlib import Path

# Search directories for datasets and logits
input_dirs = [Path("/kaggle/input"), Path("data")]

datasets_dir = Path("data/datasets")
datasets_dir.mkdir(parents=True, exist_ok=True)
checkpoints_dir = Path("checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)
logits_dst = Path("data/teacher_logits_box")
logits_dst.mkdir(parents=True, exist_ok=True)
features_dst = Path("data/teacher_features")
features_dst.mkdir(parents=True, exist_ok=True)

print("[Dataset & Logit Setup] Scanning input directories for {dset_name}...")

# Function to safely fix hardcoded absolute paths in dataset.yaml.
# data/datasets/crack500_yolo is a SYMLINK to read-only /kaggle/input.
# We NEVER write to the symlink target. Instead we copy dataset.yaml to
# a writable local path (data/datasets_yaml/<name>/dataset.yaml) and use that.
def fix_dataset_yaml_path(yaml_path):
    p = Path(yaml_path)
    if not p.exists():
        return yaml_path  # Return original path if not found
    real_p = p.resolve()
    abs_dset_dir = str(p.parent.resolve())
    original_text = real_p.read_text()
    fixed_lines = [f"path: {{abs_dset_dir}}" if l.strip().startswith("path:") else l for l in original_text.splitlines()]
    fixed_text = "\\n".join(fixed_lines) + "\\n"
    if fixed_text == original_text:
        print(f"✓ dataset.yaml path already correct: {{abs_dset_dir}}")
        return yaml_path
    # Try in-place write first (works on writable filesystems)
    try:
        p.write_text(fixed_text)
        print(f"✓ Fixed dataset.yaml in-place: {{abs_dset_dir}}")
        return yaml_path
    except OSError:
        pass
    # Kaggle input is read-only: copy to writable location
    local_dir = Path("data/datasets_yaml") / p.parent.name
    local_dir.mkdir(parents=True, exist_ok=True)
    local_yaml = local_dir / "dataset.yaml"
    local_yaml.write_text(fixed_text)
    print(f"✓ Read-only symlink — wrote fixed dataset.yaml to: {{local_yaml}}")
    return str(local_yaml)

# Track fixed YAML paths for use in training
fixed_crack500_yaml = fix_dataset_yaml_path("data/datasets/crack500_yolo/dataset.yaml")
fixed_deepcrack_yaml = fix_dataset_yaml_path("data/datasets/deepcrack_yolo/dataset.yaml")
print(f"crack500_yolo yaml: {{fixed_crack500_yaml}}")
print(f"deepcrack_yolo yaml: {{fixed_deepcrack_yaml}}")

# 1. Link raw datasets and pre-converted YOLO datasets
for inp in input_dirs:
    if not inp.exists(): continue
    for root, dirs, files in os.walk(str(inp)):
        root_path = Path(root)
        name_lower = root_path.name.lower()
        
        # Link raw or pre-converted datasets matching crack500 / deepcrack
        if name_lower in ["crack500", "crack500_yolo", "deepcrack", "deepcrack_yolo"] or "{raw_folder}" in dirs:
            target_name = "crack500" if "crack500" in name_lower or "{raw_folder}" == "traincrop" else "deepcrack"
            if name_lower.endswith("_yolo"):
                target_name = name_lower
            dest = datasets_dir / target_name
            if not dest.exists():
                try:
                    os.symlink(root_path, dest)
                    print(f"Linked dataset: {{root_path}} -> {{dest}}")
                except Exception:
                    pass

# Fix dataset.yaml paths if datasets exist
fix_dataset_yaml_path("data/datasets/crack500_yolo/dataset.yaml")
fix_dataset_yaml_path("data/datasets/deepcrack_yolo/dataset.yaml")


# 2. Find and link/copy all precomputed teacher logit and feature files
logit_linked_count = 0
feat_linked_count = 0

for inp in input_dirs:
    if not inp.exists(): continue
    for root, dirs, files in os.walk(str(inp)):
        root_p = Path(root)
        
        # Match logit npy files
        npy_files = [f for f in files if f.endswith("_logits.npy") or (f.endswith(".npy") and ("logit" in root_p.name.lower() or "teacher" in root_p.name.lower()))]
        if npy_files:
            for f in npy_files:
                src_file = root_p / f
                dst_file = logits_dst / f
                if not dst_file.exists():
                    try:
                        os.symlink(src_file, dst_file)
                    except Exception:
                        shutil.copy2(src_file, dst_file)
                    logit_linked_count += 1

        # Match feature npz files
        npz_files = [f for f in files if f.endswith("_features.npz") or (f.endswith(".npz") and ("feat" in root_p.name.lower() or "teacher" in root_p.name.lower()))]
        if npz_files:
            for f in npz_files:
                src_file = root_p / f
                dst_file = features_dst / f
                if not dst_file.exists():
                    try:
                        os.symlink(src_file, dst_file)
                    except Exception:
                        shutil.copy2(src_file, dst_file)
                    feat_linked_count += 1

total_logits = len(list(logits_dst.glob("*.npy")))
total_feats = len(list(features_dst.glob("*.npz")))

print(f"✓ Teacher logits ready: {{total_logits}} files in {{logits_dst}} (newly linked: {{logit_linked_count}})")
print(f"✓ Teacher features ready: {{total_feats}} files in {{features_dst}} (newly linked: {{feat_linked_count}})")
"""

def get_cell_12_code(dset_name="crack500"):
    script = "convert_crack500.py" if dset_name == "crack500" else "convert_deepcrack.py"
    return f"""# Ensure YOLO dataset & SAM 2 teacher logits exist
import os
from pathlib import Path

# Convert raw dataset to YOLO format if pre-converted dataset is not already present
yolo_dst = Path("data/datasets/{dset_name}_yolo")
if not (yolo_dst / "dataset.yaml").exists():
    os.system("python scripts/{script} --src data/datasets/{dset_name} --dst data/datasets/{dset_name}_yolo")

# Dynamic fix for dataset.yaml path (safe — handles read-only Kaggle symlinks)
yaml_file = yolo_dst / "dataset.yaml"
if yaml_file.exists():
    abs_dset_dir = str(yolo_dst.resolve())
    real_file = yaml_file.resolve()
    original_text = real_file.read_text()
    fixed_lines = [f"path: {{abs_dset_dir}}" if l.strip().startswith("path:") else l for l in original_text.splitlines()]
    fixed_text = "\\n".join(fixed_lines) + "\\n"
    if fixed_text != original_text:
        try:
            yaml_file.write_text(fixed_text)
            print(f"✓ Verified dataset.yaml path -> {{abs_dset_dir}}")
        except OSError:
            # Read-only: write to local copy
            local_dir = Path("data/datasets_yaml") / yolo_dst.name
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "dataset.yaml").write_text(fixed_text)
            print(f"✓ Read-only — copied dataset.yaml to {{local_dir}}")
    else:
        print(f"✓ Verified dataset.yaml path -> {{abs_dset_dir}}")

logits_dir = Path("data/teacher_logits_box")
logits_count = len(list(logits_dir.glob("*.npy"))) if logits_dir.exists() else 0
print(f"Found {{logits_count}} teacher logit files in {{logits_dir}}")

if logits_count == 0:
    print("=== WARNING: No precomputed SAM 2 logits found. Attempting generation... ===")
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = checkpoints_dir / "sam2_hiera_large.pt"
    if not ckpt_file.exists():
        os.system("wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2_hiera_large.pt")
    os.system("pip install -q SAM-2 || pip install -q git+https://github.com/facebookresearch/segment-anything-2.git")
    os.system("python scripts/generate_teacher_logits.py --prompt-type box --logits-dir data/teacher_logits_box --dataset data/datasets/{dset_name}_yolo")
    logits_count = len(list(logits_dir.glob("*.npy")))
    print(f"Post-generation logit count: {{logits_count}}")

# HARD ASSERTION: Stop execution immediately if 0 logit files are found
err_msg = f"FATAL ERROR: Found {{logits_count}} teacher logit files in {{logits_dir}}! Cannot run Knowledge Distillation without teacher logits."
assert logits_count > 0, err_msg
"""

def clean_cell_13(exp_name, dset_name, mask_kd=True, feat_mse=True, boundary_bce=True, is_seghead_frozen=False):
    extra = ""
    if is_seghead_frozen:
        extra = """,
    "distillation.progressive.enabled": True,
    "distillation.progressive.freeze_head": True,
    "distillation.progressive.unfreeze_epoch_ratio": 1.0"""

    yaml_var = f"fixed_{dset_name}_yaml"
    code = f"""import sys
from pathlib import Path
from distillation.kd_trainer import KDSegmentationTrainer
from utils.config_loader import load_config, override_config

# Pre-training verification: ensure logit files are present
logits_dir = Path("data/teacher_logits_box")
logits_count = len(list(logits_dir.glob("*.npy"))) if logits_dir.exists() else 0
err_msg = f"FATAL ERROR: Found {{logits_count}} teacher logit files in {{logits_dir}}! Cannot start run without teacher logits."
assert logits_count > 0, err_msg

# Use the fixed yaml path returned by Cell 11 (handles read-only symlinks)
try:
    _data_yaml = {yaml_var}
except NameError:
    _data_yaml = "data/datasets/{dset_name}_yolo/dataset.yaml"
print(f"Using dataset yaml: {{_data_yaml}}")

cfg = load_config("configs/config.yaml")
cfg = override_config(cfg, {{
    "project.name": "crack_distill",
    "project.experiment": "{exp_name}",
    "data.datasets": [
        {{
            "name": "{dset_name}",
            "path": "data/datasets/{dset_name}_yolo",
            "format": "yolo"
        }}
    ],
    "distillation.enabled": True,
    "distillation.losses.mask_kd.enabled": {mask_kd},
    "distillation.losses.feature.enabled": {feat_mse},
    "distillation.losses.boundary.enabled": {boundary_bce},
    "teacher.logits_dir": "data/teacher_logits_box/"{extra}
}})

trainer = KDSegmentationTrainer(cfg, overrides={{"data": _data_yaml}})
trainer.train()
print("✓ {exp_name} completed!")
"""
    return [line + "\n" for line in code.split("\n")]


def main():
    ablation_nbs = [
        ("kaggle_notebooks/nb5a_ablation_no_mask_kd.ipynb", "crack500", "ablation_no_mask_kd", False, True, True, False),
        ("kaggle_notebooks/nb5b_ablation_no_feature_mse.ipynb", "crack500", "ablation_no_feature", True, False, True, False),
        ("kaggle_notebooks/nb5c_ablation_no_boundary_bce.ipynb", "crack500", "ablation_no_boundary", True, True, False, False),
        ("kaggle_notebooks/nb5d_ablation_seghead_frozen.ipynb", "deepcrack", "ablation_seghead_frozen", True, True, True, True),
        ("kaggle_notebooks/nb2_crack500_kd.ipynb", "crack500", "crack500_kd", True, True, True, False),
        ("kaggle_notebooks/nb3_deepcrack_kd.ipynb", "deepcrack", "deepcrack_kd", True, True, True, False),
    ]

    for nb_path, dset, exp_name, mask_kd, feat_mse, boundary_bce, is_seghead_frozen in ablation_nbs:
        with open(nb_path) as f:
            data = json.load(f)

        # Update Cell 7 (kd_trainer.py embedded code)
        data["cells"][7]["source"] = kd_trainer_cell7_source

        # Update Cell 11
        if len(data["cells"]) > 11:
            data["cells"][11]["source"] = [line + "\n" for line in get_cell_11_code(dset).split("\n")]
        
        # Update Cell 12
        if len(data["cells"]) > 12:
            data["cells"][12]["source"] = [line + "\n" for line in get_cell_12_code(dset).split("\n")]

        # Update Cell 13
        if len(data["cells"]) > 13:
            data["cells"][13]["source"] = clean_cell_13(exp_name, dset, mask_kd, feat_mse, boundary_bce, is_seghead_frozen)

        with open(nb_path, "w") as f:
            json.dump(data, f, indent=1)
        print(f"✓ Fixed cfg, booleans, and dataset.yaml pathing in notebook {nb_path}")

    # Also update nb6_batch_size_sensitivity.ipynb
    nb6_path = "kaggle_notebooks/nb6_batch_size_sensitivity.ipynb"
    with open(nb6_path) as f:
        nb6_data = json.load(f)
    
    nb6_data["cells"][7]["source"] = kd_trainer_cell7_source
    nb6_data["cells"][8]["source"] = [line + "\n" for line in get_cell_11_code("crack500").split("\n")]
    nb6_data["cells"][9]["source"] = [line + "\n" for line in get_cell_12_code("crack500").split("\n")]
    
    with open(nb6_path, "w") as f:
        json.dump(nb6_data, f, indent=1)
    print(f"✓ Fixed notebook {nb6_path}")

if __name__ == "__main__":
    main()
