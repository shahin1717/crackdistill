"""
Kaggle Patch for CrackDistill
Fixes AttributeError: 'ConfigNode' object has no attribute 'keys' when initializing KDSegmentationTrainer(cfg).

Usage in Kaggle:
Run this file at the top of your notebook cell:
    %run scripts/apply_kaggle_patch.py
or paste the content directly into a Kaggle code cell.
"""

import sys
import os

# Add current working directory and repo root to sys.path
for p in [".", "/kaggle/working", os.getcwd()]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 1. Verify ConfigNode in utils.config_loader
try:
    from utils.config_loader import ConfigNode
    print("✓ Verified ConfigNode dictionary interface")
except Exception as e:
    print(f"ConfigNode import warning: {e}")

# 2. Patch KDSegmentationTrainer.__init__ to parse master ConfigNode / dict automatically
try:
    from distillation.kd_trainer import KDSegmentationTrainer
    from ultralytics.cfg import get_cfg

    original_init = KDSegmentationTrainer.__init__

    def patched_init(self, cfg=None, overrides=None, _callbacks=None, logits_dir=None, kd_cfg=None, **kwargs):
        if hasattr(cfg, "student") or (isinstance(cfg, dict) and "student" in cfg):
            master_cfg = cfg
            if kd_cfg is None:
                if hasattr(master_cfg, "distillation"):
                    kd_cfg = master_cfg.distillation
                elif isinstance(master_cfg, dict) and "distillation" in master_cfg:
                    kd_cfg = master_cfg["distillation"]

            if logits_dir is None:
                if hasattr(master_cfg, "teacher") and hasattr(master_cfg.teacher, "logits_dir"):
                    logits_dir = master_cfg.teacher.logits_dir
                elif isinstance(master_cfg, dict) and "teacher" in master_cfg and "logits_dir" in master_cfg["teacher"]:
                    logits_dir = master_cfg["teacher"]["logits_dir"]

            model_name = "yolo11n-seg"
            if hasattr(master_cfg, "student") and hasattr(master_cfg.student, "backbone"):
                model_name = master_cfg.student.backbone
            elif isinstance(master_cfg, dict) and "student" in master_cfg and "backbone" in master_cfg["student"]:
                model_name = master_cfg["student"]["backbone"]
            if not str(model_name).endswith(".pt"):
                model_name = f"{model_name}.pt"

            data_path = "data/datasets/crack500_yolo/dataset.yaml"
            if hasattr(master_cfg, "data"):
                if hasattr(master_cfg.data, "datasets") and len(master_cfg.data.datasets) > 0:
                    d_p = master_cfg.data.datasets[0].path
                    data_path = d_p if str(d_p).endswith(".yaml") else os.path.join(d_p, "dataset.yaml")
                elif hasattr(master_cfg.data, "path"):
                    d_p = master_cfg.data.path
                    data_path = d_p if str(d_p).endswith(".yaml") else os.path.join(d_p, "dataset.yaml")
            elif isinstance(master_cfg, dict) and "data" in master_cfg:
                d_dict = master_cfg["data"]
                if "datasets" in d_dict and len(d_dict["datasets"]) > 0:
                    d_p = d_dict["datasets"][0].get("path", data_path) if isinstance(d_dict["datasets"][0], dict) else getattr(d_dict["datasets"][0], "path", data_path)
                    data_path = d_p if str(d_p).endswith(".yaml") else os.path.join(d_p, "dataset.yaml")

            check_data = (overrides.get("data") if overrides and isinstance(overrides, dict) else None) or data_path
            if check_data and not Path(check_data).exists():
                for alt_candidate in [
                    Path("data/datasets/combined_yolo/dataset.yaml"),
                    Path("data/datasets/crack500_yolo/dataset.yaml"),
                ]:
                    if alt_candidate.exists():
                        check_data = str(alt_candidate)
                        data_path = check_data
                        break

                if not Path(check_data).exists() and Path("/kaggle/input").exists():
                    kaggle_yamls = sorted(list(Path("/kaggle/input").glob("**/dataset.yaml")))
                    if kaggle_yamls:
                        preferred = [y for y in kaggle_yamls if any(k in str(y).lower() for k in ["crack500", "combined", "distill"])]
                        chosen_yaml = preferred[0] if preferred else kaggle_yamls[0]
                        check_data = str(chosen_yaml)
                        data_path = check_data
                        print(f"[KD Patch] Auto-detected Kaggle dataset: {chosen_yaml}")

            proj_name = getattr(getattr(master_cfg, "project", None), "name", "runs")
            exp_name = getattr(getattr(master_cfg, "project", None), "experiment", "exp")

            seed_val = int(getattr(getattr(master_cfg, "project", None), "seed", 42))
            auto_overrides = {
                "model": model_name,
                "data": data_path,
                "epochs": getattr(getattr(master_cfg, "train", None), "epochs", 150),
                "imgsz": getattr(getattr(master_cfg, "student", None), "imgsz", getattr(getattr(master_cfg, "data", None), "image_size", 512)),
                "batch": getattr(getattr(master_cfg, "data", None), "batch_size", 16),
                "amp": getattr(getattr(master_cfg, "train", None), "amp", False),
                "lr0": getattr(getattr(master_cfg, "train", None), "lr", 0.001),
                "weight_decay": getattr(getattr(master_cfg, "train", None), "weight_decay", 0.0005),
                "project": str(proj_name),
                "name": str(exp_name),
                "seed": seed_val,
                "exist_ok": True,
                "task": "segment",
            }

            is_kd_on = (hasattr(kd_cfg, "enabled") and getattr(kd_cfg, "enabled")) or (isinstance(kd_cfg, dict) and kd_cfg.get("enabled", False))
            allow_spatial_aug = getattr(getattr(master_cfg, "train", None), "allow_spatial_aug", False)
            if is_kd_on and not allow_spatial_aug:
                auto_overrides.update({
                    "mosaic": 0.0,
                    "close_mosaic": 0,
                    "degrees": 0.0,
                    "translate": 0.0,
                    "scale": 0.0,
                    "shear": 0.0,
                    "perspective": 0.0,
                    "fliplr": 0.0,
                    "flipud": 0.0,
                    "erasing": 0.0,
                })
                print(f"[KD Patch] Geometric Alignment Enforced: Spatial augmentations disabled. Seed: {seed_val}")

            if overrides and isinstance(overrides, dict):
                auto_overrides.update(overrides)

            cfg = get_cfg(overrides=auto_overrides)
            overrides = None

        original_init(self, cfg=cfg, overrides=overrides, _callbacks=_callbacks, logits_dir=logits_dir, kd_cfg=kd_cfg, **kwargs)

    KDSegmentationTrainer.__init__ = patched_init
    print("✓ Patched KDSegmentationTrainer.__init__ for master ConfigNode support")
except Exception as e:
    print(f"KDSegmentationTrainer patch warning: {e}")
