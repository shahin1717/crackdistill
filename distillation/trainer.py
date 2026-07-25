"""
KD Trainer — Crack-Distill
===========================
Uses KDSegmentationTrainer (subclass of YOLOv8 SegmentationTrainer)
to inject KD loss directly inside the training step.

L = L_task + γ · L_boundary_kd
"""

import json
import shutil
import numpy as np
import torch
from pathlib import Path

from utils.config_loader import load_config, override_config


def optuna_callback(trainer):
    """Callback to report validation metrics to Optuna and check for pruning."""
    trial = getattr(trainer, "optuna_trial", None)
    if trial is not None:
        metrics = getattr(trainer, "metrics", {})
        # YOLOv8-seg logs segment metrics to metrics/mAP50(M) or metrics/mAP50-95(M)
        score = metrics.get("metrics/mAP50(M)", 0.0)
        offset = getattr(trainer, "optuna_epoch_offset", 0)
        step = offset + trainer.epoch
        
        trial.report(score, step=step)
        
        # Check if Optuna recommends pruning this trial
        import optuna
        if trial.should_prune():
            print(f"[Optuna Callback] Pruning trial {trial.number} at step {step} with score {score:.4f}")
            raise optuna.exceptions.TrialPruned()


class CrackDistillTrainer:

    def __init__(self, cfg_path: str = "configs/config.yaml", override_cfg=None):
        self.cfg = override_cfg if override_cfg is not None else load_config(cfg_path)
        self.optuna_trial = None


        self.backbone    = self.cfg.student.backbone
        pretrained_w     = getattr(self.cfg.student, "pretrained_weights", None)
        if pretrained_w and Path(str(pretrained_w)).exists():
            self.model_weights = str(pretrained_w)
            print(f"[Trainer] Using custom pretrained weights: {self.model_weights}")
        else:
            self.model_weights = f"{self.backbone}.pt" if not str(self.backbone).endswith(".pt") else str(self.backbone)

        self.imgsz       = self.cfg.student.imgsz
        self.epochs      = self.cfg.train.epochs
        self.batch       = self.cfg.data.batch_size
        self.logits_dir  = Path(str(self.cfg.teacher.logits_dir)).expanduser().resolve()
        # On Kaggle, redirect to /tmp to avoid exceeding 20GB disk limit
        if "/kaggle/" in str(self.logits_dir):
            self.logits_dir = Path("/tmp") / self.logits_dir.name
        self.kd_enabled  = self.cfg.distillation.enabled
        self.temperature = self.cfg.distillation.temperature
        self.kd_weight   = self.cfg.distillation.losses.boundary.weight
        self.workers     = int(self.cfg.data.num_workers)
        self.device      = str(getattr(self.cfg.student, "device", "cuda"))
        self.fraction    = float(getattr(self.cfg.data, "train_fraction", getattr(self.cfg.data, "fraction", 1.0)))

        # Dataset yaml
        ds  = self.cfg.data.datasets
        ds0 = ds[0] if isinstance(ds, list) else ds
        ds_path = Path(str(ds0.path if hasattr(ds0, "path") else ds0["path"])).expanduser().resolve()
        
        # If pointing to /kaggle/input, redirect to writable /kaggle/working symlink to use the patched dataset.yaml
        if "/kaggle/input/" in str(ds_path):
            ds_path = Path("/kaggle/working/data/datasets") / ds_path.name
            
        self.data_yaml = str(ds_path / "dataset.yaml")

        # Output dir — unique per experiment
        exp_name      = getattr(self.cfg.project, "experiment", "default")
        run_name      = f"{self.cfg.project.name}_{exp_name}_{self.cfg.task.type}_{self.backbone}"
        self.run_dir  = Path(self.cfg.project.output_dir).resolve() / run_name

        # Set environment variables for DDP processes to read trainer config safely
        import os
        os.environ["KD_LOGITS_DIR"] = str(self.logits_dir)
        os.environ["KD_CONFIG"] = json.dumps(self.cfg.distillation.dict())

        print(f"[Trainer] Task:       {self.cfg.task.type}")
        print(f"[Trainer] Student:    {self.backbone}")
        print(f"[Trainer] KD enabled: {self.kd_enabled}")
        print(f"[Trainer] Epochs:     {self.epochs}")
        print(f"[Trainer] Data:       {self.data_yaml}")
        print(f"[Trainer] Run dir:    {self.run_dir}")

    def train(self):
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from distillation.kd_trainer import KDSegmentationTrainer

        progressive_cfg = getattr(self.cfg.distillation, "progressive", None)
        is_progressive = self.kd_enabled and progressive_cfg is not None and getattr(progressive_cfg, "enabled", False)

        if is_progressive:
            print(f"[Trainer] Progressive Distillation Enabled!")
            total_epochs = int(self.epochs)
            
            # Read percentage configuration (falling back to absolute values if needed)
            if hasattr(progressive_cfg, "stage1_pct"):
                stage1_pct = float(progressive_cfg.stage1_pct)
                stage1_epochs = max(1, int(round(total_epochs * stage1_pct)))
                stage2_epochs = max(1, total_epochs - stage1_epochs)
            else:
                stage1_epochs = int(getattr(progressive_cfg, "stage1_epochs", 3))
                stage2_epochs = int(getattr(progressive_cfg, "stage2_epochs", 7))
                
            print(f"[Trainer] Total Epochs: {total_epochs}")
            print(f"[Trainer] Stage 1 (Backbone): {stage1_epochs} epochs")
            print(f"[Trainer] Stage 2 (End-to-End): {stage2_epochs} epochs")

            # Clean old runs for this experiment
            for suffix in ["", "_stage1", "_stage2"]:
                old = self.run_dir.parent / f"{self.run_dir.name}{suffix}"
                if old.exists():
                    shutil.rmtree(old)
            self.run_dir.mkdir(parents=True, exist_ok=True)

            from utils.config_loader import ConfigNode

            # ================= STAGE 1 =================
            print("\n>>> STARTING PROGRESSIVE STAGE 1: Distill Backbone (Head Frozen) <<<")
            stage1_dict = self.cfg.distillation.dict()
            if "progressive" not in stage1_dict:
                stage1_dict["progressive"] = {}
            stage1_dict["progressive"]["freeze_head"] = True
            stage1_kd_cfg = ConfigNode(stage1_dict)

            overrides_stage1 = dict(
                data        = self.data_yaml,
                epochs      = stage1_epochs,
                imgsz       = self.imgsz,
                batch       = self.batch,
                device      = self.device,
                fraction    = self.fraction,
                amp         = bool(self.cfg.train.amp),
                lr0         = float(self.cfg.train.lr),
                weight_decay= float(self.cfg.train.weight_decay),
                project     = str(self.run_dir.parent),
                name        = f"{self.run_dir.name}_stage1",
                exist_ok    = True,
                verbose     = True,
                model       = self.model_weights,
                workers     = self.workers,
            )
            cfg_obj_stage1 = get_cfg(overrides=overrides_stage1)
            trainer_stage1 = KDSegmentationTrainer(
                cfg         = cfg_obj_stage1,
                logits_dir  = self.logits_dir,
                kd_cfg      = stage1_kd_cfg,
            )
            if self.optuna_trial is not None:
                trainer_stage1.optuna_trial = self.optuna_trial
                trainer_stage1.optuna_epoch_offset = 0
                trainer_stage1.add_callback("on_fit_epoch_end", optuna_callback)
            trainer_stage1.train()

            # Find best weight of Stage 1
            stage1_run_dir = self.run_dir.parent / f"{self.run_dir.name}_stage1"
            stage1_best = stage1_run_dir / "weights/best.pt"
            if not stage1_best.exists():
                for candidate in sorted(self.run_dir.parent.glob(f"{self.run_dir.name}_stage1*/weights/best.pt")):
                    stage1_best = candidate
                    break
            print(f"[Trainer] Stage 1 finished. Best checkpoint loaded from: {stage1_best}")

            # ================= STAGE 2 =================
            print("\n>>> STARTING PROGRESSIVE STAGE 2: Distill Full Pipeline <<<")
            stage2_dict = self.cfg.distillation.dict()
            if "progressive" not in stage2_dict:
                stage2_dict["progressive"] = {}
            if not self.cfg.distillation.progressive.get("freeze_head", False):
                stage2_dict["progressive"]["freeze_head"] = False
            else:
                stage2_dict["progressive"]["freeze_head"] = True
            stage2_kd_cfg = ConfigNode(stage2_dict)

            overrides_stage2 = dict(
                data        = self.data_yaml,
                epochs      = stage2_epochs,
                imgsz       = self.imgsz,
                batch       = self.batch,
                device      = self.device,
                fraction    = self.fraction,
                amp         = bool(self.cfg.train.amp),
                lr0         = float(self.cfg.train.lr),
                weight_decay= float(self.cfg.train.weight_decay),
                project     = str(self.run_dir.parent),
                name        = f"{self.run_dir.name}_stage2",
                exist_ok    = True,
                verbose     = True,
                model       = str(stage1_best),
                workers     = self.workers,
            )
            cfg_obj_stage2 = get_cfg(overrides=overrides_stage2)
            trainer_stage2 = KDSegmentationTrainer(
                cfg         = cfg_obj_stage2,
                logits_dir  = self.logits_dir,
                kd_cfg      = stage2_kd_cfg,
            )
            if self.optuna_trial is not None:
                trainer_stage2.optuna_trial = self.optuna_trial
                trainer_stage2.optuna_epoch_offset = stage1_epochs
                trainer_stage2.add_callback("on_fit_epoch_end", optuna_callback)
            trainer_stage2.train()

            # Find best weight of Stage 2
            stage2_run_dir = self.run_dir.parent / f"{self.run_dir.name}_stage2"
            stage2_best = stage2_run_dir / "weights/best.pt"
            if not stage2_best.exists():
                for candidate in sorted(self.run_dir.parent.glob(f"{self.run_dir.name}_stage2*/weights/best.pt")):
                    stage2_best = candidate
                    break

            # Copy stage 2 best weights to main run directory
            main_weights_dir = self.run_dir / "weights"
            main_weights_dir.mkdir(parents=True, exist_ok=True)
            self.best_pt = main_weights_dir / "best.pt"
            shutil.copy2(str(stage2_best), str(self.best_pt))
            print(f"[Trainer] Progressive training complete. Copied final weights to: {self.best_pt}")

        elif self.kd_enabled:
            # Clean old runs for this experiment
            for old in self.run_dir.parent.glob(f"{self.run_dir.name}*"):
                shutil.rmtree(old)
            self.run_dir.mkdir(parents=True, exist_ok=True)

            print(f"[Trainer] Using KDSegmentationTrainer")

            # Build overrides dict for YOLO
            overrides = dict(
                data        = self.data_yaml,
                epochs      = self.epochs,
                imgsz       = self.imgsz,
                batch       = self.batch,
                device      = self.device,
                fraction    = self.fraction,
                amp         = bool(self.cfg.train.amp),
                lr0         = float(self.cfg.train.lr),
                weight_decay= float(self.cfg.train.weight_decay),
                project     = str(self.run_dir.parent),
                name        = self.run_dir.name,
                exist_ok    = True,
                verbose     = True,
                model       = self.model_weights,
                workers     = self.workers,
            )
 
            # Create KD trainer directly
            cfg_obj = get_cfg(overrides=overrides)
            trainer = KDSegmentationTrainer(
                cfg         = cfg_obj,
                logits_dir  = self.logits_dir,
                kd_cfg      = self.cfg.distillation,
            )
            if self.optuna_trial is not None:
                trainer.optuna_trial = self.optuna_trial
                trainer.optuna_epoch_offset = 0
                trainer.add_callback("on_fit_epoch_end", optuna_callback)
            trainer.train()

        else:
            print(f"[Trainer] Baseline — no KD")
            model = YOLO(f"{self.backbone}.pt")
            model.train(
                data        = self.data_yaml,
                epochs      = self.epochs,
                imgsz       = self.imgsz,
                batch       = self.batch,
                device      = self.device,
                fraction    = self.fraction,
                amp         = bool(self.cfg.train.amp),
                lr0         = float(self.cfg.train.lr),
                weight_decay= float(self.cfg.train.weight_decay),
                project     = str(self.run_dir.parent),
                name        = self.run_dir.name,
                exist_ok    = True,
                verbose     = True,
                workers     = self.workers,
            )


        # Find best.pt
        self.best_pt = self.run_dir / "weights/best.pt"
        if not self.best_pt.exists():
            for candidate in sorted(self.run_dir.parent.glob(
                    f"{self.run_dir.name}*/weights/best.pt")):
                self.best_pt = candidate
                break

        print(f"[Trainer] Best model: {self.best_pt}")

    def test(self) -> dict:
        from ultralytics import YOLO

        if not hasattr(self, "best_pt") or not self.best_pt.exists():
            self.best_pt = self.run_dir / "weights/best.pt"
            if not self.best_pt.exists():
                for candidate in sorted(self.run_dir.parent.glob(
                        f"{self.run_dir.name}*/weights/best.pt")):
                    self.best_pt = candidate
                    break

        if not self.best_pt.exists():
            print("[Trainer] No trained model found.")
            return {}

        model   = YOLO(str(self.best_pt))
        metrics = model.val(
            data    = self.data_yaml,
            imgsz   = self.imgsz,
            device  = "cuda:0" if "cuda" in self.device or "," in self.device else self.device,
            verbose = True,
        )

        results = {
            "mAP50-box":    float(metrics.box.map50),
            "mAP50-seg":    float(metrics.seg.map50),
            "mAP50-95-seg": float(metrics.seg.map),
            "model":        str(self.best_pt),
        }

        out = self.run_dir / "test_results.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

        print("\n[Test Results]")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"  Saved to: {out}")

        return results