#!/usr/bin/env python3
"""
Confirmatory experiment runner and smoke-test verification harness.
Addresses Problem P2-1:
- Executes post-remediation minimal confirmatory sets (Baseline vs Mask-KD vs GT-Soft).
- Supports --smoke-test mode (1 epoch, reduced resolution) to verify complete pipeline health in <60 seconds.
- Enforces strict augmentation parity, deterministic coordinate warping, and SHA256 provenance logging.

Usage:
  python scripts/run_confirmatory_experiments.py --smoke-test
  python scripts/run_confirmatory_experiments.py --suite minimal
  python scripts/run_confirmatory_experiments.py --suite multiseed
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config, override_config
from utils.checkpoint import resolve_checkpoint, get_checkpoint_manifest, save_checkpoint_manifest
from distillation.kd_trainer import KDSegmentationTrainer
import hashlib


def compute_sha256(path: Path) -> str:
    """Compute SHA256 checksum for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_experiment_command(
    variant_name: str,
    seed: int = 42,
    is_smoke_test: bool = False,
    epochs: int = 150,
    batch_size: int = 8,
    device: str = "cuda"
) -> list:
    """Build the command-line arguments for training an experiment arm."""
    cmd = [
        "python", "train.py",
        f"--experiment={variant_name}",
        f"--seed={seed}",
        f"--device={device}"
    ]
    if is_smoke_test:
        cmd.extend(["train.epochs=1", "data.batch_size=2", "student.imgsz=256"])
    else:
        cmd.extend([f"train.epochs={epochs}", f"data.batch_size={batch_size}"])
    return cmd


CONFIRMATORY_SUITES: Dict[str, Dict[str, Any]] = {
    "baseline_clean_seed42": {
        "description": "Baseline Fine-Tuning Control (Seed 42, No KD)",
        "seed": 42,
        "overrides": {
            "distillation.enabled": False,
            "distillation.losses.mask_kd.enabled": False,
            "distillation.losses.feature.enabled": False,
            "distillation.losses.boundary.enabled": False,
            "distillation.losses.affinity.enabled": False,
            "distillation.losses.tversky.enabled": False,
        }
    },
    "mask_kd_production_seed42": {
        "description": "Production Mask-KL Treatment (Seed 42, tau=3.7769, W=0.9612)",
        "seed": 42,
        "overrides": {
            "distillation.enabled": True,
            "distillation.temperature": 3.7769,
            "distillation.losses.task.weight": 1.0,
            "distillation.losses.mask_kd.enabled": True,
            "distillation.losses.mask_kd.weight": 0.9612,
            "distillation.losses.feature.enabled": False,
            "distillation.losses.boundary.enabled": False,
            "teacher.logits_dir": "data/teacher_logits_box/",
        }
    },
    "baseline_clean_seed123": {
        "description": "Baseline Fine-Tuning Control (Seed 123, No KD)",
        "seed": 123,
        "overrides": {
            "distillation.enabled": False,
            "distillation.losses.mask_kd.enabled": False,
            "distillation.losses.feature.enabled": False,
            "distillation.losses.boundary.enabled": False,
            "distillation.losses.affinity.enabled": False,
            "distillation.losses.tversky.enabled": False,
        }
    },
    "mask_kd_production_seed123": {
        "description": "Production Mask-KL Treatment (Seed 123, tau=3.7769, W=0.9612)",
        "seed": 123,
        "overrides": {
            "distillation.enabled": True,
            "distillation.temperature": 3.7769,
            "distillation.losses.task.weight": 1.0,
            "distillation.losses.mask_kd.enabled": True,
            "distillation.losses.mask_kd.weight": 0.9612,
            "distillation.losses.feature.enabled": False,
            "distillation.losses.boundary.enabled": False,
            "teacher.logits_dir": "data/teacher_logits_box/",
        }
    },
    "gt_soft_control_seed42": {
        "description": "GT-Soft Supervision Control Arm (Seed 42, sigma=2.0 Gaussian Softening)",
        "seed": 42,
        "overrides": {
            "distillation.enabled": True,
            "distillation.temperature": 3.7769,
            "distillation.losses.task.weight": 1.0,
            "distillation.losses.mask_kd.enabled": True,
            "distillation.losses.mask_kd.weight": 0.9612,
            "distillation.losses.feature.enabled": False,
            "distillation.losses.boundary.enabled": False,
            "teacher.logits_dir": "data/teacher_logits_gt_soft/",
        }
    },
}


def run_confirmatory_arm(
    exp_name: str,
    base_cfg_path: str = "configs/config.yaml",
    smoke_test: bool = False,
    data_override: str = None,
) -> dict:
    """
    Execute a single confirmatory arm.
    """
    arm_spec = CONFIRMATORY_SUITES[exp_name]
    print("\n" + "=" * 65)
    print(f"CONFIRMATORY ARM: {exp_name}")
    print(f"Description     : {arm_spec['description']}")
    print(f"Seed            : {arm_spec['seed']}")
    print(f"Smoke Test Mode : {smoke_test}")
    print("=" * 65)

    cfg = load_config(base_cfg_path)
    cfg = override_config(cfg, arm_spec["overrides"])
    cfg = override_config(cfg, {
        "project.name": "confirmatory_runs",
        "project.experiment": exp_name,
        "project.seed": arm_spec["seed"],
    })

    if smoke_test:
        cfg = override_config(cfg, {
            "train.epochs": 1,
            "train.amp": False,
            "data.batch_size": 2,
            "student.imgsz": 256,
        })

    if data_override:
        if hasattr(cfg, "data") and hasattr(cfg.data, "datasets") and len(cfg.data.datasets) > 0:
            cfg.data.datasets[0].path = data_override

    trainer = KDSegmentationTrainer(cfg=cfg)
    results = trainer.train()

    metrics = {}
    if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
        metrics = {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in results.results_dict.items()}

    best_ckpt = getattr(trainer, "best", None) or (Path(getattr(trainer, "save_dir", "runs")) / "weights" / "best.pt")
    if best_ckpt and Path(best_ckpt).exists():
        try:
            resolved = resolve_checkpoint(best_ckpt)
            manifest = get_checkpoint_manifest(
                checkpoint_path=resolved,
                experiment_name=exp_name,
                seed=arm_spec["seed"],
                metadata={"description": arm_spec["description"], "smoke_test": smoke_test}
            )
            manifest_p = save_checkpoint_manifest(manifest)
            metrics["checkpoint"] = str(resolved)
            metrics["sha256"] = manifest["sha256"]
            print(f"✓ Recorded manifest: {manifest_p}")
        except Exception as e:
            print(f"[Warning] Manifest saving skipped: {e}")

    return {exp_name: metrics}


def main():
    parser = argparse.ArgumentParser(description="Run post-remediation confirmatory experiments.")
    parser.add_argument("--suite", type=str, default="minimal", choices=["minimal", "multiseed", "all", "gt_soft"], help="Experiment suite to run")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid 1-epoch smoke test to verify pipeline")
    parser.add_argument("--data", type=str, default=None, help="Optional dataset override path")
    args = parser.parse_args()

    if args.suite == "minimal":
        exps = ["baseline_clean_seed42", "mask_kd_production_seed42"]
    elif args.suite == "multiseed":
        exps = ["baseline_clean_seed42", "mask_kd_production_seed42", "baseline_clean_seed123", "mask_kd_production_seed123"]
    elif args.suite == "gt_soft":
        exps = ["baseline_clean_seed42", "gt_soft_control_seed42", "mask_kd_production_seed42"]
    else:
        exps = list(CONFIRMATORY_SUITES.keys())

    all_results = {}
    for exp in exps:
        res = run_confirmatory_arm(exp, smoke_test=args.smoke_test, data_override=args.data)
        all_results.update(res)

    print("\n" + "=" * 65)
    print("CONFIRMATORY SUITE EXECUTION SUMMARY:")
    for exp, met in all_results.items():
        print(f"\nArm: {exp}")
        for k, v in met.items():
            print(f"  {k}: {v}")
    print("=" * 65)


if __name__ == "__main__":
    main()
