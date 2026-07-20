#!/usr/bin/env python3
"""
Run all Crack-Distill experiments in sequence.
Paper experiments in the correct order:
  1. baseline_finetune  — lower bound
  2. pseudo_labels      — SAM quality alone
  3. full_kd            — main result
  4-7. low_data_*       — H2 hypothesis
  8. robustness         — deployment claim

Usage:
  python scripts/run_experiments.py --exp all
  python scripts/run_experiments.py --exp full_kd
  python scripts/run_experiments.py --exp baseline_finetune
"""

import argparse
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.environ["PYTHONPATH"] = project_root + os.pathsep + os.environ.get("PYTHONPATH", "")

from utils.config_loader import load_config, override_config
from distillation.trainer import CrackDistillTrainer


EXPERIMENTS = {
    "baseline_finetune": {
        "description": "Lower bound — YOLO11 fine-tuned, no KD",
        "overrides": {
            "distillation.enabled": False,
        }
    },
    "pseudo_labels": {
        "description": "SAM quality alone — hard pseudo-labels, no soft KD",
        "overrides": {
            "distillation.enabled": True,
            "distillation.losses.mask_kd.enabled": False,
            "distillation.losses.feature.enabled": False,
            "distillation.losses.boundary.enabled": False,
        }
    },
    "full_kd_box": {
        "description": "Full KD pipeline — Bounding box only prompts",
        "overrides": {
            "distillation.enabled": True,
            "distillation.losses.mask_kd.enabled": True,
            "distillation.losses.feature.enabled": True,
            "distillation.losses.boundary.enabled": True,
            "teacher.logits_dir": "data/teacher_logits_box/",
        }
    },
    "full_kd_centroid": {
        "description": "Full KD pipeline — Bounding box + Centroid point prompts",
        "overrides": {
            "distillation.enabled": True,
            "distillation.losses.mask_kd.enabled": True,
            "distillation.losses.feature.enabled": True,
            "distillation.losses.boundary.enabled": True,
            "teacher.logits_dir": "data/teacher_logits_centroid/",
        }
    },
    "low_data_5pct": {
        "description": "H2: KD in low-data regime (5% of training set)",
        "overrides": {
            "distillation.enabled": True,
            "data.train_fraction": 0.05,
        }
    },
    "low_data_10pct": {
        "description": "H2: KD in low-data regime (10%)",
        "overrides": {
            "distillation.enabled": True,
            "data.train_fraction": 0.10,
        }
    },
    "low_data_25pct": {
        "description": "H2: KD in low-data regime (25%)",
        "overrides": {
            "distillation.enabled": True,
            "data.train_fraction": 0.25,
        }
    },
    "low_data_50pct": {
        "description": "H2: KD in low-data regime (50%)",
        "overrides": {
            "distillation.enabled": True,
            "data.train_fraction": 0.50,
        }
    },
    # Ablation: remove each KD component one at a time
    "ablation_no_mask_kd": {
        "description": "Ablation: full KD minus mask KD loss",
        "overrides": {
            "distillation.enabled": True,
            "distillation.losses.mask_kd.enabled": False,
            "distillation.losses.feature.enabled": True,
            "distillation.losses.boundary.enabled": True,
        }
    },
    "ablation_no_feature": {
        "description": "Ablation: full KD minus feature distillation",
        "overrides": {
            "distillation.enabled": True,
            "distillation.losses.mask_kd.enabled": True,
            "distillation.losses.feature.enabled": False,
            "distillation.losses.boundary.enabled": True,
        }
    },
    "ablation_no_boundary": {
        "description": "Ablation: full KD minus boundary loss",
        "overrides": {
            "distillation.enabled": True,
            "distillation.losses.mask_kd.enabled": True,
            "distillation.losses.feature.enabled": True,
            "distillation.losses.boundary.enabled": False,
        }
    },
}


def run_experiment(exp_name: str, cfg_path: str = "configs/config.yaml"):
    exp = EXPERIMENTS[exp_name]
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Description: {exp['description']}")
    print(f"Overrides: {exp['overrides']}")
    print(f"{'='*60}\n")

    # Load base config and apply overrides
    cfg = load_config(cfg_path)
    cfg = override_config(cfg, exp["overrides"])
    cfg = override_config(cfg, {"project.name": "crack_distill", "project.experiment": exp_name})

    # Run training — pass overridden config directly
    trainer = CrackDistillTrainer(cfg_path, override_cfg=cfg)
    trainer.train()
    results = trainer.test()

    return {exp_name: results}


def main():
    parser = argparse.ArgumentParser(description="Run Crack-Distill experiments")
    parser.add_argument(
        "--exp",
        type=str,
        default="full_kd_centroid",
        choices=list(EXPERIMENTS.keys()) + ["all", "paper_main", "ablation"],
        help="Experiment to run"
    )
    parser.add_argument("--cfg", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    if args.exp == "all":
        exps = list(EXPERIMENTS.keys())
    elif args.exp == "paper_main":
        exps = ["baseline_finetune", "pseudo_labels", "full_kd_box", "full_kd_centroid"]
    elif args.exp == "ablation":
        exps = ["ablation_no_mask_kd", "ablation_no_feature", "ablation_no_boundary"]
    else:
        exps = [args.exp]

    all_results = {}
    for exp_name in exps:
        results = run_experiment(exp_name, args.cfg)
        all_results.update(results)
        print(f"\n✓ {exp_name} done: {results}\n")

    print("\n" + "="*60)
    print("ALL RESULTS SUMMARY")
    print("="*60)
    for exp_name, metrics in all_results.items():
        print(f"\n{exp_name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()