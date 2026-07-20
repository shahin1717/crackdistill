#!/usr/bin/env python3
"""
Optuna KD Weight Tuning Script — Crack-Distill
=============================================
Tunes KD loss weights (mask_kd, feature, boundary) and temperature
to maximize student validation performance (mAP50-seg).
Cleans up trial run directories to prevent disk exhaustion.
"""

import argparse
import sys
import os
import shutil
import json
from pathlib import Path
import torch

# Add project root to path and environment for subprocesses
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.environ["PYTHONPATH"] = project_root + os.pathsep + os.environ.get("PYTHONPATH", "")

from utils.config_loader import load_config, override_config
from distillation.trainer import CrackDistillTrainer

try:
    import optuna
except ImportError:
    print("Optuna is not installed. Please run: pip install optuna")
    sys.exit(1)


def objective(trial, args, base_cfg) -> float:
    # 1. Suggest hyperparameters
    temp = trial.suggest_float("temperature", 2.0, 4.0)
    w_mask = trial.suggest_float("mask_kd", 0.5, 2.5)
    w_feat = trial.suggest_float("feature", 0.5, 2.0)
    w_bound = trial.suggest_float("boundary", 0.5, 3.0)

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Suggested parameters:")
    print(f"  temperature: {temp:.4f}")
    print(f"  mask_kd weight: {w_mask:.4f}")
    print(f"  feature weight: {w_feat:.4f}")
    print(f"  boundary weight: {w_bound:.4f}")

    # 2. Setup overrides
    overrides = {
        "distillation.enabled": True,
        "distillation.temperature": temp,
        "distillation.losses.mask_kd.enabled": True,
        "distillation.losses.mask_kd.weight": w_mask,
        "distillation.losses.feature.enabled": True,
        "distillation.losses.feature.weight": w_feat,
        "distillation.losses.boundary.enabled": True,
        "distillation.losses.boundary.weight": w_bound,
        "train.epochs": args.epochs,
        "teacher.logits_dir": "data/teacher_logits_centroid/",
        "data.train_fraction": args.train_fraction,
    }
    
    # Run name unique per trial
    exp_name = f"optuna_trial_{trial.number}"
    overrides.update({
        "project.name": "crack_distill",
        "project.experiment": exp_name
    })

    # Apply overrides
    cfg = override_config(base_cfg, overrides)

    # Instantiate trainer and attach trial
    trainer = CrackDistillTrainer(args.cfg, override_cfg=cfg)
    trainer.optuna_trial = trial
    
    score = 0.0
    try:
        # Run training
        trainer.train()
        
        # Evaluate model on validation
        results = trainer.test()
        score = results.get("mAP50-seg", 0.0)
        
        # Check if this trial is the best so far
        is_best = False
        try:
            best_trial = trial.study.best_trial
            if score > best_trial.value:
                is_best = True
        except ValueError:
            # First trial completed
            is_best = True
            
        if is_best:
            # Save best parameters to json
            best_params_path = Path("runs/best_optuna_params.json")
            best_params_path.parent.mkdir(parents=True, exist_ok=True)
            best_info = {
                "trial_number": trial.number,
                "score": score,
                "parameters": {
                    "temperature": temp,
                    "mask_kd": w_mask,
                    "feature": w_feat,
                    "boundary": w_bound
                }
            }
            with open(best_params_path, "w") as f:
                json.dump(best_info, f, indent=2)
                
            # Copy best model weight
            best_model_src = trainer.best_pt
            if best_model_src.exists():
                best_model_dst = Path("runs/optuna_best_model.pt")
                shutil.copy2(best_model_src, best_model_dst)
                print(f"[Optuna] New best trial {trial.number}! Score (mAP50-seg): {score:.4f}. Saved best weights to {best_model_dst}")
                
    except optuna.exceptions.TrialPruned:
        print(f"[Optuna] Trial {trial.number} was pruned early.")
        raise
    except Exception as e:
        print(f"[Optuna] Trial {trial.number} failed with exception: {e}")
        import traceback
        traceback.print_exc()
        score = 0.0
    finally:
        # 3. Clean up runs to save disk space
        print(f"[Optuna] Cleaning up Trial {trial.number} run directories...")
        run_dirs_to_clean = [
            trainer.run_dir,
            trainer.run_dir.parent / f"{trainer.run_dir.name}_stage1",
            trainer.run_dir.parent / f"{trainer.run_dir.name}_stage2"
        ]
        for d in run_dirs_to_clean:
            # Delete direct directories
            if d.exists():
                try:
                    shutil.rmtree(d)
                except Exception as err:
                    print(f"[Optuna] Warning: Failed to delete {d}: {err}")
            
            # Clean any wildcard matches (YOLO sometimes appends suffixes like 2, 3...)
            for p in trainer.run_dir.parent.glob(f"{d.name}*"):
                if p.exists() and p.is_dir():
                    try:
                        shutil.rmtree(p)
                    except Exception:
                        pass
        
        # Clear CUDA memory cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return score


def main():
    parser = argparse.ArgumentParser(description="Tune KD weights using Optuna")
    parser.add_argument("--cfg", type=str, default="configs/config.yaml")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per trial")
    parser.add_argument("--train-fraction", type=float, default=0.20, help="Fraction of training data to use for tuning")
    parser.add_argument("--study-name", type=str, default="kd_weight_tuning")
    parser.add_argument("--storage", type=str, default=None, help="Database URL for Optuna storage (optional)")
    args = parser.parse_args()

    # Load base config
    base_cfg = load_config(args.cfg)

    # Set up Optuna logging verbosity
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5)
    )

    print(f"Starting study '{args.study_name}' with {args.trials} trials, each training for {args.epochs} epochs.")
    
    study.optimize(lambda trial: objective(trial, args, base_cfg), n_trials=args.trials)

    print("\n" + "="*60)
    print("OPTUNA TUNING COMPLETED")
    print("="*60)
    try:
        print(f"Best Trial: #{study.best_trial.number}")
        print(f"Best Score (mAP50-seg): {study.best_value:.4f}")
        print("Best Parameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v:.4f}")
    except ValueError:
        print("No trials completed successfully.")
    print("="*60)


if __name__ == "__main__":
    main()
