# Implementation Plan — Fix Optuna Weight Tuning & Hyperparameters

This plan addresses the degradation in distillation performance. We analyze why the previous Optuna-tuned weights led to worse results and detail the steps to fix the tuning pipeline on Kaggle.

---

## Technical Analysis: Why did the Optuna-tuned model perform worse?

Our research into the experimental logs and notebooks revealed two major issues:

### 1. The Path Override Bug in the Optuna Notebook
In the runned Kaggle notebook `distill-optuna-runned-results.ipynb`, the overrides dictionary inside the `tune_kd_weights.py` script was missing the key `"teacher.logits_dir": "data/teacher_logits_centroid/"`. 
- As a result, the training fallback searched the default `data/teacher_logits/` folder, which was **completely empty**.
- Since there were `0` logit files, `kd_trainer.py` silently skipped computing soft-target mask losses, boundary losses, and feature losses.
- Consequently, all 10 trials trained identical baseline student models without any knowledge distillation, resulting in the exact same score (`0.3759`) across every trial. Optuna simply returned the random weights generated in Trial 0.

### 2. Short-Run Fitting Bias (5-Epoch Limitation)
Tuning hyperparameters over only 5 epochs creates a severe bias:
- Knowledge distillation losses (KL divergence on soft masks and MSE on features) act as strong regularizers. In the early epochs (warmup + first few steps), regularization slows down fitting on the training set, causing lower early validation scores.
- Standard supervised task loss allows the student to fit the ground truth quickly, showing higher early validation scores.
- By evaluating at epoch 5, Optuna was biased towards **minimizing** the distillation weights (e.g., tuning them down to $\alpha = 0.16$ and $\beta = 0.17$), effectively turning KD off.
- When these low weights were scaled up to 100 epochs, the student trained with almost no regularization, leading to worse performance and poorer generalization than the manual run where larger weights ($\alpha = 1.0$, $\beta = 1.5$, $\gamma = 2.0$) were enforced.

---

## How Optuna Works & Our Configuration Strategy

Optuna uses a combination of a **Sampler** (to decide which parameter values to try next) and a **Pruner** (to stop bad trials early). We will configure them as follows:

### 1. Tree-structured Parzen Estimator (TPE) Sampler
By default, Optuna uses the **TPESampler**. This is a Bayesian optimization algorithm:
- Instead of searching randomly, TPE models the probability distribution of hyperparameters in two groups: high-performing configurations (below a quantile threshold) and low-performing ones.
- It tries to choose hyperparameter values that maximize the likelihood of being in the high-performing group.
- To ensure TPE searches in a literature-aligned, active range, we will restrict the bounds of the search space so it cannot choose near-zero weights.

### 2. Median Pruner (Early Stopping)
To handle the Kaggle session limit and GPU time constraints, we will configure a `MedianPruner(n_startup_trials=2, n_warmup_steps=5)`:
- **`n_startup_trials=2`**: The first 2 trials are allowed to run to completion (full 15 epochs) to build a baseline historical curve of validation scores.
- **`n_warmup_steps=5`**: For subsequent trials, pruning is disabled for the first 5 epochs. This lets the student model get past the 3-epoch learning rate warmup and stabilize its training path.
- **Early Stopping Rule**: After epoch 5, at the end of each epoch, the trial reports its validation score to Optuna. If the current validation score is lower than the median score of all previous trials at that exact same epoch, Optuna halts the trial early (`TrialState.PRUNED`).
- This allows us to search with **15 epochs per trial** (much more representative of 100-epoch convergence) while saving 60-70% of GPU compute time by aborting bad trials early.

---

## Proposed Changes

To fix these issues, we will implement the following changes:

1. **Restrict Search Space to Active KD Ranges**: Adjust hyperparameter bounds to prevent Optuna from choosing near-zero weights, enforcing substantial distillation signal as recommended in literature.
2. **Increase Epochs and Add Optuna Pruning**: Increase epochs per trial to **15 epochs** while using `optuna.pruners.MedianPruner` to stop poorly performing trials early.
3. **Synchronize Notebooks**: Ensure all Kaggle and Colab notebooks contain the correct, patched tuning scripts.

---

### Component 1: Distillation Trainer (`distillation/trainer.py`)

We will add a callback to report metrics to Optuna and trigger pruning.

#### [MODIFY] [trainer.py](file:///home/shahin/distill/distillation/trainer.py)
- Define a global or module-level callback function `optuna_callback(trainer)` that retrieves `'metrics/mAP50(M)'` from Ultralytics' epoch metrics.
- Pass the current `trial` object and epoch offset to `KDSegmentationTrainer` in both Stage 1 and Stage 2 of progressive training.
- Register `on_fit_epoch_end` callback to report the validation score and raise `optuna.exceptions.TrialPruned` if pruning conditions are met.

---

### Component 2: Optuna Script (`scripts/tune_kd_weights.py`)

We will update the search bounds, add a pruner to the study, and handle pruning exceptions.

#### [MODIFY] [tune_kd_weights.py](file:///home/shahin/distill/scripts/tune_kd_weights.py)
- **Update Search Bounds**:
  - `temperature` ($\tau$): `[2.0, 4.0]`
  - `mask_kd` weight ($\alpha$): `[0.5, 2.5]` (enforce active soft logits distillation)
  - `feature` weight ($\beta$): `[0.5, 2.0]` (enforce intermediate feature distillation)
  - `boundary` weight ($\gamma$): `[0.5, 3.0]` (enforce boundary alignment)
- **Add Pruner**: Instantiate `optuna.create_study` with `pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5)` to prune bad trials after 5 epochs.
- **Support Pruning in Objective**: Catch `optuna.exceptions.TrialPruned` in the training loop, perform directory cleanup, and propagate the exception so Optuna registers the trial as pruned.
- **Pass Trial Object**: Pass the `trial` object to `CrackDistillTrainer` so it can be accessed by the callback.

---

### Component 3: Notebooks (`run_on_kaggle_optuna.ipynb`)

#### [MODIFY] [run_on_kaggle_optuna.ipynb](file:///home/shahin/distill/run_on_kaggle_optuna.ipynb)
- Update the embedded `%%writefile scripts/tune_kd_weights.py` cell to match the updated tuning script.
- Change the run command to support more epochs and fewer trials (e.g. `--trials 8 --epochs 15`).

---

## Verification Plan

### Automated Tests
We will verify that:
1. The config load and overrides work properly.
2. A local dry run of the tuning script for 1 trial of 3 epochs completes without syntax or callback errors.
3. The validation metrics are successfully extracted and reported to Optuna.

```bash
# Verify config loader and trainer instantiation
/home/shahin/miniconda3/envs/cv-analysis/bin/python -c "from utils.config_loader import load_config; print(load_config('configs/config.yaml'))"

# Dry run 1 trial of 3 epochs to test callback/pruning logic locally
/home/shahin/miniconda3/envs/cv-analysis/bin/python scripts/tune_kd_weights.py --trials 1 --epochs 3
```

### Manual Verification
- Confirm that the best parameters saved in `runs/best_optuna_params.json` contain weights inside the new active ranges.
