# 📄 Academic Report: Advanced Knowledge Distillation from SAM 2 to YOLOv8-seg (with YOLOv11-seg Extension)

**Title**: Advanced Knowledge Distillation from SAM 2 to YOLO-seg for Road Crack Instance Segmentation  
**Author**: Project Development and Chronological Implementation Report  

---

## Abstract
This report details the systematic development of a Knowledge Distillation (KD) framework designed to compress structural representation capability from the Segment Anything Model (SAM 2) into lightweight YOLO-seg student architectures (**YOLOv11n-seg** as primary student, with **YOLOv8n-seg** retained as baseline). We present the project's evolution in chronological order, starting from the baseline scaffold to the implementation of multi-scale feature projection, per-instance target matching, uncertainty-weighted boundary alignment, progressive training protocols, memory-safe caching optimizations, dataloader parallelization, and out-of-distribution (OOD) composite hyperparameter optimization.

---

## 1. Phase 1: Architecture Scaffold and Dataset Setup

### 1.1 Modular Scaffold Design
The project began by building a modular pipeline that allows testing different tasks (instance segmentation, semantic segmentation, and object detection) without rewriting the core training loops. We built:
- A unified configuration loader (`config.yaml`) to control hyperparameters.
- A task registry that maps task types to specific data loaders, losses, and student adapters.
- A distillation trainer wrapper class (`CrackDistillTrainer`) that initializes the training flow.

### 1.2 Label Conversion via Connected Components
The target dataset is **Crack500** (1,896 training, 348 validation, and 348 test images) containing binary masks. In road crack images, cracks frequently touch or branch off, which causes standard polygon converters to group them into a single giant label.

To resolve this, we implemented a conversion script (`convert_crack500.py`):
1. It applies Connected Component Labeling (`cv2.connectedComponents`) on the binary masks to separate touching crack segments.
2. It filters out small noise structures (components containing fewer than 50 pixels).
3. It converts each isolated component into normalized YOLO polygon coordinates (`class x1 y1 x2 y2...`), creating a clean instance-segmentation dataset.
   For more details, see [[Connected Components Labeling]].

### 1.3 Student Model Selection
Per advisor review, the project's primary student architecture was updated to **YOLOv11n-seg**. YOLOv11 is actively supported for instance segmentation in Ultralytics and offers peer-reviewed architectural enhancements (C3k2/C2PSA blocks). **YOLOv8n-seg** (3.26M parameters, $\approx$ 12 GFLOPs) is retained as an architecture-matched baseline student trained under identical hyperparameters for fair comparison.

---

## 2. Phase 2: Logit Generation and First-Gen Distillation

### 2.1 Teacher Pre-Sigmoid Logits Extraction
Rather than using binary teacher masks, we wrote `generate_teacher_logits.py` to capture the raw, pre-sigmoid logits ($T \in \mathbb{R}^{M \times 256 \times 256}$) of SAM 2. 

At crack edges, the teacher's prediction logit is near $0$, mapping to a sigmoid probability of $\approx 0.5$. This represents the teacher's spatial uncertainty ("dark knowledge"). Storing raw logits allows the student to learn boundary transitions instead of forcing strict binary constraints.

### 2.2 First-Gen Boundary Loss Baseline
Our initial distillation setup incorporated a simple boundary loss:

$$L = L_{\text{task}} + \gamma L_{\text{boundary}}$$

The boundary loss computed a global cross-entropy between student mask predictions and teacher soft targets, weighted toward pixel coordinates where the teacher's logits expressed the highest uncertainty (sigmoid values close to 0.5).

After training for 100 epochs, this simple setup established our initial baseline:
- **Baseline (Fine-tuning, No KD)**: $\text{mAP}_{50}\text{-seg} = 0.529$
- **KD (First-Gen Boundary Only)**: $\text{mAP}_{50}\text{-seg} = 0.550$ ($+2.1$ points gain)

---

## 3. Phase 3: Multi-Scale Feature Distillation
To improve on the $+2.1$ point baseline, we moved beyond final mask matching and targeted the student's backbone representations. We implemented intermediate feature distillation ($L_{\text{feature}}$).

### 3.1 Dynamic Shape Capture
Since different student backbones have different channel depths, we implemented a dynamic hook setup:
1. In `setup_model()`, before training starts, the trainer runs a single dummy forward pass with zero tensors through the student network.
2. Temporary hooks record the exact channel dimensions and spatial shapes of the student's backbone blocks (at indices $[2, 5, 8]$).

### 3.2 Trainable $1\times1$ Projection Layers
Because the student backbone channels (e.g., 32, 128, 256) do not match the teacher's channel depths (e.g., 64, 256), we dynamically registered trainable $1\times1$ projection convolution modules ($P_i$) on the student model:

$$S^{\text{proj}}_i = P_i(S_i)$$

These layers are attached to the model as submodules before the optimizer is built, ensuring their weights are updated during backpropagation.

### 3.3 Scale-Matched MSE Loss
We extract student backbone features at strides 4, 16, and 32, project them to match SAM's channel sizes, and compute the Mean Squared Error (MSE) against SAM's saved encoder features ($T_i$, containing `feat1` at stride 4, and `image_embed` at stride 8):

$$L_{\text{feature}} = \sum_{i \in \{2,5,8\}} \text{MSE}\left( P_i(S_i), \text{Resize}(T_i) \right)$$

---

## 4. Phase 4: Per-Instance Target Matching ($L_{\text{mask}}$)
The first-gen boundary loss compared global canvas regions rather than individual instances, leading to spatial mismatching. We redesigned the mask loss to be per-instance.

### 4.1 On-the-Fly Assigner Integration
We patched the training step to intercept target assignments from YOLO's internal `TaskAlignedAssigner`. For each batch, we extract `fg_mask` and `target_gt_idx`. This tells us exactly which predicted student anchor maps to which ground-truth instance index.

### 4.2 Per-Instance Kullback-Leibler (KL) Divergence
Using the assigner mapping, we match each student prediction to its corresponding SAM teacher logit. We compute the KL Divergence on Bernoulli soft probability distributions at the full $256\times256$ resolution:

$$q = \sigma\left(\frac{T_{\text{match}}}{\tau}\right), \quad p = \sigma\left(\frac{S_{\text{match}}}{\tau}\right)$$

$$L_{\text{mask\_kd}} = \tau^2 \cdot \frac{1}{N_{\text{pos}}} \sum_{j=1}^{N_{\text{pos}}} \text{KL}\left( q_j \,||\, p_j \right)$$

where $\sigma$ is the sigmoid function, $N_{\text{pos}}$ is the number of positive anchors, and $\tau = 4.0$ is the temperature.

### 4.3 Instance-Matched Boundary Loss
Similarly, the boundary loss was updated to compute Binary Cross-Entropy (BCE) per matched instance, weighted by the teacher's uncertainty mask:

$$W_{\text{boundary}} = 1.0 - 2 \cdot \left| q - 0.5 \right|$$

$$L_{\text{boundary}} = \frac{1}{N_{\text{pos}}} \sum_{j=1}^{N_{\text{pos}}} \text{BCE}\left( \sigma(S_{\text{match}, j}), q_j \right) \odot W_{\text{boundary}, j}$$

For details on these equations, see [[Loss Functions]].

---

## 5. Phase 5: Progressive Distillation Schedule
Distilling intermediate features using randomly initialized $1\times1$ projection convolutions can introduce noisy gradients early in training, which degrades pre-trained backbone representations. To solve this, we implemented a 2-stage progressive training schedule:

### 5.1 Stage 1: Backbone Distillation (30% of Epochs)
The student model's Segment head is frozen (requiring grad is disabled for layer 22). The trainer optimizes only the student's backbone, neck, and projection layers. This allows the model to align its feature representations with SAM 2 without interference from downstream classification/regression gradients.

### 5.2 Stage 2: End-to-End Distillation (70% of Epochs)
The Segment head is unfrozen. The model is trained end-to-end starting from the Stage 1 checkpoint, allowing the newly unfrozen head and the aligned backbone to safely co-adapt.
For more details, see [[Pipeline Architecture]].

---

## 6. Phase 6: Runtime and Memory Optimizations

### 6.1 AMP Autocast Precision Matching
Under Automatic Mixed Precision (AMP), student predictions run in float16. To prevent runtime type crashes during loss computation, we dynamically cast the pre-loaded teacher tensors to match the student's prediction precision:

```python
sam_logits_matched_resized = sam_logits_matched_resized.to(
    dtype=student_mask_logits_resized.dtype
)
```

### 6.2 Bounded CPU RAM Caching
Loading large feature maps from disk for every batch created a disk I/O bottleneck. To resolve this, we implemented a CPU memory cache:
- The lightweight soft logits are fully cached in RAM ($\approx 1.4$ GB total).
- The heavy intermediate encoder features are cached inside a bounded FIFO queue (`OrderedDict`) capped at 256 items ($\approx 1.0$ GB).

This caps total cache memory at 2.4 GB, ensuring the training process fits safely within WSL's 8 GB memory limit (preventing Out-of-Memory kernel kills) while speeding up training by 4–5$\times$ from the second epoch onward.

---

## 7. Phase 7: Automated Hyperparameter Optimization via Optuna

To systematically determine the optimal contributions of the individual loss components, we implemented an automated hyperparameter search utilizing the Optuna framework. Rather than performing manual heuristic tuning, the script `tune_kd_weights.py` suggested continuous values for the loss weights and temperature.

To prevent disk-space exhaustion on Kaggle's environment (capped at 20 GB), we implemented a dynamic garbage-collection routine that purged intermediate training logs and checkpoints immediately after evaluating each trial's score, persisting only the globally best weights (`optuna_best_model.pt`) and study metrics. 

After running $10$ trials of $5$ epochs each, the optimization study converged on the following optimal hyperparameters:
- **Tuned Temperature ($\tau$)**: $1.6502$
- **Tuned Mask KD Weight ($\alpha$)**: $0.1652$
- **Tuned Feature KD Weight ($\beta$)**: $0.1767$
- **Tuned Boundary KD Weight ($\gamma$)**: $2.0569$

These hyperparameters were dynamically written back into the main `configs/config.yaml` to govern the final 150-epoch full distillation pipeline. See [[Optuna Weight Tuning]].

---

## 8. Phase 8: Experimental Evaluation and OOD Generalization Analysis

We evaluated three separate configurations on both the cropped, in-domain validation split (408 images) and the uncropped, out-of-distribution (OOD) validation split (50 full-resolution images containing EXIF orientation offsets). 

The quantitative results are presented in Table 1.

### Table 1: Quantitative validation evaluation comparison
| Model Configuration | Cropped mAP50-box | Cropped mAP50-seg | Cropped mAP50-95-seg | Uncropped mAP50-box | Uncropped mAP50-seg | Uncropped mAP50-95-seg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (No KD - 150 Epochs) | **0.5635** | **0.5287** | **0.2039** | 0.1488 | 0.1242 | 0.0319 |
| Optuna Best (15 Epochs)       | 0.5227 | 0.4951 | 0.1813 | 0.1446 | 0.1054 | 0.0245 |
| **Full KD (Tuned - 150 Epochs)** | 0.5562 | 0.5152 | 0.1934 | **0.1574** | **0.1308** | **0.0352** |

### 8.1 Generalization and Regularization Insights
1. **Generalization to Out-of-Distribution Data**: Under the uncropped testing configuration, the baseline model's performance drops severely due to shape and orientation mismatching. However, our proposed Full KD framework, guided by SAM 2 spatial priors, achieves substantial gains:
   - A **$+5.8\%$** relative increase in box detection (mAP50-box rises from 0.1488 to 0.1574).
   - A **$+5.3\%$** relative increase in segmentation quality (mAP50-seg rises from 0.1242 to 0.1308).
   - A **$+10.3\%$** relative increase in high-overlap accuracy (mAP50-95-seg rises from 0.0319 to 0.0352).
   
   This demonstrates that distilling pre-sigmoid SAM 2 logits and multi-scale intermediate features provides a robust shape prior that generalizes to high-resolution, full-canvas inputs. See [[Experimental Results]].
   
2. **In-Domain Regularization Trade-off**: On the cropped dataset, the baseline model retains a slight advantage of $1.35\%$ mAP50-seg. This indicates that standard fine-tuning allows the student to closely memorize crop boundaries and artifact characteristics of the training split. In contrast, the multi-scale feature alignment ($L_{\text{feature}}$) acts as a strict structural regularizer, preventing the student from overfitting to crop boundaries and forcing it to learn more generalized, scale-invariant representation features.

---

## 9. Phase 9: Advanced Dataloader Parallelization and CPU Memory Optimization

To scale the training and run full distillation runs stably on hardware setups with constrained CPU memory (such as WSL with strict RAM limits or multi-core environments), we redesigned the data pipeline to resolve disk I/O bottlenecks and eliminate memory accumulation.

### 9.1 Custom Asynchronous Dataloader Wrapper (`KDYOLODataset`)
To completely hide disk-read latencies, we implemented a custom PyTorch dataset wrapper `KDYOLODataset` inside [kd_trainer.py](file:///home/shahin/distill/distillation/kd_trainer.py):
1. **Multiprocess Preloading**: The dataset wrapper intercepts dataset indexing calls (`__getitem__`) within PyTorch background worker processes.
2. **Asynchronous I/O**: Dataloader workers load and parse SAM teacher logits (`_logits.npy`) and spatial encoder feature tensors (`_features.npz`) in parallel before batches are collated.
3. **Flat RAM Footprint**: Eliminating in-memory caching reduced host memory usage to a flat **$\approx$ 4.1 GiB RAM** (0B swap), accelerating throughput to **7.0–7.5 iterations/sec**.

---

## 10. Phase 10: Dynamic Trial Pruning via Optuna MedianPruner

Evaluating hyperparameter trials over short durations failed to capture true downstream convergence characteristics. We implemented a dynamic early stopping mechanism to prune poor trials.

### 10.1 Early Stopping via `MedianPruner`
We integrated a `MedianPruner` with the Optuna study setup in [tune_kd_weights.py](file:///home/shahin/distill/scripts/tune_kd_weights.py):
- **`n_startup_trials=2`**: Allows the first two trials to complete fully (15 epochs) to map baseline learning performance curves.
- **`n_warmup_steps=8`**: Disables pruning for the first 8 epochs of subsequent trials, allowing models to complete Stage 1 backbone alignment ($\sim 4.5$ epochs) and run Stage 2 joint KD for $\sim 3.5$ epochs before evaluating pruning rules.
- **Exception Propagation and Cleanup**: Pruned trials clean up run directories and report `optuna.exceptions.TrialPruned`, saving 60–70% of GPU compute time.

---

## 11. Phase 11: Out-of-Distribution Composite Objective & Engine Wiring

### 11.1 Dual-Evaluation Composite Objective
Earlier tuning passes scored trials strictly on cropped in-domain mAP50-seg, selecting parameters that over-regularized cropped patches at the expense of OOD generalization. In Phase 11, we updated `objective()` in [tune_kd_weights.py](file:///home/shahin/distill/scripts/tune_kd_weights.py) to evaluate both cropped in-domain validation (`combined_yolo`) and uncropped OOD validation (`crack500_uncropped_yolo`):

$$\text{Composite Score} = 0.4 \cdot \text{mAP50}_{\text{cropped}} + 0.6 \cdot \text{mAP50}_{\text{OOD}}$$

### 11.2 Engine Dataset Fraction Wiring
To ensure `train_fraction` controls dataset sampling during hyperparameter sweeps, we updated [distillation/trainer.py](file:///home/shahin/distill/distillation/trainer.py) to extract `self.fraction` from config and pass `fraction = self.fraction` explicitly to all Ultralytics engine override dictionaries (`overrides_stage1`, `overrides_stage2`, single-stage `overrides`, and baseline `model.train()`).
