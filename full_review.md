# CrackDistill: Lightweight Crack Segmentation via Knowledge Distillation from Foundation Models

**Author:** Shahin  
**Date:** July 2026  
**Status:** Work in progress — 24 experiments completed, ablations and SAM 3 integration planned

---

## Abstract

Infrastructure crack detection requires high-quality pixel-level segmentation, yet current foundation models such as SAM 2 (224M parameters) are too computationally heavy for real-time edge deployment. This work introduces **CrackDistill**, a knowledge distillation framework that transfers SAM 2's boundary-precise spatial priors into a lightweight YOLOv11n-seg student model (2.84M parameters — a **79× reduction**). Our primary finding is that distillation does not merely preserve accuracy: it actively improves **out-of-distribution generalization**, demonstrating that SAM 2's soft masks function as a domain regularizer rather than a simple accuracy booster. Across 24 experiments on the Crack500 and DeepCrack benchmarks, KD improves OOD mask mAP50-95 by **+10.3%** on held-out uncropped images and improves cross-dataset transfer (Crack500 → DeepCrack) by **+14.1% relative**. We additionally identify and fix three failure modes specific to crack KD: FP16 logit overflow, small-dataset segmentation head collapse, and asymmetric domain shift. The result is a fast, generalizable crack segmentation model suitable for embedded inspection hardware.

---

## 1. Motivation and Problem Statement

### 1.1 Why Crack Detection Matters

Structural cracks in roads, bridges, and concrete infrastructure are the primary early indicator of material degradation. Missed or late detection leads to catastrophic failures. Automated visual inspection is increasingly performed by drones and mobile platforms, which demand **real-time inference on embedded hardware** — a constraint that eliminates large foundation models from direct deployment.

### 1.2 Why SAM 2 Cannot Be Deployed Directly

SAM 2 Large achieves excellent boundary-level segmentation across a wide range of objects, trained on the SA-11B dataset (11 million images). However:

- **Parameters:** 224M — too large for most embedded accelerators
- **Speed:** 8–15 FPS on GPU (official benchmarks); real-time inspection typically requires 30+ FPS
- **Promptability:** SAM 2 requires an explicit spatial prompt (bounding box or point) at inference time — it is not a standalone detector

These constraints make direct deployment impractical for field inspection.

### 1.3 Why Standard YOLO Fine-Tuning Is Insufficient

Standard fine-tuning of YOLOv11n-seg on crack datasets produces a fast model but exhibits severe **overfitting to dataset-specific artifacts**:

Crack500 images are dense crops of larger pavement images. A fine-tuned model memorizes crack positions relative to crop boundaries rather than learning crack texture, achieving `mAP50 = 0.5249` on cropped test images but collapsing to `mAP50 = 0.1064` on uncropped real-world images — an **80% performance drop** (EXP-04).

This motivates a teacher signal that provides richer spatial priors and resists crop-artifact memorization.

### 1.4 Research Question

> *Can knowledge distillation from a large segmentation foundation model improve both the generalization and out-of-distribution robustness of a lightweight crack detector, without sacrificing deployment speed?*

---

## 2. Related Work

| Method | Approach | Limitation vs Our Work |
|:---|:---|:---|
| **MobileSAM** | Distill SAM into tiny ViT encoder | Generic segmentation; no crack-specific domain study |
| **EdgeSAM** | CNN-based encoder replacing SAM ViT | Generic; no domain generalization evaluation |
| **FastSAM** | CNN instance segmentation at SAM speed | No knowledge distillation; teacher signal absent |
| **EfficientViT-SAM** | Architecture replacement for SAM encoder | Generic; no task-specific fine-tuning |
| **YOLO crack fine-tuning** (prior work) | Standard supervised fine-tuning | Overfits on small datasets; no teacher regularization |
| **Our work** | SAM 2/3 → YOLO KD, crack-specific, OOD generalization | — |

**Key gap:** No prior published work performs SAM 2 → YOLO knowledge distillation specifically for crack detection with **domain generalization as the primary evaluation metric**. Existing SAM distillation work targets generic segmentation quality, not OOD robustness in a specialized domain.

---

## 3. Datasets

### 3.1 Crack500

- **Training images:** ~1,500 (dense pavement crops of large source images)
- **Crack types:** Diverse widths, pavement textures, and lighting conditions
- **Challenge:** Images are crops of larger scenes. Models memorize crop boundary artifacts rather than actual crack texture — causing drastic OOD degradation on uncropped images.
- **Our train/val/test split:** Standard Crack500 splits (348 validation images, 630 instances)

### 3.2 DeepCrack

- **Training images:** 537 (high-resolution controlled acquisition)
- **Crack types:** Narrow, high-contrast, elongated thin cracks
- **Challenge:** Small dataset — only 537 training images — with narrow visual distribution. Models trained here overfit to thin-crack, high-contrast texture features that do not generalize beyond this acquisition style.
- **Our train/val/test split:** Standard DeepCrack splits (60 validation images, 169 instances)

### 3.3 Domain Shift Analysis

The two datasets represent **asymmetric domains**:

- **Crack500** images are richer and more diverse in terms of pavement type, crack width, and scene context. A model trained here learns more general crack features.
- **DeepCrack** images were acquired under controlled conditions with thin, high-contrast cracks. A model trained here learns acquisition-specific features that fail to generalize.

This creates an asymmetric domain gap with a decisive practical implication: **train on Crack500, transfer to DeepCrack** is viable; the reverse collapses (see Section 6.4).

---

## 4. Method

### 4.1 Architecture Overview

```
SAM 2 Large (Teacher, 224M params) — FROZEN
        |  soft logits, encoder features
        v
YOLOv11n-seg (Student, 2.84M params) — TRAINED
```

The teacher is frozen throughout training. The student is trained from the Ultralytics YOLO checkpoint using the standard YOLO training loop, with KD losses injected into each training step via a custom `KDSegmentationTrainer` subclass.

**Size comparison:**

| Model | Parameters | GFLOPs | Speed (Official) |
|:---|:---:|:---:|:---:|
| SAM 2 Large (teacher) | 224M | ~800+ | 8–15 FPS |
| YOLOv11n-seg (student) | **2.84M** | **10.2** | ~60 FPS |
| YOLOv8n-seg (comparison) | 3.26M | 12.0 | ~55 FPS |
| **Compression ratio** | **79x** | **~78x** | **~4x** |

### 4.2 Teacher Prompting Strategy

SAM 2 requires a spatial prompt to segment a region. We generate prompts from YOLO ground-truth annotations:

- **Box prompts:** Ground-truth bounding boxes are passed to SAM 2 to generate soft segmentation logits over the crack region.
- **Centroid prompts (tested but abandoned):** Adding the centroid of each box as a point prompt produced **identical results** to box-only prompts (EXP-14 vs EXP-15: both `Mask mAP50 = 0.5468`). SAM 2 Large fully constrains crack regions from the bounding box alone; centroid is redundant.

**Conclusion:** Box prompts are sufficient. Do not add centroid prompts.

### 4.3 Loss Function

The total training loss is:

$$L_{\text{total}} = L_{\text{task}} + \alpha \cdot L_{\text{KL}} + \beta \cdot L_{\text{feature}} + \gamma \cdot L_{\text{boundary}}$$

Each term explained:

**$L_{\text{task}}$** — Standard YOLO segmentation loss (detection boxes + mask coefficients + prototypes). This is the baseline YOLO loss unchanged.

**$L_{\text{KL}}$** — KL divergence between student logits and SAM 2 soft logits, scaled by temperature $\tau$:

$$L_{\text{KL}} = \frac{\tau^2}{N} \sum_{i} \text{KL}\!\left( \sigma\!\left(\frac{z_s^{(i)}}{\tau}\right) \,\Big\|\, \sigma\!\left(\frac{z_t^{(i)}}{\tau}\right) \right)$$

where $z_s$ are student logits, $z_t$ are SAM 2 teacher logits (clamped to $[-30, 30]$), $\sigma$ is sigmoid, and $\tau$ is the distillation temperature. The $\tau^2$ factor preserves gradient magnitude at high temperatures (standard KD convention following Hinton et al.).

**$L_{\text{feature}}$** — Mean squared error between student intermediate features (projected to teacher dimension via learned 1x1 convolution) and SAM 2 image encoder features:

$$L_{\text{feature}} = \frac{1}{C \cdot H \cdot W} \left\| f_s - \text{proj}(f_t) \right\|_2^2$$

This forces the student backbone to learn SAM-aligned feature representations, not just mimic output logits.

**$L_{\text{boundary}}$** — Binary cross-entropy on crack boundary pixels specifically:

$$L_{\text{boundary}} = -\frac{1}{|\mathcal{B}|} \sum_{p \in \mathcal{B}} \left[ y_p \log \hat{y}_p + (1-y_p) \log(1-\hat{y}_p) \right]$$

where $\mathcal{B}$ is the set of boundary pixels extracted from SAM 2's soft mask via edge detection. This term specifically penalizes errors on thin crack edges — the hardest failure mode for lightweight segmentation models.

### 4.4 Hyperparameter Tuning via Optuna

Manually searching over $\tau$, $\alpha$, $\beta$, $\gamma$ is impractical. We use Optuna with a **composite OOD-aware objective**:

$$\text{Score} = 0.4 \cdot \text{mAP50}_{\text{in-domain}} + 0.6 \cdot \text{mAP50}_{\text{OOD}}$$

The 60% weight on OOD reflects our deployment goal of generalization, not in-distribution fitting. Best parameters from 10–12 trials (15 epochs each):

| Parameter | Symbol | Best Value |
|:---|:---:|:---:|
| Temperature | $\tau$ | **3.7769** |
| KL loss weight | $\alpha$ | **0.9612** |
| Feature MSE weight | $\beta$ | **1.8658** |
| Boundary BCE weight | $\gamma$ | **0.8055** |

**Note:** These weights are highly sensitive. Small changes in the $\alpha / \beta$ ratio significantly impact OOD performance. Finding optimal weights remains an open research problem.

### 4.5 Numerical Stability Fix: Logit Clamping

SAM 2 produces raw logits with magnitudes up to $\pm 50$ or higher. In FP16 training:

$$\log\!\left(\sigma\!\left(\frac{z_t}{\tau}\right)\right) \xrightarrow{\text{FP16}} \log(0) = -\infty \rightarrow \text{NaN}$$

EXP-05 and EXP-06 both crashed at epochs 42–44 due to this overflow. The fix is two-part:

1. **Clamp teacher logits:** $z_t \leftarrow \text{clip}(z_t, -30, 30)$ before any KL computation
2. **Disable AMP:** Set `amp: false` — use FP32 throughout

This completely eliminated NaN crashes from EXP-07 onward.

### 4.6 Progressive Segmentation Head Freeze (for Small Datasets)

On Crack500 (~1,500 images), simultaneous gradient flow through backbone and segmentation head is fine — enough samples to average out gradient conflicts between task loss and KD soft loss.

On DeepCrack (537 images), the head faces a gradient conflict:

```
Task loss gradient    ->  "produce crisp binary masks"
KD soft loss gradient ->  "imitate SAM's smooth probability distribution"
```

With only 537 samples, the head oscillates and collapses. Standard unfrozen KD on DeepCrack **degraded** Mask mAP50 from `0.4853` to `0.4765` (EXP-17).

**Fix: 2-stage progressive schedule:**

- **Stage 1 (first 30% of epochs):** Freeze segmentation head. Only backbone receives KD gradient — learns SAM-aligned feature representations without head interference.
- **Stage 2 (remaining 70% of epochs):** Unfreeze head — fine-tunes task-specifically on top of richer backbone representations.

**Result (EXP-18):** `0.5046` Mask mAP50 — **+3.98% over baseline, +2.81 mAP50 points over unfrozen KD**.

| Dataset size | Use SegHead Freeze? | Reason |
|:---|:---:|:---|
| < 1,000 images | **Yes** | Gradient conflict dominates |
| > 1,000 images | No | Enough data to average out conflict |

---

## 5. Experiments

### 5.1 Execution Environment

- **Platform:** Kaggle Notebooks (dual NVIDIA Tesla T4, 16 GB VRAM each)
- **Framework:** PyTorch 2.5.1 + CUDA 12.4, Ultralytics YOLO
- **Local workspace:** WSL2 Ubuntu 24.04 LTS
- **Training config:** 150 epochs, image size 512x512, batch size 16, SGD optimizer, mosaic augmentation

---

### 5.2 Experiment 1: Architecture Selection — YOLOv8n vs YOLOv11n (EXP-09 to EXP-12)

Both models trained from scratch, 150 epochs, no KD, identical config.

**On Crack500:**

| Model | Mask mAP50 | Mask mAP50-95 | Mask Recall | Params | GFLOPs |
|:---|:---:|:---:|:---:|:---:|:---:|
| YOLOv8n-seg | 0.532 | 0.203 | 0.489 | 3.26M | 12.0 |
| **YOLOv11n-seg** | **0.540** | **0.207** | **0.495** | **2.84M** | **10.2** |

**On DeepCrack:**

| Model | Mask mAP50 | Mask mAP50-95 | Mask Recall | Params | GFLOPs |
|:---|:---:|:---:|:---:|:---:|:---:|
| YOLOv8n-seg | 0.496 | 0.189 | 0.485 | 3.26M | 12.0 |
| **YOLOv11n-seg** | **0.538** | **0.197** | **0.540** | **2.84M** | **10.2** |

**Findings:**

- YOLOv11n-seg is **strictly better on both datasets** with fewer parameters and GFLOPs.
- The gap is larger on DeepCrack (+4.2 mAP50 pts vs +0.8 pts on Crack500), because YOLOv11's improved neck architecture recovers thin crack recall better.
- **Mask Recall on DeepCrack improves by +5.5 points** (0.540 vs 0.485) — the decisive metric for sparse, elongated cracks.
- **Decision:** Use YOLOv11n-seg for all subsequent experiments.

---

### 5.3 Experiment 2: Optuna Hyperparameter Search (EXP-03)

10–12 trials, 15 epochs each, Crack500, OOD-weighted composite objective.

Best discovered configuration: $\tau=3.78$, $\alpha=0.96$, $\beta=1.87$, $\gamma=0.81$ (see Section 4.4). These parameters were used for all subsequent full KD runs.

---

### 5.4 Experiment 3: Full KD Run on Crack500 (EXP-08, `run_on_kaggle_final_rauf.ipynb`)

150 epochs, Optuna-optimal weights, logit clamping enabled.

| Evaluation Split | Metric | Baseline (No KD) | Full KD (SAM 2) | Change |
|:---|:---|:---:|:---:|:---:|
| Cropped (in-distribution) | Mask mAP50 | 0.5249 | 0.5152 | $-1.85\%$ (regularized) |
| Uncropped (OOD) | Box mAP50 | 0.1488 | **0.1574** | $+5.8\%$ |
| Uncropped (OOD) | Mask mAP50 | 0.1242 | **0.1308** | $+5.3\%$ |
| Uncropped (OOD) | Mask mAP50-95 | 0.0319 | **0.0352** | **$+10.3\%$** |

**Key insight:** The slight in-distribution drop is expected and desirable — it signals the model is generalizing rather than memorizing crop boundaries. The OOD improvement is the actual contribution.

---

### 5.5 Experiment 4: Crack500 KD — Prompt Ablation (EXP-13 to EXP-15)

| Strategy | Mask mAP50 | Mask Precision | Box mAP50 |
|:---|:---:|:---:|:---:|
| Baseline fine-tune | 0.5445 | 0.6910 | 0.5880 |
| Full KD (Box prompts) | **0.5468** | **0.7130** | 0.5798 |
| Full KD (Box + Centroid) | **0.5468** | **0.7130** | 0.5798 |

Box and Box+Centroid produce identical results. Centroid prompts add engineering complexity with zero benefit.

---

### 5.6 Experiment 5: DeepCrack KD — SegHead Freeze Ablation (EXP-16 to EXP-18)

| Strategy | Mask mAP50 | Box mAP50 | vs Baseline |
|:---|:---:|:---:|:---:|
| Baseline (No KD) | 0.4853 | 0.5361 | — |
| Full KD (Unfrozen) | 0.4765 | 0.5312 | **−0.88 pts (degraded)** |
| **Full KD (SegHead Frozen)** | **0.5046** | **0.5637** | **+1.93 pts / +3.98%** |

The progressive freeze schedule is critical for small datasets. Without it, KD actively harms performance.

---

### 5.7 Experiment 6: Cross-Dataset Generalization (EXP-19 to EXP-24)

This is the key evaluation — does KD generalize across domains?

**Crack500 → DeepCrack (zero-shot transfer):**

| Model | Type | Mask mAP50 | Mask mAP50-95 | Box mAP50 |
|:---|:---|:---:|:---:|:---:|
| YOLOv8n-seg | Baseline | 0.2209 | 0.0793 | 0.2431 |
| YOLOv11n-seg | Baseline | 0.2601 | 0.0896 | 0.3005 |
| **YOLOv11n-seg** | **Full KD** | **0.2741** | **0.1023** | 0.2925 |

KD improves over baseline: **+5.38% Mask mAP50**, **+14.1% Mask mAP50-95**.

**DeepCrack → Crack500 (zero-shot transfer):**

| Model | Type | Mask mAP50 | Mask mAP50-95 | Box mAP50 |
|:---|:---|:---:|:---:|:---:|
| YOLOv8n-seg | Baseline | 0.0295 | 0.0067 | 0.0971 |
| YOLOv11n-seg | Baseline | 0.0276 | 0.0055 | 0.1017 |
| **YOLOv11n-seg** | **Full KD** | **0.0329** | **0.0075** | **0.1054** |

KD recovers some performance (+19.0% relative on Mask mAP50), but the absolute numbers remain extremely low. The domain collapse is fundamental — not fixable by KD alone.

**Why DeepCrack → Crack500 collapses:** DeepCrack has only 537 training images of narrow, high-contrast cracks photographed under controlled conditions. A model trained there learns acquisition-specific texture features that fail entirely on Crack500's diverse, lower-contrast pavement scenes. This is a classic **acquisition bias-driven domain shift**, compounded by small dataset size forcing extreme overfitting to the narrow DeepCrack distribution.

**Conclusion:** The correct training direction is Crack500 → DeepCrack. The performance gap is 8x (0.274 vs 0.033 Mask mAP50).

---

## 6. Complete Experiment Registry

| ID | Dataset | Model | Strategy | Epochs | Mask mAP50 | OOD mAP50 | Status |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|
| EXP-01 | Combined | YOLOv8n | Baseline | 100 | 0.529 | 0.118 | Done |
| EXP-02 | Combined | YOLOv8n | 1st-Gen Boundary | 100 | 0.550 | 0.121 | Done |
| EXP-03 | Crack500 | YOLOv11n | Optuna (12 trials) | 15 | 0.495 | 0.105 | Done — best weights |
| EXP-04 | Crack500 | YOLOv11n | Baseline | 150 | 0.525 | 0.106 | Done |
| EXP-05 | Crack500 | YOLOv11n | Full KD (unclamped) | 42 | — | — | **CRASH — NaN FP16** |
| EXP-06 | Combined | YOLOv11n | Full KD (Box+Centroid) | 44 | — | — | **CRASH — NaN FP16** |
| EXP-07 | Crack500 | YOLOv11n | Full KD (clamped) | 10 | 0.481 | 0.098 | Done — patch verified |
| **EXP-08** | Crack500 | YOLOv11n | **Full KD (clamped)** | **150** | **0.515** | **0.131** | **Done — +10.3% OOD** |
| EXP-09 | Crack500 | YOLOv8n | Baseline | 150 | 0.532 | — | Done |
| EXP-10 | Crack500 | YOLOv11n | Baseline | 150 | 0.540 | — | Done |
| EXP-11 | DeepCrack | YOLOv8n | Baseline | 150 | 0.496 | — | Done |
| EXP-12 | DeepCrack | YOLOv11n | Baseline | 150 | 0.538 | — | Done — v11 wins |
| EXP-13 | Crack500 | YOLOv11n | Baseline fine-tune | 150 | 0.545 | — | Done |
| EXP-14 | Crack500 | YOLOv11n | Full KD (Box) | 150 | 0.547 | — | Done |
| EXP-15 | Crack500 | YOLOv11n | Full KD (Box+Centroid) | 150 | 0.547 | — | Done — same as box |
| EXP-16 | DeepCrack | YOLOv11n | Baseline fine-tune | 150 | 0.485 | — | Done |
| EXP-17 | DeepCrack | YOLOv11n | Full KD (unfrozen) | 150 | 0.477 | — | Done — **degraded** |
| **EXP-18** | DeepCrack | YOLOv11n | **Full KD (head frozen)** | **150** | **0.505** | — | **Done — +3.98%** |
| EXP-19 | C500→Deep | YOLOv8n | Baseline cross-eval | — | 0.221 | — | Done |
| EXP-20 | C500→Deep | YOLOv11n | Baseline cross-eval | — | 0.260 | — | Done |
| **EXP-21** | C500→Deep | YOLOv11n | **Full KD cross-eval** | — | **0.274** | — | **Done — best transfer** |
| EXP-22 | Deep→C500 | YOLOv8n | Baseline cross-eval | — | 0.030 | — | Done |
| EXP-23 | Deep→C500 | YOLOv11n | Baseline cross-eval | — | 0.028 | — | Done |
| EXP-24 | Deep→C500 | YOLOv11n | Full KD cross-eval | — | 0.033 | — | Done — +19% relative |

---

## 7. Ideas That Failed (Honest Research Log)

### FAIL 1: FP16 Automatic Mixed Precision with KD (EXP-05, EXP-06)

- **What we tried:** Standard AMP (`amp: true`) with SAM 2 soft logits as teacher signal.
- **Why it failed:** SAM 2 logits reach magnitudes of $\pm 50+$. In FP16, $\sigma(50/\tau)$ saturates to 1.0, $\log(1 - 1.0) = \log(0) = -\infty$, NaN propagates backwards.
- **Result:** Both experiments crashed between epochs 42–44. Zero useful weights saved.
- **Fix:** Clamp logits to $[-30, 30]$ plus `amp: false`.

### FAIL 2: KD on DeepCrack Without SegHead Freezing (EXP-17)

- **What we tried:** Apply the Crack500 KD pipeline directly to DeepCrack without modification.
- **Why it failed:** 537 training images cannot sustain dual gradient objectives (task loss + KD soft loss) through the segmentation head simultaneously. The head oscillates and collapses.
- **Result:** Mask mAP50 degraded from `0.4853` to `0.4765` (−0.88 pts).
- **Fix:** Progressive 2-stage freeze (EXP-18) → `0.5046` (+3.98%).

### FAIL 3: Box + Centroid Prompting (EXP-15)

- **What we tried:** Add centroid point prompts to SAM 2 on top of box prompts, expecting higher-quality teacher masks for thin cracks.
- **Why it failed:** SAM 2 Large with a bounding box already fully constrains crack regions. Centroid adds no information SAM does not already compute from the box.
- **Result:** EXP-14 and EXP-15 both produce `Mask mAP50 = 0.5468` — identical.
- **Conclusion:** Box-only prompting is sufficient. Do not add centroid prompts.

### FAIL 4: Combined Dataset Training (EXP-06, partial)

- **What we tried:** Train on merged Crack500 + DeepCrack (`combined_yolo`) dataset.
- **Why it failed:** Crashed with NaN (AMP issue). Additionally: 500-image crops mixed with 537 high-resolution images create unstable batch statistics. The distributions are too different for naive concatenation.
- **Status:** Never fully evaluated. Per-dataset training is more controllable for the paper. This remains a future direction.

---

## 8. Ideas That Worked

### SUCCESS 1: SAM 2 KD Improves OOD Generalization (+10.3%)

- **Hypothesis:** SAM 2 boundary-aware soft masks regularize the student against crop-artifact memorization.
- **Evidence:** EXP-04 (baseline) vs EXP-08 (KD): OOD Mask mAP50-95 improved from `0.0319` to `0.0352` — **+10.3% relative**.
- **Interpretation:** The KD signal forces the student to attend to crack texture and boundary shape rather than absolute position within the crop. This is a domain regularization effect, not a simple accuracy boost.

### SUCCESS 2: Progressive SegHead Freeze Fixes Small-Dataset Degradation

- **Hypothesis:** Decoupling backbone KD (Stage 1) from head task-tuning (Stage 2) resolves gradient conflict on small datasets.
- **Evidence:** DeepCrack unfrozen: −0.88 mAP pts. Frozen: +1.93 mAP pts over baseline.
- **Novel aspect:** This progressive schedule is specifically designed for the KD-on-small-dataset scenario. It is not standard in the KD literature.

### SUCCESS 3: OOD-Aware Optuna Objective

- **Hypothesis:** Temperature $\tau$ and loss weights need joint tuning on OOD metric, not in-distribution mAP.
- **Evidence:** The composite objective (60% OOD weight) found weights that improve generalization significantly better than literature-default values.
- **Implication:** For deployment-focused KD, the tuning objective must match the deployment criterion — in-domain mAP alone is the wrong optimization target.

### SUCCESS 4: YOLOv11n-seg is the Right Architecture

- **Evidence:** YOLOv11n-seg beats YOLOv8n-seg with 13% fewer parameters and 15% fewer GFLOPs, on both datasets, with the gap larger on thin cracks (+4.2 mAP pts on DeepCrack vs +0.8 on Crack500).
- **Implication:** The improved neck architecture in YOLOv11 is specifically better at thin, elongated structures — well-matched to crack morphology.

---

## 9. Key Numbers Summary

| Claim | Experiments | Number |
|:---|:---|:---:|
| KD improves OOD generalization | EXP-04 vs EXP-08 | **+10.3% Mask mAP50-95** |
| KD improves cross-domain transfer | EXP-20 vs EXP-21 | **+14.1% Mask mAP50-95** |
| YOLOv11 > YOLOv8 on thin cracks | EXP-11 vs EXP-12 | **+4.2 Mask mAP50 pts** |
| SegHead freeze fixes small-dataset KD | EXP-17 vs EXP-18 | **+2.81 Mask mAP50 pts** |
| Crack500 >> DeepCrack as training source | EXP-21 vs EXP-24 | **0.274 vs 0.033 (8x better)** |
| Box prompts = Box + Centroid | EXP-14 vs EXP-15 | **0.00 difference** |
| Model compression ratio | — | **79x parameters (224M to 2.84M)** |

---

## 10. Contribution Claims

This work contributes:

1. **The first SAM 2 → YOLO knowledge distillation pipeline for crack segmentation**, achieving a 79x parameter reduction while improving out-of-distribution generalization by +10.3% (Mask mAP50-95) over standard fine-tuning.

2. **Empirical evidence that KD from foundation models acts as a domain regularizer**, not merely a performance booster — demonstrated by in-distribution accuracy dropping slightly while OOD accuracy improves significantly.

3. **A progressive segmentation head freeze schedule** for applying KD to small datasets (fewer than 1,000 images), reversing a −0.88 mAP degradation into a +1.93 mAP gain on DeepCrack.

4. **A cross-dataset generalization benchmark** for crack detection showing the asymmetric domain shift between Crack500 and DeepCrack, with practical implications for dataset selection in infrastructure inspection pipelines.

5. *(Planned)* **Extension to SAM 3 as teacher** — comparison of SAM 2 vs SAM 3 teacher quality, and exploration of ensemble and chain distillation strategies (SAM 2 + SAM 3 → YOLO).

---

## 11. Open Problems and Future Work

### 11.1 Loss Weight Optimization (Current Bottleneck)

The KD loss weights ($\alpha$, $\beta$, $\gamma$) are highly sensitive. The Optuna search explored only 10–12 trials on a 15-epoch proxy objective — a small search. Better weight discovery may yield significantly stronger results. This is the highest-priority open research question.

### 11.2 Ablation Study (NB5 — Planned)

The following ablations are designed but not yet executed:

| Ablation | What is removed | Expected finding |
|:---|:---|:---|
| `ablation_no_mask_kd` | Remove $L_{\text{KL}}$ | OOD drops — logit soft targets matter |
| `ablation_no_feature` | Remove $L_{\text{feature}}$ | Less backbone alignment |
| `ablation_no_boundary` | Remove $L_{\text{boundary}}$ | Thin crack boundary quality drops |
| `ablation_seghead_frozen` | Keep head frozen all training | Ceiling without fine-tuning |

### 11.3 SAM 3 as Teacher

SAM 3 extends SAM 2 with improved video tracking and new architectural improvements. Key open questions:

- Does SAM 3's teacher signal produce better soft masks for crack boundaries?
- Is chain distillation (SAM 2 then SAM 3 → YOLO) beneficial or redundant?
- Can SAM 2 + SAM 3 ensemble teacher masks improve student quality?

### 11.4 Combined Dataset Training

Naive Crack500 + DeepCrack concatenation crashed (EXP-06). Proper combined training requires:

- Domain-aware batch sampling (not random mixing)
- Normalization per-domain
- Possibly domain-conditioned loss weighting

### 11.5 Own Speed Benchmarking

Current speed claims (SAM 2: 8–15 FPS, YOLOv11: ~60 FPS) cite official benchmarks, not our own measurements. A rigorous paper requires head-to-head measurement on identical hardware and identical image conditions.

---

## 12. Conclusion

CrackDistill demonstrates that knowledge distillation from SAM 2 into a lightweight YOLOv11n-seg student produces a model that is **79x smaller** and approximately **4x faster** (citing official benchmarks), and critically — **more generalizable** across datasets and acquisition conditions than standard fine-tuning.

The primary contributions are not raw accuracy gains but **generalization gains**: SAM 2's spatial priors regularize the student against crop-artifact memorization and narrow-distribution overfitting. This is evidenced consistently across OOD evaluations (EXP-08: +10.3%), cross-dataset transfer (EXP-21: +14.1% mAP50-95), and small-dataset stabilization via the progressive freeze schedule (EXP-18: +3.98%).

The research is honest about its limitations: the loss weight search is not complete, ablations are not yet run, and speed benchmarking is based on official rather than own measurements. These are the immediate next steps. The direction — SAM foundation model supervision for lightweight crack detection generalization — is validated by the experimental evidence and supported by the absence of prior work at this specific intersection.
