# CrackDistill v2 — how the distillation will work

*2026-10-14. Replaces the training/evaluation protocol in `EXPERIMENTS.md` (v1). The diagrams are Mermaid:
they render on GitHub, in Obsidian and in VS Code's Markdown preview.*

## 1. Decisions at a glance

| Question | Decision | Why (details below) |
| :--- | :--- | :--- |
| Training data | Crack500 **train crops only** (1,896), identical for every arm | Standard benchmark; uncropped train photos are not in the public release; extra data would confound KD vs baseline |
| Test data | (a) the **1,124 canonical test crops**, (b) the **200 full test photos**, (c) **zero-shot** CFD, GAPs384, CRKWH100 (+ DeepCrack as near-OOD) | (a) matches the literature, (b) is the deployment case, (c) is real out-of-distribution |
| Train on crops or full photos? | Crops, at their **native pixel scale** (×0.8 inside the 512 canvas) | Crack widths: median 15–17 px, 10th percentile 3.4 px — downscaling full photos erases the thin ones |
| Test on full photos how? | **Same pixel scale as training**: resize ×0.8, 512×512 tiles, 50% overlap, Gaussian blending, val-tuned threshold | Train/test scale must match; tiling is standard (nnU-Net, SAHI) |
| Augmentation | Every arm: photometric + **D4** (8 flips / 90° rotations); the cached teacher map gets the **same** D4 element | v1 had no spatial augmentation and every arm overfit by epoch 9–17 |
| Distillation signal | Per-instance Bernoulli mask-KL to SAM 2 box-prompt logits (unchanged loss, T=3.7769, w=0.9612) | Isolates one change (augmentation); loss variants come later only if needed |
| Baselines | **A1** same D4 policy, no KD (the paired baseline) · **A4** Ultralytics default augmentation (mosaic, scale, flips) — the "normal YOLO" reference | KD must beat a strong baseline, not a crippled one |
| Primary metric | Pixel F1 at a val-tuned threshold on the test crops; COCO mask mAP50 secondary | Crack benchmarks are semantic; reviewers expect F1 / IoU / clDice |

## 2. Where we are (v1, seed 0, Kaggle T4, val split)

| Arm (no spatial augmentation) | `best.pt` mask mAP50 | mean epochs 6–25 | mean epochs 26–60 | best epoch |
| :--- | :---: | :---: | :---: | :---: |
| Baseline | 0.410 | 0.376 | 0.331 | 15 |
| Mask-KD | **0.463** | **0.404** | **0.383** | 17 |
| GT-soft control | 0.414 | 0.391 | 0.374 | 9 |

- KD beats both baseline and the soft-label control in every window — promising, one seed.
- **Every arm overfits**: val peaks early while train seg loss falls 3.1 → 0.3. This is the "fixed teacher"
  failure on small data ([Beyer et al. 2022](https://arxiv.org/abs/2106.05237)): a precomputed target that
  does not change with the view.
- The real level of an *augmented* baseline is **unknown**. The old 0.54 came from a baseline that still
  received KD supervision before the September fixes. A4 measures it.

## 3. Data

**Crack500 facts (checked locally):** 250 / 50 / 200 photos. Train exists only as crops: an exact 4×4 grid
(640×360 from 2560×1440, 648×484 from 2592×1936), keeping crops with >1,000 crack pixels. The 1,124 test crops
come from **148** of the 200 test photos; the other 52 (`IMG_*`, mostly 3264×2448) exist only as full photos.
65% of full photos carry an EXIF rotation that their masks do **not** follow.

| Use | Dataset | Notes |
| :--- | :--- | :--- |
| Train / val / test | Crack500 official split | 1,896 / 348 / 1,124 crops; 50 / 200 full photos |
| Zero-shot OOD | [CFD](https://github.com/cuilimeng/CrackForest-dataset) (118, 480×320, asphalt) | Used by FPHBN for cross-dataset tests |
| Zero-shot OOD | GAPs384 (509 test crops, German asphalt) | FPHBN protocol; registration needed |
| Zero-shot OOD | [CRKWH100](https://github.com/qinnzou/DeepCrack) (100, 512×512, asphalt) | Thin labels → use clIoU with 4 px tolerance |
| Near-OOD (local) | DeepCrack test (237, 544×384) | Mixed concrete/stone; never in our training set |
| **Not used** | CrackSeg9k, Khanh11k, OmniCrack30k | Contain Crack500/CFD/DeepCrack images in random splits → test leakage |

**Known limitation:** every training crop contains a crack, so the student never sees bare pavement and will
produce false positives on full photos. The val-tuned threshold partly compensates; report precision on full
photos separately.

## 4. Training at native scale, testing on full photos

The rule: **a crack must be the same number of pixels wide at train and test time.**

```mermaid
flowchart LR
    subgraph TRAIN["Training (crops)"]
        C["Crop 640×360<br/>native scale"] --> R1["resize ×0.8<br/>→ 512×288"] --> L["letterbox to 512×512<br/>(pad 112 px top/bottom)"]
    end
    subgraph TEST["Testing (full photo)"]
        P["Photo 2560×1440<br/>read with EXIF ignored"] --> R2["resize ×0.8<br/>→ 2048×1152"] --> T["512×512 tiles<br/>50% overlap, edge pad 114"]
    end
    L -.-|"same pixels per crack"| T
```

Full-photo test recipe:

1. Read raw pixels (`cv2.IMREAD_IGNORE_ORIENTATION`) — the masks follow the raw sensor orientation.
2. Resize by 0.8 (the training scale). The 52 `IMG_*` photos come from other cameras: report them separately.
3. Pad right/bottom with gray 114, cut 512×512 tiles with 50% overlap.
4. Run the student with `conf=0.001` and build a **soft** crack map per tile: for each instance,
   sigmoid(prototype · coefficients), cropped to its box, × its score; combine instances with max.
5. Blend tiles with a 2-D Gaussian weight (centre counts more), crop the padding, resize back to native size.
6. Threshold with τ\* chosen on the 50 full val photos (never on test); compute metrics against the native mask.

The test crops use the same model and the same soft map, without tiling (one crop = one 512 canvas).

## 5. The distillation architecture

### 5.1 System overview

```mermaid
flowchart TB
    subgraph OFF["Offline, once"]
        D["Crack500 train crops<br/>+ YOLO polygon labels"] --> B["one box per labelled crack<br/>(label-file order)"]
        B --> SAM["SAM 2 Large<br/>box prompt per crack"]
        D --> SAM
        SAM --> TL[("teacher logits<br/>&lt;stem&gt;_logits.npy<br/>M × 256 × 256, full crop span")]
        D --> GS["GT-soft generator<br/>blur σ=2, logits capped ±14"] --> GL[("GT-soft logits<br/>(control arm only)")]
    end
    subgraph TRN["Training — Kaggle T4, 150 epochs, best.pt on val"]
        DS["Ultralytics dataset<br/>letterbox + HSV"] --> D4["D4AugDataset<br/>random k ∈ {0..7}"] --> ST["YOLOv11n-seg student<br/>2.84 M params"]
        TL --> KD["KD loss<br/>(teacher warped + same k)"]
        ST --> TASK["task loss<br/>box · cls · dfl · mask · semantic"]
        ST --> KD
        TASK --> SUM["Σ loss"]
        KD --> SUM
    end
    subgraph EVAL["Evaluation — test once"]
        CK["best.pt"] --> E1["test crops 1,124"]
        CK --> E2["200 full photos<br/>scale-matched tiling"]
        CK --> E3["zero-shot CFD · GAPs384 · CRKWH100 · DeepCrack"]
        CK --> E4["T4 latency<br/>tile + full scene"]
    end
    TRN --> CK
```

At inference the student runs alone: no SAM 2, no extra parameters.

### 5.2 One training step

```mermaid
flowchart TB
    IMG["crop 640×360"] --> LB["letterbox → 512×512<br/>ratio_pad = (h_ratio, w_ratio)"]
    LB --> AUGK["D4 element k<br/>applied to image, GT masks,<br/>semantic mask, boxes"]
    AUGK --> NET["YOLOv11n-seg"]
    NET --> PROTO["prototypes 128×128<br/>× coefficients of anchors<br/>matched to GT instance g"]
    PROTO --> SL["student mask logits<br/>resized to 256×256"]

    CACHE[("teacher[stem]<br/>M × 256 × 256")] --> PICK["pick teacher[g]<br/>(same instance order as GT)"]
    PICK --> BOX["letterbox_content_box<br/>(ori_shape, ratio_pad)"]
    BOX --> WARP["warp into 256×256 canvas<br/>pad = −20 (background)"]
    WARP --> TK["apply the same k<br/>to teacher + valid-pixel mask"]

    SL --> KL["Bernoulli KL( σ(t/T) ‖ σ(s/T) ) · T²<br/>on valid pixels only"]
    TK --> KL
    KL --> W["× w = 0.9612"]
    W --> TOT["total = task + KD"]
```

**Why D4 and not mosaic/scale:** after letterboxing the canvas is square, so all 8 flips/90° rotations are
exact index permutations of the image, the ¼-resolution masks and the warped teacher alike — no resampling,
no instance can drop out, the teacher-to-instance pairing (`teacher[target_gt_idx]`) is untouched. Mosaic,
scale and translate would need the cached map to be resampled through Ultralytics' affine pipeline and would
drop instances (breaking the pairing). Cracks have no preferred orientation, and rotation was the single most
useful augmentation in a small-crack-dataset ablation ([Wang et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7506713/)).

**Verified (tests in `tests/test_d4_augmentation.py`):** boxes follow the pixels for all 8 elements; through
Ultralytics' real training transforms, the warped + rotated teacher lands on the rotated GT (IoU > 0.9) for all
8 elements; a 1-epoch smoke run reached the KD loss with all 8 elements and 0 errors.

### 5.3 Teacher quality (what the student can learn from)

- SAM 2 vs GT on train crops: median IoU **0.605** (the blurred-GT control: 0.796). SAM misses thin
  branches — a known weakness on thin structures ([SAM](https://arxiv.org/abs/2304.02643),
  [WACV 2026](https://arxiv.org/abs/2412.04243)).
- Cache resolution: 256×256 over the whole crop → ~2.5 px per cell horizontally, ~1.4 px vertically.
- GT masks at Ultralytics' ¼ resolution keep 97.5% of crack pixels on Crack500 (measured), so the mask
  resolution is not the bottleneck.

## 6. Experiment plan

```mermaid
flowchart TB
    A["Stage A — seed 0, 4 runs, ~10 T4-h<br/>A1 baseline+D4 · A2 Mask-KD+D4 · A3 GT-soft+D4 · A4 baseline+Ultralytics aug"]
    A --> G{"val (crops): A2 vs A1 and A4"}
    G -->|"A2 > A1 and A2 ≥ A4"| C["Stage C — confirm<br/>A1, A2, A3 × seeds 1–4 (+A4 × 1–4)<br/>~35–40 T4-h"]
    G -->|"A2 > A1 but A4 > A2"| B["Stage B — close the gap (pick 1–2, seed 0)<br/>• GT-gated KD: distil only where SAM agrees with GT<br/>• teacher TTA: average SAM over 8 D4 views<br/>• teacher follows scale/translate too"]
    G -->|"A2 ≤ A1"| N["KD does not help once augmentation is on.<br/>Report v1 + A-stage honestly as a negative result;<br/>add a data-efficiency curve (25/50/100% data)"]
    B --> G
    C --> D["Stage D — local, inference only<br/>test crops · 200 full photos · zero-shot OOD · speed<br/>seed-paired stats, bootstrap over parent photos"]
```

| Stage | Runs | T4-hours | Output |
| :--- | :--- | :---: | :--- |
| A | `A1`–`A4`, seed 0 (generated in `final_notebooks/`) | ~10 | which arm wins on val |
| B (only if needed) | 1–2 loss/teacher variants, seed 0 | ~5 | a KD variant that beats A4 |
| C | winners × seeds 1–4 | ~35–40 | means, CIs, paired tests |
| D | none (local inference) | — | final tables |

**Statistics (Stage C/D):** same seeds in every arm, paired differences with 95% CIs, bootstrap over the 148
parent photos (crops are not independent), Holm correction over A2 vs A1 and A2 vs A3, and the minimum
detectable effect. 5 seeds detect ~0.01 if the seed SD is ≤ 0.004.

## 7. What "better than baseline" will be allowed to mean

| Result | Claim |
| :--- | :--- |
| A2 > A1, A2 > A3, A2 ≥ A4 (CIs exclude 0) | Distilling SAM 2 improves a strongly augmented YOLOv11n-seg, beyond label softening |
| A2 > A1, A2 ≈ A3 | The gain is soft-label regularization; SAM is not needed for it |
| A2 > A1 but A4 > A2 | KD helps only where geometric augmentation can't be used — a narrower, still honest claim |
| A2 ≤ A1 | No KD benefit under augmentation; v1's gain was a substitute for augmentation |

## 8. Status

| Piece | State |
| :--- | :--- |
| Letterbox warp, instance pairing, GT-soft control, strict guards | done, tested (v1) |
| D4 augmentation (dataset + teacher + trainer switch `train.d4_aug`) | **done, tested** |
| Stage A notebooks `A1`–`A4` + `B1_benchmark_speed` | **generated** |
| Scale-matched tiling, soft instance maps, pixel F1 / IoU / clDice / clIoU evaluator | to build before Stage D |
| CFD / GAPs384 / CRKWH100 download + conversion | to do before Stage D |
| GT-gated KD, teacher TTA | Stage B, only if the gate needs them |

## 9. Sources

Beyer et al. 2022, *A good teacher is patient and consistent* — [2106.05237](https://arxiv.org/abs/2106.05237) ·
ReLabel — [2101.05022](https://arxiv.org/abs/2101.05022) · FKD — [2112.01528](https://arxiv.org/abs/2112.01528) ·
Niemann et al. 2023 — [2309.03659](https://arxiv.org/abs/2309.03659) · Yuan et al. 2020 (KD ≈ label smoothing) —
[1909.11723](https://arxiv.org/abs/1909.11723) · SVLS — [2104.05788](https://arxiv.org/abs/2104.05788) ·
FPHBN / Crack500 — [1901.06340](https://arxiv.org/abs/1901.06340) · OmniCrack30k —
[CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Benz_OmniCrack30k_A_Benchmark_for_Crack_Segmentation_and_the_Reasonable_Effectiveness_CVPRW_2024_paper.pdf) ·
nnU-Net — [1809.10486](https://arxiv.org/abs/1809.10486) · SAHI — [2202.06934](https://arxiv.org/abs/2202.06934) ·
clDice — [2003.07311](https://arxiv.org/abs/2003.07311) · Metrics Reloaded — [2206.01653](https://arxiv.org/abs/2206.01653) ·
Kompanets et al. (patches, negatives) — [2403.17725](https://arxiv.org/abs/2403.17725) ·
Wang et al. 2020 (crack augmentation ablation) — [PMC7506713](https://pmc.ncbi.nlm.nih.gov/articles/PMC7506713/) ·
ACE gating — [PMC7471971](https://pmc.ncbi.nlm.nih.gov/articles/PMC7471971/)
