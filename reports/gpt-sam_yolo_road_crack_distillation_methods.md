# SAM → YOLO Knowledge Distillation for Road-Crack Segmentation

## Problem

The task is to distill a segmentation model from **SAM (teacher)** into **YOLO-Seg (student)** for road-crack segmentation.

The model performs well when trained on **cropped images**, but training directly on **uncropped/full images** produces poor results.

This is especially important for road cracks because cracks are usually very thin structures and can lose most of their useful pixel information when a large road image is resized to a small YOLO input resolution.

---

# 1. Why Cropped Training Works Better

Consider a high-resolution road image:

```text
Original image
2048 × 1536

+--------------------------------+
|                                |
|          ─────────             |
|             crack              |
|                                |
+--------------------------------+
```

If the complete image is resized to 640×640:

```text
2048 × 1536
     ↓
  640 × 640
     ↓
very thin crack
```

The crack can become only a few pixels wide or even effectively disappear.

With a crop:

```text
Original image
      ↓
crack-centered crop
      ↓
512 × 512
      ↓
crack occupies many more pixels
      ↓
YOLO-Seg
```

The model receives much more information about the crack.

Therefore, **cropping is not necessarily a problem**. For thin-object segmentation, it can be an important part of the preprocessing pipeline.

---

# 2. Method 1 — Keep Crop-Based Training

If crop-based training gives significantly better segmentation, keep it as the baseline.

```text
Road image
    ↓
Crack/object crop
    ↓
SAM
    ↓
Teacher mask
    ↓
YOLO-Seg
```

### Advantages

- Crack occupies more pixels.
- Fine crack structures are easier to learn.
- SAM can potentially generate better masks for small/thin cracks.
- YOLO has more useful information at its input resolution.

### Disadvantage

The student may become dependent on the crop distribution.

It may perform poorly when the crack is small or appears in an entire road image.

---

# 3. Method 2 — Overlapping Tiling

Instead of resizing the complete road image to the YOLO input size, divide the large image into smaller overlapping tiles.

```text
Full road image

+-----------------------+
|       |       |       |
| Tile1 | Tile2 | Tile3 |
|-------+-------+-------|
| Tile4 | Tile5 | Tile6 |
|-------+-------+-------|
| Tile7 | Tile8 | Tile9 |
+-----------------------+
```

Each tile is processed independently.

```text
Full image
     ↓
Overlapping tiles
     ↓
YOLO-Seg
     ↓
Tile masks
     ↓
Merge / stitch masks
     ↓
Full-resolution crack mask
```

### Why it works

The crack remains relatively large inside each tile.

Instead of:

```text
Full image → 640×640
```

you can use:

```text
Full image
    ↓
640×640 / 768×768 / 1024×1024 tiles
    ↓
YOLO
```

This preserves much more spatial information.

### Overlap

Use overlapping tiles so cracks crossing tile boundaries are not unnecessarily cut.

A reasonable starting experiment is:

```text
20–30% overlap
```

The optimal overlap should be determined experimentally.

---

# 4. Method 3 — Train on Tiles and Test on Tiles

Training and inference should ideally use the same image distribution.

Recommended:

```text
TRAIN

Full road images
       ↓
Overlapping tiles
       ↓
SAM teacher
       ↓
Teacher masks
       ↓
YOLO student
```

Then:

```text
TEST

Full road images
       ↓
Same tiling procedure
       ↓
YOLO student
       ↓
Merge tile predictions
       ↓
Full-image mask
```

This avoids a major train-test mismatch.

---

# 5. Method 4 — Crop + Full-Image Tiles

This is one of the strongest approaches to test.

Instead of choosing only crops or only full images, train using both.

```text
                 Training data
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
 Crack-centered crops        Full-image tiles
          ↓                         ↓
       SAM teacher              SAM teacher
          ↓                         ↓
       YOLO-Seg                YOLO-Seg
          └────────────┬────────────┘
                       ↓
                 Student model
```

The model learns:

- fine-grained crack features from crops;
- different backgrounds from tiles;
- different crack scales;
- different spatial positions;
- realistic deployment conditions.

This can be more effective than forcing the model to learn only from uncropped images.

---

# 6. Method 5 — Fine-Tune a Crop-Trained YOLO Model

If you already have a good YOLO model trained on crops, do not necessarily retrain from scratch.

Use:

```text
Crop-trained YOLO
       ↓
Fine-tune on full-image tiles
       ↓
Final YOLO model
```

### Stage 1

Learn detailed crack features:

```text
SAM masks
    ↓
Cropped images
    ↓
YOLO
```

### Stage 2

Adapt to deployment:

```text
Full images
    ↓
Tiles
    ↓
YOLO fine-tuning
```

This allows the model to retain the useful representation learned from crops while adapting to different context and scale.

---

# 7. Method 6 — Multi-Scale Training

Road cracks can appear at very different scales.

Train the student using different tile/image scales:

```text
Small crop
    ↓
Large crack representation

Medium tile
    ↓
Medium crack representation

Large tile
    ↓
Small crack representation
```

Possible tile sizes to experiment with:

```text
512 × 512
640 × 640
768 × 768
1024 × 1024
```

Do not assume that larger is always better.

For thin cracks, increasing the effective crack resolution is often more important than simply increasing the amount of context.

---

# 8. Method 7 — Scale Augmentation

Apply random scale transformations during training.

For example:

```text
Large crack
Medium crack
Small crack
Very small crack
```

Useful augmentations include:

- Random scaling
- Random resized crop
- Random zoom
- Random translation
- Random padding
- Multi-scale training

The objective is to prevent YOLO from learning:

> "Cracks always occupy a large percentage of the image."

Instead, it should learn:

> "This visual structure is a crack regardless of its scale."

---

# 9. Method 8 — Random Position Augmentation

If your crops are always centered on the crack, the student can learn a positional bias.

For example:

```text
Training crop 1

+----------------+
|                |
|      CRACK     |
|                |
+----------------+
```

Instead, use different positions:

```text
+----------------+
| CRACK          |
|                |
|                |
+----------------+
```

```text
+----------------+
|                |
|          CRACK |
|                |
+----------------+
```

This helps the model become less dependent on the crack being centered.

---

# 10. Method 9 — Background Augmentation

Crop-based datasets may contain limited background variation.

The student can accidentally learn correlations between:

```text
background + crack
```

rather than the crack itself.

Use different backgrounds when possible:

```text
Crack + asphalt A
Crack + asphalt B
Crack + asphalt C
Crack + lighting condition D
```

This can improve robustness to real road scenes.

---

# 11. Method 10 — Check SAM Teacher Masks

This is particularly important in a **SAM → YOLO knowledge-distillation** pipeline.

Do not assume that the teacher produces equally good masks on crops and full images.

Compare:

```text
SAM + crop
```

against:

```text
SAM + full image
```

For example:

```text
SAM on crop

+----------------+
|                |
|    ───────     |
|   crack        |
|                |
+----------------+

       ↓

precise teacher mask
```

versus:

```text
SAM on full image

+--------------------------------+
|                                |
|             tiny crack        |
|                ───             |
|                                |
+--------------------------------+

       ↓

potentially weaker mask
```

If SAM's full-image masks are worse, YOLO is being trained to imitate a worse teacher.

Therefore, the problem may not be YOLO alone.

---

# 12. Method 11 — Generate Teacher Masks on Tiles

Instead of running SAM on the entire road image:

```text
Full image
    ↓
SAM
    ↓
Mask
```

use:

```text
Full image
    ↓
Overlapping tiles
    ↓
SAM
    ↓
High-resolution crack masks
    ↓
Merge masks
```

This allows SAM to see the crack at a larger effective scale.

The resulting masks can then be used as the distillation targets for YOLO.

---

# 13. Method 12 — Teacher and Student Should See Similar Inputs

For knowledge distillation, the teacher and student should ideally operate on corresponding images.

Recommended:

```text
Original road image
       ↓
     Tile
       │
       ├──────────────┐
       ↓              ↓
     SAM             YOLO
  Teacher           Student
       ↓              ↓
 Teacher mask     Student mask
       │              │
       └──────┬───────┘
              ↓
       Distillation loss
```

This is preferable to:

```text
SAM sees crop
      ↓
teacher mask

YOLO sees full image
      ↓
student mask
```

because the teacher and student are then solving slightly different input problems.

---

# 14. Method 13 — Distillation Loss

If your implementation supports it, the student can be trained using a combination of normal segmentation loss and teacher supervision.

Conceptually:

```text
Total Loss
    =
Segmentation Loss
    +
λ × Distillation Loss
```

For example:

```text
L_total = L_seg + λ L_distill
```

where:

- `L_seg` = segmentation loss against available ground-truth labels;
- `L_distill` = loss encouraging YOLO predictions to reproduce SAM's masks;
- `λ` = strength of the distillation term.

If SAM masks are pseudo-labels rather than ground truth, it is particularly important to evaluate their quality before giving them high weight.

---

# 15. Method 14 — Confidence/Quality Filtering of SAM Masks

Not every SAM mask should necessarily become a training target.

Conceptually:

```text
SAM
 ↓
Candidate masks
 ↓
Quality filtering
 ↓
High-quality masks
 ↓
YOLO training
```

Poor teacher masks can introduce noise into the student.

You can filter masks using appropriate quality criteria available from your SAM pipeline, or manually inspect a validation subset.

---

# 16. Method 15 — Sliding-Window Inference Without Retraining

If you cannot change the trained YOLO model, use the crop-trained model on tiles.

```text
Uncropped image
       ↓
Sliding window
       ↓
Crop 1 → YOLO
Crop 2 → YOLO
Crop 3 → YOLO
...
       ↓
Combine masks
       ↓
Full-image prediction
```

This is useful when the existing crop-trained model is already good.

It effectively makes the inference distribution more similar to training.

---

# 17. Method 16 — Mask Stitching

After predicting each tile, map every predicted mask back to its original coordinates.

Conceptually:

```text
Tile 1 → mask 1 ─┐
Tile 2 → mask 2 ─┤
Tile 3 → mask 3 ─┼→ Full-resolution mask
Tile 4 → mask 4 ─┤
Tile 5 → mask 5 ─┘
```

For overlapping areas, combine predictions appropriately.

Possible approaches include:

- maximum probability;
- averaging probabilities;
- weighted averaging;
- confidence-based merging.

The exact method should be evaluated on your validation set.

---

# 18. Important Experiment: Diagnose the Actual Failure

Do not only compare:

```text
Crop training
vs
Full-image training
```

Run a controlled experiment.

| Experiment | Training Input | Test Input |
|---|---|---|
| A | Crop | Crop |
| B | Full image | Full image |
| C | Crop | Tiles |
| D | Tiles | Tiles |
| E | Crop + tiles | Tiles |
| F | Crop → fine-tune on tiles | Tiles |

The most informative comparison is likely:

```text
A: Crop → Crop
D: Tile → Tile
E: Crop + Tile → Tile
F: Crop → Tile fine-tuning → Tile
```

---

# 19. Determine Whether Resolution Is the Main Problem

Measure the crack width in pixels after preprocessing.

For example:

```text
Original crack width: 12 px

Full-image resize:
12 px → 2 px

Crop resize:
12 px → 15 px
```

If this happens, the model is not necessarily failing because of insufficient context.

It may simply be losing the crack signal.

For thin structures, this is a critical diagnostic.

---

# 20. Recommended Pipeline

For your specific SAM → YOLO road-crack task, I would start with this:

```text
                    Original road image
                           │
                           ↓
                 Overlapping tiling
                           │
             ┌─────────────┴─────────────┐
             ↓                           ↓
       SAM on tiles              Crack-centered crops
       (teacher)                        │
             ↓                          ↓
       Teacher masks                 SAM masks
             │                          │
             └────────────┬─────────────┘
                          ↓
                     YOLO-Seg
                          ↓
                 Multi-scale training
                          ↓
                  Tile-based inference
                          ↓
                   Mask stitching
                          ↓
                Full-resolution mask
```

---

# 21. Recommended Training Strategy

### Stage 1 — Strong crack representation

Train YOLO on high-quality SAM-generated crops.

```text
SAM crop masks
      ↓
YOLO-Seg
```

### Stage 2 — Deployment adaptation

Fine-tune using overlapping full-image tiles.

```text
Full image
      ↓
Tiles
      ↓
SAM teacher masks
      ↓
YOLO fine-tuning
```

### Stage 3 — Robustness

Add:

- scale augmentation;
- position augmentation;
- background variation;
- multiple tile sizes.

### Stage 4 — Evaluation

Evaluate separately on:

```text
1. Cropped images
2. Full images
3. Full images using tiling
```

This tells you exactly where the performance is being lost.

---

# 22. What I Would Try First

If your current results are:

```text
Crop → YOLO → GOOD

Full image → YOLO → BAD
```

I would test these in this order:

### Experiment 1

```text
Existing crop-trained YOLO
        ↓
Full image → overlapping tiles
        ↓
YOLO
        ↓
stitch masks
```

If this becomes good, the problem is mainly **effective resolution / scale**.

### Experiment 2

```text
Train YOLO on tiles
        ↓
Test YOLO on tiles
```

If this works, your previous full-image training failed because of the resizing/distribution.

### Experiment 3

```text
Crop-trained YOLO
        ↓
Fine-tune on tiles
        ↓
Test on tiles
```

This is a very strong candidate for your final model.

### Experiment 4

```text
Crop + tiles
      ↓
YOLO
      ↓
tiles
```

This tests whether combining detailed crop information with realistic scene information gives the best generalization.

---

# Final Recommendation

For **road-crack segmentation with SAM → YOLO knowledge distillation**, I would **not simply switch from cropped images to uncropped images**.

The better strategy is:

```text
              HIGH-RESOLUTION CRACK INFORMATION
                              +
                    REALISTIC ROAD CONTEXT
                              ↓
                 ┌─────────────────────┐
                 │ Crop + Tile Training│
                 └─────────────────────┘
                              ↓
                         YOLO-Seg
                              ↓
                    Overlapping tiles
                              ↓
                       Mask stitching
                              ↓
                    Full road prediction
```

The central principle is:

> **Do not sacrifice crack resolution just to give the model more context. Use tiling to preserve crack resolution while still operating on the original uncropped road image.**

For this particular problem, **crop → tile fine-tuning → tiled inference** is the first pipeline I would benchmark.
