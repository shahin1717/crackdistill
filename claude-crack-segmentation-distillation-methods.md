# SAM → YOLO Knowledge Distillation for Road Crack Segmentation
## Fixing the Uncropped-Image Performance Drop

## Why Uncropped Training Fails Here
- Cracks are thin, sparse structures — tiny fraction of total pixels in a full road image.
- Downsampling a full-res image to YOLO's input size (e.g. 640) destroys 1–5px wide crack detail before the model sees it.
- YOLO-seg's mask prototypes are coarse (often 1/4–1/8 of input resolution) — too low-res for thin structures.
- SAM's high-res teacher mask gets downsampled to match YOLO's output during distillation, erasing crack signal before the loss is even computed.
- Cropping "fixed" this by increasing the crack-to-image pixel ratio, but broke generalization since real inference images aren't pre-cropped.

## 1. Tiling Instead of Cropping (Primary Fix)
- Split uncropped images into overlapping tiles (e.g. 512×512, 20–25% overlap) for both training and inference.
- Preserves a favorable crack-to-tile pixel ratio at train time.
- At inference: tile the full image, run predictions per tile, stitch back together.
- Stitching: use max or averaged overlap in overlapping regions to avoid seam artifacts.
- Eliminates train/test distribution mismatch entirely — no crop-vs-uncropped gap.

## 2. Increase Input/Output Resolution
- Use larger YOLO input size (e.g. 1280 instead of 640) to preserve thin structures.
- Check YOLO-seg's mask prototype resolution; upsample before comparing to SAM's mask if needed.
- Higher resolution costs compute/memory — tiling (above) often achieves this more efficiently than brute-force upscaling.

## 3. Loss Function Adjustments
- Add or switch to Dice / Tversky loss instead of relying solely on BCE or IoU.
- Tversky loss lets you weight false negatives higher — directly targets the "missed thin crack" failure mode.
- Consider boundary-aware losses (e.g. boundary IoU) since crack edges carry most of the signal.

## 4. Fix Resolution Mismatch in the Distillation Loss
- Don't downsample SAM's mask to YOLO's coarse output before computing loss.
- Instead: upsample YOLO's prediction to SAM's resolution, or distill at intermediate feature-map level rather than final mask level.
- Feature-level distillation (matching encoder activations) can preserve more fine-grained spatial detail than mask-level distillation alone.

## 5. Class-Imbalance-Aware Sampling
- Pure random sampling from uncropped/tiled images will be dominated by crack-free regions.
- Oversample tiles/crops that contain cracks during training (e.g. weighted sampler based on crack pixel count).

## 6. Augmentation Suited to Elongated Structures
- Favor: rotation, elastic deformation, contrast/brightness jitter, mild noise/blur.
- Avoid: aggressive random cropping as primary augmentation (reintroduces the original problem).
- Cracks vary heavily in contrast against pavement — augment for that variability specifically.

## Priority Order (Practical)
1. Switch from cropping to tiling (train + inference) — biggest single fix
2. Fix resolution handling in the distillation loss (upsample prediction, not downsample teacher mask)
3. Add Dice/Tversky loss for thin-structure sensitivity
4. Add crack-weighted sampling
5. Tune augmentation away from crop-heavy strategies
6. Increase resolution further only if tiling + above aren't sufficient
