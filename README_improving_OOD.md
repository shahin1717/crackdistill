# Crack-Distill — Improving OOD (Uncropped) Performance

*This assumes the OOD eval from the previous README has been run and shows a real gap. Measure first — don't optimize against a number you don't have yet.*

---

## 0. Before anything else: measure it with the fixed pipeline

You don't yet have a verified OOD number from the current (working, box-only, mask-KD) pipeline. Get:
- No-KD baseline on uncropped val
- Box-only mask-KD on uncropped val

Both from checkpoints you already have — pure inference, no retraining. Everything below only matters once you know the size of the actual gap.

---

## 1. Rule out a resolution/scale mismatch first — highest leverage, costs nothing

Crack500 crops are tight, dense views where a crack fills a meaningful fraction of a 512×512 frame. A full uncropped photo resized down to 512×512 can shrink that same crack to a 1–2px hairline — invisible to the model regardless of training quality.

**Check:** look at a handful of resized uncropped images yourself. If cracks are barely visible at that resolution, this is the answer.

**Fix (deployment-side, no retraining):** tiled/sliding-window inference — split the uncropped image into overlapping crop-sized tiles, run inference per tile, merge results. This alone could close most of the gap and is testable same-day on your existing checkpoint.

---

## 2. Scale-diverse training augmentation

If training data is only ever tightly-cropped at one apparent scale, the model hasn't learned scale invariance. Add:
- Random-scale / zoom-out augmentation (wider scale range than default)
- Crops with more surrounding context, pulled from the same source images, not just tight crack-centered crops

This targets the train/test domain gap directly rather than hoping KD bridges it indirectly.

---

## 3. Distill on unlabeled OOD-style images (the KD-specific lever)

Right now teacher logits are almost certainly generated only from the same cropped images the labels come from — the student never gets *any* signal, hard or soft, about what OOD images look like.

**Idea:** run SAM2 (box prompts from the student's own predictions, or any available annotation) over uncropped/wide-shot pavement images — even unlabeled ones — and include those as extra soft-target training examples.

This is the one that would actually validate the "SAM2 KD as domain regularizer" hypothesis your project is built around, rather than assuming it. Worth doing if 1–2 don't fully close the gap.

---

## 4. Test-time augmentation / multi-scale inference — cheap, no retraining

Average predictions across a couple of scales/crops of the same uncropped image at inference time. Doesn't fix the underlying model, but nearly free to try, and can meaningfully soften a scale-mismatch-driven collapse while you work on 1–3.

---

## 5. Heavier domain adaptation — last resort

Adversarial feature alignment, batch-norm recalibration on target-domain statistics, etc. More moving parts, more risk of introducing another silent bug like the ones already hit in this project (missing logits, stale config overrides, checkpoint fallback collisions). Only reach for this after 1–4 have been tried and measured — not before.

---

## Practical order

1. Get the real OOD number on baseline + box-only KD checkpoints (measure, don't guess).
2. Eyeball resized uncropped images for the scale/visibility problem.
3. If scale is the issue: try tiled inference on the existing checkpoint — same day, no training.
4. If a real gap remains after that: scale-diverse augmentation retrain.
5. If still not closed: SAM2-on-OOD-images distillation — the most interesting and most work.
6. Domain adaptation machinery only if all of the above are exhausted.

**Don't skip straight to 3 or 5 because they sound like the "real" fix.** 1 and 2 are free and have historically been the actual cause of this kind of collapse in similar setups — check them first.
