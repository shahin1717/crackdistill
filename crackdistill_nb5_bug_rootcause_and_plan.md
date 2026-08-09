# Crack-Distill: The nb5a–e Bug, and How to Actually Improve Results

*Root-caused the "no logits" issue directly in the repo's own notebook-generator scripts. This explains the whole ablation table from last time. Below: the exact bug, then a real answer to "is it data, is it freezing" — not just a bug list.*

---

## 1. The bug, fully root-caused

You were right. Here's the exact chain of evidence, traced through the code that *generates* the notebooks (`scripts/build_all_notebooks.py`), not just the notebooks themselves.

**nb2 and nb3 (the ones with real, working KD) both contain an explicit call to generate teacher logits before training:**
```python
!python scripts/generate_teacher_logits.py --prompt-type box --logits-dir data/teacher_logits_box --dataset data/datasets/crack500_yolo
```
That line — or its DeepCrack equivalent — is what actually runs SAM2 over the training images and writes the `.npy` logit files that `KDYOLODataset` later looks for.

**nb5a (and, structurally identically, nb5b/c/d) has no such line, and never even writes the `generate_teacher_logits.py` script into the notebook's filesystem in the first place.** I checked this directly against the generator source:

```python
# scripts/build_all_notebooks.py, nb5a_cells:
nb5a_cells = [
    make_cell("markdown", "..."),
] + get_self_contained_writefile_cells() + [
    ...link Crack500 raw images...,
    ...convert_crack500.py...,
    ...call KDSegmentationTrainer directly...
]
```
`get_self_contained_writefile_cells()` writes config/utils/kd_trainer/converters — but not `generate_teacher_logits.py`. There is no cell anywhere in nb5a that runs SAM2 inference or produces a single `.npy` logit file. `data/teacher_logits_box/` gets created empty by the `mkdir -p` in cell 1 and nothing ever touches it again. I confirmed this by searching the actual generated `.ipynb` file itself, not just the generator script — zero occurrences of `generate_teacher_logits` anywhere in nb5a's cell source.

**There's a second, independent bug stacked on top of this.** nb2/nb3 route training through `CrackDistillTrainer` (`distillation/trainer.py`), which correctly builds an Ultralytics config via `get_cfg(overrides=...)` and explicitly passes `logits_dir=self.logits_dir` into `KDSegmentationTrainer(...)`. nb5a skips that wrapper entirely and calls:
```python
trainer = KDSegmentationTrainer(cfg)   # cfg = the whole app-level ConfigNode from load_config()
```
with no `logits_dir` and no `kd_cfg` argument. `KDSegmentationTrainer.__init__` then falls back to `os.environ.get("KD_LOGITS_DIR", "data/teacher_logits/")` — a **different path** than the `data/teacher_logits_box/` set in the override dict, and nb5a never sets that environment variable either. So even in a hypothetical world where the logits *had* been generated, this call is pointed at the wrong directory to find them.

**Net effect:** every nb5a–e run trained with `self._sam_targets` empty for the entire run. `_kd_loss_from_preds` returns `{}` every batch, `patched_loss` adds nothing to the base task loss, and you get five differently-seeded plain fine-tunes wearing five different ablation labels. That's the real explanation for last time's "removing feature MSE *helps*" result — there was no feature MSE loss active in any of those runs to remove. The 0.5–1.0 mAP-point spread across nb5a/b/c/Full-KD was noise from independent training runs, full stop.

### The fix
Two changes, both required:
1. Add the missing logit-generation cell to nb5a–e, matching nb2/nb3's pattern exactly:
   ```python
   !python scripts/generate_teacher_logits.py --prompt-type box --logits-dir data/teacher_logits_box --dataset data/datasets/crack500_yolo
   ```
   (run once, before any of the ablation training cells — or once per notebook if you're not sharing a persistent Kaggle dataset of pre-computed logits across notebooks, which would be the smarter fix — see §4).
2. Fix the trainer call to actually route through `CrackDistillTrainer` (or explicitly pass `logits_dir="data/teacher_logits_box/"` and a proper `kd_cfg` into `KDSegmentationTrainer` directly), instead of `KDSegmentationTrainer(cfg)` with the raw app config.

**Before trusting any rerun**, check the printed diagnostic lines in the notebook output:
```
[KD] logit files: N        # must be > 0
[KD] ✓ KD losses computed: mask_kd: ..., boundary: ..., feature: ...   # must appear
```
If `logit files: 0` or you never see the "✓ KD losses computed" line, the run is still broken — don't wait until the final mAP table to find out.

---

## 2. Is it "data"? — the honest, deeper answer

Fixing the logits bug is necessary but it's not the whole story. Once real KD signal is flowing again, there are genuine, structural data-quality questions worth chasing — not bugs, but real limitations that would affect results even in a perfectly-executed run:

**a) Box-prompt quality for thin diagonal cracks.** SAM2 is prompted with a bounding box derived from the ground-truth polygon. A crack that's a thin diagonal line has a bounding box whose area is mostly *background* — the fill ratio (mask pixels / box pixels) for a diagonal crack can be extremely low compared to, say, a round object. That means the "teacher" signal for exactly the hardest, most crack-like cases (long, thin, diagonal) is generated from the *least informative* prompt shape. This isn't a code bug, it's a structural mismatch between SAM2's promptable-segmentation design (built for compact, blob-like objects) and crack morphology. Worth measuring directly: compute mean mask-fill-ratio-within-bbox for Crack500 and DeepCrack instances, and check whether SAM2's soft-logit uncertainty (how far from 0/1 the sigmoid output sits) correlates with low fill ratio. If it does, that's a real, publishable data-quality finding, and it suggests trying **rotated/oriented box prompts** or **multi-point prompts along the crack skeleton** instead of axis-aligned boxes — genuinely better-targeted teacher supervision, not just a bigger dataset.

**b) Train/val leakage from cropping (still unverified from earlier).** Crack500 images are dense crops of larger source photos. If train and val crops share a source image, val mAP is inflated by leakage, independent of anything about KD. This directly inflates the apparent "80% cropped→uncropped collapse" that motivates the whole KD-as-regularizer narrative. Check: does the train/val split partition by source-image ID, or by crop, in `convert_crack500.py`? This is a five-minute check with a potentially paper-changing answer.

**c) Instance definition noise.** Instances are derived via `cv2.connectedComponents` on binary masks, meaning a single physical crack that's visually broken by a shadow, stain, or pothole becomes multiple "instances." This inflates instance count and could make the per-instance matching in `_kd_loss_from_preds` (which pairs student predicted instances to SAM logits via `target_gt_idx`) noisier than it needs to be, since adjacent fragments of the same crack compete for very similar predicted regions. Worth a quick sanity check: histogram of instance count per image and instance area — if there's a long tail of tiny fragments, that's adding label noise on top of everything else.

**d) DeepCrack's 80/20 split is script-generated, not the dataset's original split.** If DeepCrack has a canonical published train/val/test split, using a custom 80/20 makes your DeepCrack numbers non-comparable to any other paper's DeepCrack results, which matters if a reviewer tries to sanity-check against prior work.

None of this is "the data is bad" in a fatal sense — Crack500 and DeepCrack are standard, usable benchmarks. But "is data the problem" is the right instinct, specifically around **(a)** — box prompting is very plausibly a genuine ceiling on how good the SAM2 teacher signal can be for this domain, independent of any code bug, and worth investigating before assuming more epochs or better weights will close the gap.

---

## 3. Is it "freezing"? — the honest, deeper answer

Also a fair instinct, and there are real open questions here, separate from the off-by-one head-index bug found earlier:

**a) The freeze/no-freeze comparison is confounded with dataset, not just dataset size.** SegHead freeze is only ever applied to DeepCrack (< 1,000 images) and never to Crack500. So "freeze helps on small datasets" is currently a claim with an n of one dataset. The clean experiment — subsample Crack500 to ~537 images and run the same freeze/no-freeze comparison — has never been done. Without it, you can't distinguish "freeze helps because of sample size" from "freeze helps because DeepCrack specifically has some other property (narrower crack width distribution, higher contrast, whatever) that makes the head more prone to collapse."

**b) The 30%/70% stage split was never swept.** It's a reasonable-looking default, but there's no evidence in the repo that 15/85 or 50/50 wouldn't do better or worse. Given how sensitive the loss weights already are (per your own Optuna findings), it would be surprising if the stage split were *not* also sensitive.

**c) A real, principled concern independent of any bug: capacity mismatch during backbone alignment.** During Stage 1, the feature-MSE loss is pulling a 2.84M-parameter student backbone toward alignment with a 224M-parameter SAM2 encoder's intermediate features. The student simply doesn't have the representational capacity to match SAM2's features everywhere — so forcing exact alignment (via MSE) risks using up the student's limited capacity matching *SAM2's* priorities rather than the task's priorities, only for Stage 2 to then have to partially undo that. This is a known failure mode in feature-distillation literature when teacher/student capacity gaps are large (it's part of why some KD work uses relational or attention-transfer losses instead of raw feature MSE for big capacity gaps). This is worth testing directly, now that real KD signal exists: run Stage 1 with **only** mask_kd + boundary active (feature MSE off) versus the current all-three-losses Stage 1, and see which produces a better Stage-2 starting checkpoint.

**d) Freezing only the head may be freezing the wrong granularity.** Right now it's all-or-nothing on the `Segment` module. A more surgical version — freezing just the mask-prototype branch while leaving box/class heads trainable, or vice versa — has never been tried, and might isolate the actual source of the DeepCrack gradient conflict more precisely than freezing the whole head.

---

## 4. What to actually do next (in order)

### Immediate — fix and reproduce properly
1. Add the missing `generate_teacher_logits.py` invocation to nb5a–e; fix the `KDSegmentationTrainer` call to route through `CrackDistillTrainer` (or pass `logits_dir`/`kd_cfg` explicitly).
2. Before trusting a single mAP number, confirm the `[KD] ✓ KD losses computed` line appears and `logit files: N` is nonzero in each notebook's saved output.
3. **Practical Kaggle tip**: generate teacher logits *once* into a Kaggle Dataset (upload the `.npy`/`.npz` files as a Kaggle input), then have nb5a–e symlink from that input dataset instead of regenerating SAM2 inference every single notebook run. This also protects you from this exact class of bug recurring — a missing symlink step is easier to notice (empty folder, fast failure) than a missing generation step buried in a 150-epoch run.

### Data track (parallel, doesn't block the KD fix)
4. Check Crack500 train/val split for source-image leakage; re-split by source ID if needed.
5. Measure mask-fill-ratio-within-bbox for both datasets; check whether SAM2 soft-logit uncertainty correlates with low fill ratio (thin diagonal cracks). If confirmed, this becomes a real contribution: "box prompting underserves elongated thin objects," with a proposed fix (rotated box or skeleton-point prompts) as future work or an actual experiment if time allows.
6. Sanity-check instance fragmentation (connectedComponents splitting single cracks into multiple instances).

### Freezing track (do after the KD fix, since it needs real signal to mean anything)
7. Run SegHead freeze vs. no-freeze on a **size-matched Crack500 subsample (~537 images)** to separate "sample size" from "DeepCrack-specific" as the explanation.
8. Sweep the Stage 1/Stage 2 split ratio (15/85, 30/70, 50/50) on DeepCrack.
9. Try Stage 1 with feature-MSE disabled (mask_kd + boundary only) vs. the current all-three, to test the capacity-mismatch hypothesis.

### Still open from before, unaffected by this bug
10. Null-teacher control for the SAM2-as-regularizer claim.
11. Seeded reruns (≥3 seeds) once the above is stable — these effect sizes are all in a range where run-to-run noise is a real competing explanation.

---

## 5. One-line status

The nb5a–e ablation results from last time are not just "noisy" — they're measuring nothing, because the notebooks never generated or pointed at real SAM2 logits. That's fixable in an afternoon. What's not a quick fix, and is worth real investigation before assuming a rerun will "just work," are the two things you already suspected: box-prompt quality on thin diagonal cracks (a data-side ceiling) and whether the freeze schedule is tuned for DeepCrack's sample size or DeepCrack's domain specifically (a freezing-side confound). Fix the bug first — it's blocking everything — then chase those two, in that order.
