# 🎯 Possible OOD Improvements (`possibleOODimprovements.md`)

Consolidated findings on why the student generalizes poorly from cropped Crack500 training to full-resolution/uncropped evaluation, and concrete next steps. My own findings (verified directly against the repo's files, not assumptions) are listed first since they change the picture the rest of this plan was built on; the broader literature-derived recommendations and the cross-checked external (Gemini) plan follow.

---

## 🔍 1. My Findings — Verified Against the Actual Data

### 1.1 The training set has ZERO background-only tiles — likely the dominant OOD driver
Checked directly: `data/datasets/crack500_yolo/labels/train/` has **1896 label files, 0 empty ones**. Every single training crop contains at least one crack instance.

**Why this happened**: Crack500's authors tiled each source photo into a 4×4 grid (640×360 tiles) and shipped only the 5–7 tiles per photo that actually contained a crack — background-only tiles were discarded before the dataset was ever released (confirmed by parsing the `{stem}_{x}_{y}.jpg` offset pattern in `traincrop/`: 250 unique source photos, each missing 9–11 of its 16 possible grid positions).

**Why it matters**: Tiled inference (notebook 07) slides a window across the *entire* full-resolution photo — mostly plain, crack-free asphalt. The model has never once been trained to output "nothing here" on a real road surface. This is a more direct explanation for tiled Dice sitting at ~0.27 (vs. 0.53+ in-domain) than pixel-scale collapse alone: the failure mode is likely **false positives / unstable behavior on empty tiles**, not just missed thin cracks.

**Action**: Mine or label genuine background-only 640×360 crops and add them (with empty label files) to `crack500_yolo/images/train`.

**Update — tested**: built and ran `scripts/mine_negative_and_mosaic_tiles.py` against the real data. Only **5 of 250 train-split source photos** have an exact-filename match in `valdata`/`testdata` (their full uncropped photo), yielding just **10 background-only crops** — and even those come from photos that overlap the OOD eval set, so they're not clean training additions. Checked DeepCrack too (`data/datasets/deepcrack/train_lab/*.png`): **0 of 300 masks are background-only** — also a curated crack-only dataset. **Conclusion: negative-tile mining from data already in this repo doesn't scale.** Getting real negative tiles requires external data (new road photos, or a public negative/background set) — deprioritized until that's sourced. Downgraded from "highest-confidence fix" to "blocked pending new data."

### 1.2 Correction: native crop resolution is 640×360, not 512×512
Checked directly: `data/datasets/crack500_yolo/images/train/*.jpg` are **640×360 pixels on disk**. "512" was never a real resolution — it's only ever been the Ultralytics `imgsz` resize target. Every run at `imgsz=512` (notebooks 01, 03, 04, 05, 06, 09) was **downsampling** the true 640×360 source before training even started.

**Implication for notebook 10** (`imgsz=768`): this recovers detail the 512-imgsz runs were throwing away (mild ~1.2× upsample of 640×360, not a genuine resolution increase), but it **cannot exceed the 640×360 ceiling already baked into the shipped files**. It's a real, if smaller-than-assumed, improvement — not the "genuinely higher native resolution" fix the literature recommends. That fix requires actual higher-resolution source pixels, which the current dataset doesn't provide beyond 640×360.

### 1.3 The "shrinks below 1px" story only applies to direct-resize eval, not tiled eval
Tiled inference (notebook 07) extracts true native-pixel 512×512 windows directly from the full-resolution photo — no resize, no shrinkage. The "cracks disappear below 1-2px" mechanism is real but specifically explains the **direct-resize OOD numbers** (mAP50 ~0.08–0.11); it does not by itself explain why *tiled* Dice (~0.27) still trails in-domain (~0.53+) so heavily. §1.1's negative-tile gap is a more direct candidate for that remaining shortfall.

### 1.4 "Corrupt JPEG restored and saved" warnings are a false lead
Every uncropped val/test image triggers this Ultralytics warning during every eval run. Verified with `PIL.Image.verify()` — the files are not actually corrupt (routine EXIF/marker handling, cosmetic warning). Ruled out as a confound; not worth further investigation.

### 1.5 Free infrastructure: the tile-offset grid unlocks three things from data you already have
Because `traincrop/` filenames encode `{stem}_{x}_{y}` grid offsets (640×360 stride, 250 source stems), a single stitching script can serve:
- **Negative-tile mining** (§1.1) — identify unused grid offsets, extract those regions as background-only crops.
- **Native-scale SAM2 teacher logits** — stitch contiguous surviving tiles into larger real composites (e.g. 1280×720), run `generate_teacher_logits.py` against them for teacher supervision beyond the 640×360 ceiling.
- **Scale-diverse training data** — sample variable-size crops from those composites instead of only fixed 640×360, without any new data collection or labeling.

**Update — built and tested**: `scripts/mine_negative_and_mosaic_tiles.py` (mosaic stitching) + `scripts/build_augmented_training_set.py` (merges composites into a new `crack500_yolo_augmented` training set, converting mosaic masks to YOLO polygons via the existing `binary_mask_to_yolo_instances()` from `convert_crack500.py`) both exist and run clean:
- **Mosaic stitching: 250/250 source photos succeeded**, composites up to 1920×720 (genuinely beyond the 640×360 ceiling) — this part of the plan works as hoped.
- **Negative-tile mining: only 10 crops from 5 stems** (see §1.1 update) — does not scale, deprioritized.
- Kaggle notebook suite built in `OODimprovements/` (see that folder's `README.md`) implementing mosaic stitching → native-scale teacher logits → retraining → eval, chained via Kaggle "Notebook Output Files" the same way the rest of this project's notebooks chain.

Real bug caught while building this: the crop grid isn't a fixed 640×360 everywhere — Crack500 mixes landscape and portrait source photos, so tile pixel dimensions had to be detected per-stem rather than assumed as a global constant. Fixed in the script; worth remembering if extending it further.

---

## 📚 2. Literature-Derived Recommendations (Broader Research Pass)

### Training-time
| Method | Mechanism | Cost |
|---|---|---|
| Train at genuinely higher native resolution | 2025 crack-segmentation study: 1024×1024 training gave best cross-scale generalization (89.64% mIoU) vs. lower res ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2352492825012395)) — requires real higher-res source pixels (see §1.5) | Medium |
| Ultralytics `scale` / `multi_scale=True` augmentation | Random zoom exposes the model to varied apparent crack widths per epoch | Very low |
| Negative/background training examples | See §1.1 | Low–Medium |

### Architectural
| Method | Mechanism | Cost |
|---|---|---|
| clDice / Skeleton Recall Loss | Penalizes broken crack connectivity directly (skeleton-space Dice), not just pixel overlap ([arXiv 2404.03010](https://arxiv.org/html/2404.03010v1)) | Medium |
| High-res auxiliary branch (CDEM-style) | Preserves fine edge detail lost to backbone downsampling stride | High |
| Foreground-dilated mask-KL (already implemented) | Confirmed best raw-OOD variant (notebook 03, 0.1007 mAP50) | Already done |

### Test-time
| Method | Mechanism | Cost | Status |
|---|---|---|---|
| Gaussian-weighted tile blending | Replaces flat overlap-averaging with center-weighted blending (nnU-Net/Cellpose standard) to kill hard tile-boundary artifacts | Low | ✅ **Done** (commit `36994ab`) |
| SAHI-style formalized slicing | Tuned NMS-based merge vs. manual max/average | Low–Medium | Not started |
| Multi-scale TTA | Predict at 512 + 768 tile size, average | Low (2× inference cost) | Not started |
| Super-resolution pre-pass | Upsample before tiling | Medium | Lower priority — least crack-specific evidence |

### Teacher-side (SAM2)
Confirmed via `scripts/generate_teacher_logits.py`: the SAM2 teacher has **only ever seen the same 640×360 (nominally "512") training crops** — it has no native-resolution knowledge to distill in the first place. Native-scale teacher logits (§1.5) is the fix; not yet built.

---

## ✅ 3. Cross-Checked External Plan (Gemini) — Status

| # | Item | Status |
|---|---|---|
| P0 | Gaussian tiled sliding-window (25% overlap) | ✅ Done (commit `36994ab`) |
| P1 | Combine `06_layerkd` + `03_dilated` | ✅ Done — merged into notebook 10 alongside P2 |
| P2 | Scale-diverse training jitter | ⚠️ Partial — notebook 10 sets `imgsz=768`, but per §1.2 this only recovers the existing 640×360 ceiling, it doesn't add genuine multi-scale diversity or exceed native resolution |
| P3 | Distill SAM2 on uncropped wide-shot images | ❌ Not feasible as literally stated — Crack500 ships **no full-resolution train-split images** (`traincrop/` only; `traindata/` doesn't exist). Real path is the stitching script from §1.5 |

**Currently running**: `10_run_layerkd_dilated_hires.ipynb` (768px, CWD layer-KD + foreground-dilated mask-KL, 150 epochs, ~3–3.5 hrs on Kaggle). Worth finishing and evaluating, but should not be treated as a full resolution fix given §1.2 — check whether its error mode shifts from missed-thin-cracks toward false-positives-on-background, which would support prioritizing §1.1 next.

---

## 📋 4. Prioritized Action Plan

1. **Let notebook 10 finish and evaluate with notebook 07** (tiled Dice + direct mAP50). Inspect qualitative failure mode: thin cracks still missed, or false positives on background regions?
2. ✅ **Built**: `scripts/mine_negative_and_mosaic_tiles.py` + `scripts/build_augmented_training_set.py` — mosaic stitching works (250/250), negative-tile mining doesn't scale (10 crops, deprioritized per §1.1 update).
3. ~~Mine negative (background-only) training tiles~~ — tested, doesn't scale from data already in this repo. Needs external data if pursued further.
4. ✅ **Built**: `OODimprovements/02_generate_native_teacher_logits.ipynb` — generates native-scale SAM2 teacher logits from the mosaic composites (§1.5 / former P3).
5. ✅ **Built**: `OODimprovements/03_run_mosaic_native_kd.ipynb` — retrains the CWD + foreground-dilated recipe from notebook 10 on the mosaic-augmented set (`imgsz=640`, matching native tile width rather than upsampling), with merged crop-scale + native-scale teacher logits. **Not yet run on Kaggle.**
6. **Re-run notebook 07-style eval (`OODimprovements/04`) across all checkpoints** on the same fresh cross-checkpoint protocol (don't trust hardcoded tables) and update `final_results_exp.md`.

See `OODimprovements/README.md` for the full run order and Kaggle attachment instructions.
