# CrackDistill — experiment plan (from 2026-09-28)

This file replaces the old audit reports, leaderboards and "next moves" notes. Anything not listed here
is either done or dropped.

## 1. Where things stand

**Every KD result before 2026-09-28 is invalid.** The training code had two bugs that affected every KD run:

| Bug | Effect | Fixed in |
| :--- | :--- | :--- |
| Training batches carry `ratio_pad` as `(h_ratio, w_ratio)` with no padding entry | Teacher placed ~140 source px off the student canvas | `e6293db` |
| `overlap_mask=True` re-sorts batch instances by area, teacher logits stay in label order | On the 45% of images with several cracks, only 42% of instances got the right teacher | `e6293db` |

Evaluation had bugs too: EXIF-rotated test photos were scored against unrotated masks (130/200),
`.JPG` files were skipped, the "batched" throughput was the serial time × 0.35, and the GT-soft
control notebook actually trained on SAM logits. All are fixed on this branch.

**Verified on real data after the fixes** (`reports/alignment_verification.json`, training transforms, 969 instances):

| Check | Result |
| :--- | :--- |
| SAM teacher vs GT, median IoU (warped / plain resize) | 0.605 / 0.000 |
| Same, with the trainer's `teacher[k]` pairing | 0.604 |
| GT-soft target vs GT, median IoU | 0.796 (paired 0.796) |
| Shift scan: IoU peak within 1 px of zero offset | 66.7% of images, mean offset (0.01, −0.31) px |
| 2-epoch smoke runs, baseline and Mask-KD | exit 0; KD 0 errors, 0/63,470 instances dropped |
| Local 1-epoch run of the generated notebooks | all three arms complete, results JSON written |

## 2. What is kept and what is dropped

Tuned-baseline re-benchmarks show most dense-KD extras don't survive ([Niemann et al. 2023](https://arxiv.org/abs/2309.03659),
[Ali et al. 2026](https://arxiv.org/abs/2604.25530)). Stacked terms add ~0.5 mIoU, about seed noise
([CIRKD](https://arxiv.org/abs/2204.06986)). The old 0.014 spread between variants came from a misaligned teacher and one seed.

- **Method:** plain Bernoulli mask-KL to cached SAM 2 box-prompt logits (T = 3.7769, W = 0.9612).
- **Dropped, no valid evidence:** foreground-dilated KL, focal KL, pixel affinity, multi-scale 512, neck
  CWD/LayerKD, boundary BCE, Tversky, resolution-preserving KL, two-stage tuning, mosaic-native
  teacher, centroid prompts. They may appear only as "explored before the fix, 1 seed, not comparable".
  The loss code stays in `kd_trainer.py` behind config flags; nothing runs it.
- **Controls, not variants:**
  - **GT-soft (SVLS-style):** the same KL loss with Gaussian-blurred GT targets. Needed because KD
    can act as learned label smoothing ([Yuan et al. 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.html);
    SVLS, [Islam & Glocker 2021](https://arxiv.org/abs/2104.05788)).
  - **Binarized SAM masks** (Tier 2): separates the shape of SAM's masks from their softness.
- **Tiled Gaussian inference** is standard practice (nnU-Net, MONAI, [SAHI](https://arxiv.org/abs/2202.06934)).
  Report it as engineering, with a tiled-vs-direct ablation on the 200 test photos, not as a contribution.
  With the fixed evaluator its gain on a 2-epoch model was only +0.03 Dice, so the old "+49%" must be re-measured.

## 3. Fixed in advance (do not change after seeing results)

- T and W are kept from the old Optuna search. They were tuned on the buggy pipeline, but re-tuning
  would use the val set again.
- **GT-soft targets:** σ = 2.0, logits capped at ±14. The cap is SAM's median background logit; the old
  cap of ±9.2 put 8% crack probability on every background pixel. With the cap, σ = 2.0 matches SAM's
  near-crack entropy at T: 0.70 vs 0.66 bits over 440 train instances.
- Every arm uses the same seeds (0–4), the same 150 epochs and the same zeroed spatial augmentation,
  and keeps `best.pt` by Ultralytics val fitness.
- **Primary metric:** mask mAP50 on the canonical test split (1,124 crops), evaluated **once**, after all training.
- Ultralytics is pinned to 8.4.60, the version the fixes were verified against.

## 4. Runs

| Tier | Arm | Seeds | Notebooks | T4-hours |
| :--- | :--- | :--- | :--- | :---: |
| 1 | A: Baseline | 0–4 | `1_baseline_seed{0..4}` | 15 |
| 1 | B: Mask-KD | 0–4 | `2_mask_kd_seed{0..4}` | 15 |
| 1 | C: GT-soft | 0–2 | `3_gt_soft_seed{0..2}` | 9 |
| 2* | C: GT-soft | 3–4 | `3_gt_soft_seed{3,4}` | 6 |
| 2* | D: Binarized SAM targets | 0–4 | not generated yet | 15 |
| — | Speed | — | `4_benchmark_speed` | <0.1 |

\* Run Tier 2 only if the B − A CI excludes 0. See `final_notebooks/README.md` for how to run on Kaggle.

**Power:** detecting Δ = 0.01 mAP needs about 4 seeds per arm if the seed SD is 0.004, and about 12
if it is 0.008. Five seeds give a minimum detectable effect of ~0.008–0.016. Report the SD of arm A
first; if it is large, the answer is "no detectable effect at this budget", not a win.

## 5. Analysis

1. Local test evaluation, once per checkpoint:
   `python scripts/evaluate_canonical_test_set.py --weights <best.pt> --arm <a> --seed <s>` → `results/test/`
2. Aggregate, paired by seed:
   `python scripts/aggregate_multiseed_results.py --results-dir results/test --metric in_domain_mask_mAP50`
   This prints each arm's mean ± CI and, for B vs A and B vs C, the per-seed difference with a t-CI,
   Cohen's d, a paired t-test and a Wilcoxon test.
3. **Still to build** (needs only the checkpoints):
   - per-parent scores and a hierarchical bootstrap: resample seeds, then the 200 parent photos
     (the 1,124 crops are not independent) ([Bouthillier et al.](https://arxiv.org/abs/2103.03098), [Dror et al.](https://aclanthology.org/P18-1128/))
   - Holm correction over B vs A and B vs C
   - crack metrics reviewers expect: union-mask IoU, ODS/OIS ([FPHBN protocol](https://arxiv.org/abs/1901.06340)), clDice ([Shit et al.](https://arxiv.org/abs/2003.07311))
   - student–teacher agreement
4. Speed: single-tile and measured full-scene latency from `4_benchmark_speed` on a T4.

## 6. How to read the outcome

| Outcome | Claim |
| :--- | :--- |
| B > A (CI excludes 0) and B > C | SAM 2 adds knowledge beyond softened labels |
| B > A, B ≈ C | The gain is label smoothing; SAM is not needed for it |
| B ≈ A | No detectable KD effect at this budget; report as a negative result |
| B < A | The teacher hurts. SAM is known to be weak on thin structures ([SAM](https://arxiv.org/abs/2304.02643), [WACV 2026](https://arxiv.org/abs/2412.04243)); the teacher scores only 0.605 median IoU vs GT |

Novelty: no prior work was found that distils box-prompted SAM 2 logits into YOLO-seg prototype
masks for cracks. That was not a systematic search, so describe it as a combination of known parts.
