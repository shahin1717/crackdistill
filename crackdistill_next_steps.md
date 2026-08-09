# Crack-Distill: What's Next — Self-Critique & Action Plan

*A research audit written by grilling the project with the hardest questions I could ask, then answering them honestly.*

---

## 0. Honest inventory — where the project actually stands

| Component | Status | Evidence |
|---|---|---|
| nb1 (YOLOv8 vs v11 baseline) | ✅ Executed, numbers verified against docs | Real Ultralytics logs in `_runned.ipynb` |
| nb2 (Crack500 KD, box vs box+centroid) | ✅ Executed | Real logs |
| nb3 (DeepCrack KD, SegHead freeze) | ✅ Executed | Real logs, matches `nb_exp_results.md` |
| nb4 (Cross-dataset generalization) | ✅ Executed | Real logs |
| nb5a–e (loss-term ablations) | ❌ **Never run** — 0 output cells across all 5 notebooks | Checked every cell's `execution_count` |
| `run_on_kaggle_final_rauf.ipynb` (source of the headline EXP-08 "+10.3% OOD" claim) | ❌ **Never run** — 0 output cells | Same check |
| Local CLI reproduction (`run_experiments.py`, `tune_kd_weights.py`) | ❌ Broken — imports `distillation/trainer.py`, which only exists inline inside notebook cells, never saved to the shared repo folder | `distillation/` only contains `kd_trainer.py` |
| YOLO26 compatibility | ❌ Two silent-failure bugs found (criterion unwrapping, head class-name match) | Verified against live `ultralytics==8.4.116` |
| Statistical rigor | ❌ Every result is a single run, no seeds, no error bars | No seed-sweep code anywhere in repo |
| Train/val leakage check (Crack500 crops) | ❓ Unknown — never verified | Not addressed in any script |

**Bottom line: the strongest, best-isolated result in the whole project (SegHead freeze, EXP-18) is solid. The two flashiest claims (SAM2-as-regularizer +10.3% OOD, and the entire ablation study) currently rest on either a single unverified run or literally zero runs.**

---

## 1. Grilling the project — hard questions, honest answers

### Q1: If a reviewer asked "how do you know the OOD improvement is from SAM2's knowledge and not just from adding *any* extra loss term," what would I say?
**Honest answer: I couldn't defend it.** There's no null-teacher control (e.g., KD against random logits of the same shape/temperature) and no ablation showing which specific loss term drives OOD gains. This is exactly what nb5a–e were designed to test — and they've never been run. This is not a "nice to have," it's a hole in the core claim.

### Q2: If a reviewer asked "is 0.352 vs 0.319 OOD mAP50-95 actually different, or is that within run-to-run noise?" — what's my answer?
**I don't know, because there's only one run of each.** On numbers this small, a couple hundredths of a point is a plausible amount of seed variance for YOLO training. Until there are ≥3 seeds per key config, "+10.3%" is a point estimate with no confidence interval, not a finding.

### Q3: Is the in-domain val set actually a fair estimate of generalization, or is it leaking?
**Unverified — and this could quietly undermine everything.** Crack500 images are crops of larger source photographs. If train and val crops came from the same source image (adjacent tiles), val mAP is inflated by leakage that has nothing to do with KD. That would also *artificially inflate* the apparent "80% cropped→uncropped collapse," making the KD-as-regularizer story look more dramatic than it is. This needs a one-line check (split by source-image ID, not by crop) before trusting anything built on top of the OOD-drop narrative.

### Q4: Am I sure the "+10.3% OOD" number is even real, i.e. did that training run happen as described?
**No.** `run_on_kaggle_final_rauf.ipynb` — the notebook literally cited as the source of EXP-08 — has zero saved outputs. Either the notebook was cleared before committing, or the number came from somewhere else (screenshot, separate log, memory) that isn't in the repo. Right now the single most-quoted number in the project has no artifact backing it.

### Q5: Does the DeepCrack→Crack500 "collapse" tell me anything beyond "537 images isn't enough data, period"?
**Probably not yet.** Mask mAP50 of 0.028–0.033 is near a degenerate/near-chance detector for a single-class task. I don't have a same-size Crack500 subsample (~537 images) run to check whether the asymmetry is about *domain diversity* or just *sample size*. Without that control, "Crack500's domain is richer" is a plausible story, not a demonstrated one.

### Q6: If I swapped the student to YOLO26 today, would the results be trustworthy?
**No — silently wrong, not loudly broken.** Two concrete bugs, both verified against a live YOLO26 model:
- `model.criterion` under YOLO26's default end-to-end mode is an `E2ELoss` wrapper, not `v8SegmentationLoss`. The KD trainer calls `.parse_output()` / `.get_assigned_targets_and_loss()` directly on it — these don't exist on `E2ELoss`. The broad `except Exception` swallows this, prints one warning, and **silently drops mask_kd + boundary loss for the entire run**, leaving only feature-MSE KD active. You'd think you ran full KD; you'd actually have run ~1/3 of it.
- The SegHead-freeze logic matches the head by `type(module).__name__ == "Segment"`. YOLO26's head class is `Segment26`. The match never fires, so `head_idx` silently stays at its hardcoded fallback (which points at the last neck block, not the head) — meaning the exact fix that rescued DeepCrack (+3.98%) would be freezing the wrong layer entirely if run today on YOLO26.
Both are cheap to fix, but **must be fixed and spot-checked (print a loss breakdown for the first 3 batches) before trusting any YOLO26 KD run.**

### Q7: Is the Optuna search (10–12 trials, 15 epochs) actually good enough to claim these are "the" optimal weights?
**No, and the project's own docs already say so.** A 4-dimensional continuous space (τ, α, β, γ) with 10–12 trials is a coarse grid at best. The weights are explicitly flagged as "highly sensitive." This is a known, acknowledged gap — just not yet acted on.

### Q8: Am I confident the 30%/70% stage split is a good choice, or is it just "the first thing that worked"?
**The latter, as far as I can tell.** No sweep over the split ratio exists anywhere in the repo. It's plausible 30/70 is fine, but there's zero evidence it's better than, say, 15/85 or 50/50.

### Q9: Does the SegHead-freeze finding generalize beyond DeepCrack, or is it a DeepCrack-specific artifact?
**Untested, but cheaply testable.** `data.train_fraction` is already implemented in `dataset.py` and wired into `run_experiments.py` (`low_data_5/10/25/50pct`), but never run. Running SegHead-freeze at those fractions on Crack500 would tell me whether the effect is about *dataset size* (general, more citable claim) or *DeepCrack specifically* (narrower claim).

### Q10: If I had to bet, which of the four "successes" would survive a hostile re-review?
1. **SegHead freeze** — survives. Clean before/after, plausible mechanism, matches theory.
2. **YOLOv11 > YOLOv8** — survives. Controlled, non-confounded (smaller AND better).
3. **Train Crack500→test DeepCrack is the right direction** — survives in *direction*, but the *explanation* (domain diversity vs. sample size) is unconfirmed.
4. **SAM2 KD as OOD regularizer (+10.3%)** — does **not** currently survive. No null-teacher control, no seeds, no confirmed source notebook, and the leakage question is open. This is the shakiest of the four flagship claims despite being the one being led with.

---

## 2. Prioritized action plan

Legend: 🔴 P0 blocks the paper's core claims · 🟡 P1 strengthens claims · 🟢 P2 nice-to-have / scope-widening

### 🔴 P0 — Fixes and runs that determine whether the current claims hold up

1. **Run nb5a → nb5d → nb5e in sequence on Kaggle.** This is the single highest-value thing left to do — it's the only way to know which loss term actually drives the OOD gain, and it's already written, just unexecuted.
2. **Re-run and save `run_on_kaggle_final_rauf.ipynb`.** The headline number needs an artifact. If the re-run doesn't reproduce ~0.352 OOD mAP50-95, that's important to know now, not after submission.
3. **Add a null-teacher control.** Clone the full-KD config, replace SAM2 logits with random noise of the same shape/temperature, re-run EXP-08's setup. If OOD still improves similarly, the "SAM2 knowledge" story is wrong and needs to be reframed as "any auxiliary regularizer helps."
4. **Check Crack500 train/val split for source-image leakage.** If crops from the same source image appear in both train and val, re-split by source ID and re-run the baseline (EXP-04) and full-KD (EXP-08) numbers — the whole "80% collapse" narrative depends on this being clean.
5. **3-seed reruns of the four flagship comparisons** (EXP-04 vs EXP-08, EXP-17 vs EXP-18, EXP-11 vs EXP-12, EXP-20 vs EXP-21). Report mean ± std. This is expensive but non-negotiable if the paper wants to claim any of these deltas are real effects rather than noise — several are within a few hundredths of mAP.

### 🟡 P1 — Strengthens claims that are directionally fine but under-evidenced

6. **Same-size Crack500 subsample (~537 images) cross-eval control**, to separate "domain diversity" from "dataset size" as the explanation for the Crack500↔DeepCrack asymmetry.
7. **Widen the Optuna search**: bump to 30–50 trials, enable TPE pruning (already supported by the `optuna` package used), and consider re-running it *after* the null-teacher control confirms the loss terms are doing something real — no point precision-tuning weights for a mechanism that hasn't been validated yet.
8. **Sweep the Stage 1/Stage 2 split ratio** (e.g. 15/85, 30/70, 50/50) on DeepCrack to check whether 30/70 is actually a good choice or just the first one tried.
9. **Run SegHead-freeze at `low_data_5/10/25/50pct` on Crack500** — already-implemented code path, currently unused — to test whether the freeze effect is about dataset size generally (stronger, more citable claim) or DeepCrack specifically.
10. **Fix and spot-check YOLO26 compatibility** before any YOLO26 KD run: unwrap `E2ELoss` → `.one2one` for `parse_output`/`get_assigned_targets_and_loss`, and fix the head-name match to `"Segment" in type(module).__name__` (or an `isinstance` check against the base `Segment` class). After patching, print the per-loss breakdown (`mask_kd`, `feature`, `boundary`) for the first few batches to confirm all three are actually non-zero — don't trust a clean-looking loss curve alone.

### 🟢 P2 — Scope-widening, do after P0/P1 or if time allows

11. Fix `distillation/trainer.py` not being saved to the shared repo (it currently only exists inline inside notebook `%%writefile` cells) — mostly a hygiene issue since Kaggle runs don't need it, but worth doing before anyone tries to reproduce this off-Kaggle.
12. Revisit combined-dataset (Crack500+DeepCrack) training with domain-aware batch sampling instead of naive concatenation — currently abandoned after the EXP-06 NaN crash, and the crash itself was likely conflated with the (now-fixed) AMP/clamping issue rather than being a fundamental blocker.
13. Try YOLO26 as the student architecture for a full baseline-vs-KD comparison (after the P1 #10 fix), and report FLOPs/params/speed alongside v11 — directly relevant given the paper's framing around "compact real-time model for edge deployment."
14. Add own hardware speed benchmarking (SAM2 vs YOLOv11n-seg vs YOLO26n-seg, same GPU, same images) instead of citing official benchmarks for the "4x faster" claim.

---

## 3. Definition of done (before calling this submission-ready)

- [ ] nb5a–e executed with saved outputs; ablation table populated with real numbers
- [ ] `run_on_kaggle_final_rauf.ipynb` re-run and saved, or headline number sourced from wherever it actually came from
- [ ] Null-teacher control run and reported (even if it complicates the story — better to know now)
- [ ] Crack500 split checked for leakage; re-run affected experiments if leakage found
- [ ] Flagship deltas reported as mean ± std over ≥3 seeds, not single runs
- [ ] YOLO26 compatibility patched and verified via per-loss-term logging, if YOLO26 results are going in the paper
- [ ] Every claim in `research_analysis.md` / `full_review.md` cross-checked against which of the above it actually depends on — anything resting only on a single unverified run gets downgraded from "finding" to "preliminary observation" in the writeup until it's backed
