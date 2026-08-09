# Crack-Distill: Updated Next Steps — Post nb5e Execution

*Follow-up to the earlier plan. nb5e actually ran (Aug 4, 2026) since then — this updates the audit with what the real data says, including a bug it exposed and one finding that pushes back on the paper's core narrative.*

---

## 0. What changed since the last plan

| Item from the old P0 list | Status now |
|---|---|
| Run nb5a → nb5e | ✅ Done — `nb5e-ablation-results-summary_runned.ipynb` executed Aug 4 |
| Re-run/save `run_on_kaggle_final_rauf.ipynb` | ❓ Still not confirmed — not addressed by this run |
| Null-teacher control | ❌ Still not done |
| Crack500 leakage check | ❌ Still not done |
| Seeded reruns | ❌ Still not done |
| YOLO26 compatibility fixes | ❌ Still not done |

So: real progress, but the run that happened surfaced **a new bug** and **a result that contradicts the paper's headline framing** — both need to be dealt with before anything from this run goes in a writeup.

---

## 1. What the new data actually says — read carefully, not optimistically

### 1.1 There's a checkpoint collision bug, and the doc itself admits it
`reviews-analysis2.md` is upfront about this: **"Full KD (Box Prompts)"** and **"Ablation 4: Full SegHead Freeze"** both resolved to the exact same file — the DeepCrack Stage-2 frozen-head checkpoint from nb3, not their own checkpoints. Root cause stated: the resolver does a keyword search and falls back to "first matching `.pt`" when it can't find an exact path. Both of those table rows (0.1614 / 0.0554 / 0.2532 / 0.1548) are noise right now — they say something about a DeepCrack checkpoint evaluated out of domain, nothing about box-prompt KD or permanent-freeze KD.

**Don't stop at "noted the bug."** The resolver is a *keyword-search-with-fallback* — that's exactly the kind of bug that fails silently on some entries and not others. nb5a/nb5b/nb5c are marked "✅ Valid Run" in the doc, but that's an assertion, not a verification. Before trusting those three numbers:
- Print the **full resolved path** for every one of the 7 variants, not just the collided two.
- Check file **size/mtime** against the actual Kaggle run folder for each experiment — a wrong-but-plausible-looking checkpoint (e.g. two runs with similar names) wouldn't necessarily raise an error, just give you a subtly wrong number.

### 1.2 The baseline row isn't a valid baseline
Look at the checkpoint path for `"Baseline (No KD Fine-tune)"`: it's `.../nb3-deepcrack/.../baseline_finetune.../best.pt` — **a model trained on DeepCrack**, evaluated on the *combined* 408-image validation set (which is mostly Crack500-domain images). That's not "baseline vs KD" — that's a cross-domain generalization data point wearing a baseline label. It happens to reproduce the known DeepCrack→other-domain collapse pattern (0.1883 is in the same ballpark as EXP-23's 0.0276-ish collapse numbers, scaled up because the combined set isn't pure Crack500).

This matters a lot: as currently labeled, the table implies **"Full KD (0.4818) beats Baseline (0.1883) by +156%,"** which is not a real, controlled comparison — it's KD-on-Crack500-checkpoint vs a DeepCrack checkpoint tested out of domain. If this table (or a number derived from it) goes in the paper as-is, it's a serious, easy-to-catch error.

### 1.3 The genuinely interesting (and uncomfortable) finding, in the *valid* rows
Restricting to the three ablation rows that share the same eval protocol and aren't part of the collision bug — nb5a, nb5b, nb5c, compared against the Full KD (Box+Centroid) reference (0.4818 Mask mAP50):

| Removed term | Mask mAP50 | Δ vs Full KD |
|---|---|---|
| Mask KL (nb5a) | 0.4762 | **−0.56 pts** (hurts, as hypothesized) |
| Feature MSE (nb5b) | **0.4921** | **+1.03 pts** (removing it *helps*) |
| Boundary BCE (nb5c) | 0.4897 | **+0.79 pts** (removing it *helps*) |

`reviews-analysis2.md`'s own conclusion already says the quiet part: *"Feature MSE can over-constrain student feature maps."* Read plainly, this table says **your best-performing configuration on this eval set isn't the 3-loss "Full KD" — it's task loss + mask-KL alone, with feature-MSE and boundary-BCE removed.** That directly contradicts `research_analysis.md`'s framing that the full 3-loss combination is the method ("SUCCESS 1/2/3" all implicitly assume all three terms are pulling their weight).

Before treating this as a real finding, two caveats worth resolving, not skipping:
- **Interaction effects**: the ablations reuse the *jointly*-tuned Optuna weights (α/β/γ) with one term zeroed out, rather than re-tuning weights per-ablation. Zeroing β (feature) while keeping α and γ at values that were only optimal *given* β's presence is not a clean ablation — the "feature MSE hurts" conclusion could really be "feature MSE's weight, un-recalibrated after removing the interaction, is now poorly scaled." This needs a mini re-tune (even a handful of Optuna trials) per ablation arm before it's a solid claim.
- **No one has tested the 2-loss config that this data implies is best** (task + mask_kd only, no feature, no boundary). It's the obvious next experiment, and it's cheap — one Kaggle run with `feature.enabled: false` and `boundary.enabled: false` simultaneously.

### 1.4 Eval protocol mismatch across the project
nb5e evaluates on the **combined 408-image val set**. EXP-04/EXP-08 (the flagship OOD claim) evaluate on **Crack500's own cropped/uncropped splits**. nb1–nb4 evaluate on **per-dataset splits** (Crack500: 348 img, DeepCrack: 60 img). None of these numbers are on the same footing. Right now the project has three different validation protocols in play across its own tables, which makes cross-referencing "EXP-08: 0.515 Mask mAP50" against "nb5e Full KD: 0.4818 Mask mAP50" meaningless even though they sound like they're measuring the same thing.

---

## 2. Updated priority action plan

### 🔴 P0 — Before any number from nb5e goes near the paper

1. **Fix the nb5e checkpoint resolver.** Make it fail loudly (raise, not silently fall back) when it can't find an exact match, or require an explicit run-name → path mapping instead of keyword search. Re-run the resolution step and confirm all 7 paths by hand.
2. **Re-run "Full KD (Box Prompts)" and "Ablation 4 (Full SegHead Freeze)"** with the correct checkpoints once the resolver is fixed — they currently have zero valid data.
3. **Get a real baseline.** Train (or re-evaluate an existing) Crack500-only, no-KD model and evaluate it on the *same* combined val set nb5e uses. The current baseline row needs to be replaced, not footnoted.
4. **Spot-verify nb5a/nb5b/nb5c checkpoints** (path + size/mtime cross-check against the actual Kaggle run directories) before trusting them as clean — the resolver bug was only caught because two rows collided into an obviously-identical result; a single wrong-but-different checkpoint wouldn't be caught that way.
5. **Run the "task + mask_kd only" config** (feature and boundary both disabled) on the same protocol. If it beats 0.4921 (current best), that's your actual best configuration and the paper's method section needs to change — better to find this now than after writing "our 3-loss combination is optimal."
6. **Pick one canonical eval protocol** for the whole project (combined val, or per-dataset val — pick one) and re-run the flagship EXP-04/EXP-08 numbers under it so the ablation table and headline results are directly comparable in the same document.

### 🟡 P1 — Needed to trust whatever P0 lands on

7. **Re-tune loss weights per ablation arm** (small Optuna budget, e.g. 5–8 trials) rather than reusing the jointly-tuned weights with one term zeroed — this separates "the term is genuinely unhelpful" from "the term's weight is miscalibrated once another term is removed."
8. **Seed the corrected comparisons** (≥3 seeds) once P0 gives a trustworthy single-run baseline — these deltas are in the 0.5–1.0 mAP-point range, similar magnitude to what's plausible from run-to-run noise.
9. Carry forward from the previous plan, still unresolved: **null-teacher control** for the SAM2-as-regularizer claim, **Crack500 train/val leakage check**, **YOLO26 criterion/head-freeze fixes** if YOLO26 is still on the roadmap.

### 🟢 P2

10. If "task + mask_kd only" does turn out to be the winner, consider whether feature-MSE and boundary-BCE are worth keeping in the framework at all, or whether they become optional/future-work components rather than core contributions — this changes how the abstract should be written.
11. Once the resolver is fixed, consider hardening it as reusable infra (explicit `{experiment_name: checkpoint_path}` mapping saved as JSON at training time) so this class of bug can't recur across future notebooks.

---

## 3. One-line summary if someone asks "what's the status"

nb5e ran, which is real progress — but it shipped with a checkpoint-resolution bug that invalidates 2 of its 7 rows, an invalid baseline that invalidates the headline comparison, and (in the rows that *are* trustworthy) a result that quietly contradicts the paper's current "3-loss KD is best" framing. None of this is fatal — it's exactly the kind of thing an ablation study is supposed to catch before submission — but it means the ablation table isn't paper-ready yet, and the likely real finding (mask-KL matters, feature/boundary may not) is more interesting and more honest than what's currently written in `research_analysis.md`.
