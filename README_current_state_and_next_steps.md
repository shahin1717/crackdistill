# Crack-Distill — Current State & What To Do Next

*Snapshot after box-only vs box+centroid comparison. This is the up-to-date source of truth — earlier docs in this thread reference decisions that have since changed (weights, freeze, prompt type).*

---

## ✅ Locked decisions (don't revisit these without new evidence)

| Decision | Choice | Why |
|---|---|---|
| Training dataset | **Crack500 only** | DeepCrack-specific findings (freeze) don't apply here; keep them separate |
| Loss recipe | **Task + Mask-KL only** (`mask_kd`) | feature_mse and boundary_bce disabled entirely — not just weight 0 |
| Temperature / weight | **T = 3.7769, W = 0.9612** | Empirically beats the "properly tuned" nb8a search values (T=1.93/W=0.458) at full 150-epoch length — short-horizon search ranking didn't hold |
| Progressive SegHead freeze | **Disabled** | Crack500 (~1900 train images) is above the <1000-image threshold where freezing helped on DeepCrack — untested directly on Crack500, but a reasonable default |
| Prompt type | **Box-only** | Box+centroid underperforms *even the no-KD baseline* — closed out, don't revisit |

## 📊 Verified results so far (in-domain, cropped Crack500 val: 348 img / 630 instances)

| Config | Mask mAP50 | Mask mAP50-95 |
|---|---|---|
| No-KD baseline (YOLOv11n-seg) | 0.540 | 0.207 |
| **Box-only mask-KD (current recipe)** | **0.550** | **0.207** |
| Box+centroid mask-KD (ruled out) | 0.525 | 0.199 |

**Honest read:** box-only KD's in-domain edge over baseline (+0.010 mAP50, flat on mAP50-95) is small — comparable in size to the noise you've already seen from weight variation alone (~0.008). It's a real, verified number, but not yet a strong result on its own.

---

## 🔴 What to do next, in order

### 1. OOD/uncropped eval — the single most important remaining step
Every run so far only reports in-domain cropped performance. Your project's actual hypothesis — SAM2 KD improves generalization, not in-domain accuracy — has never been tested with real (non-broken) data. This is pure inference on checkpoints you already have (baseline + box-only KD), not new training. Needed:
1. An uncropped Crack500 `dataset.yaml` (conversion step, if not already built).
2. Run `.val()` on both the baseline and box-only KD checkpoints against it.
3. Compare the *gap* between them OOD vs in-domain — that gap is the actual finding, not either number alone.

**This determines whether the whole KD approach is worth keeping.** If box-only KD doesn't show a real OOD advantage either, the honest conclusion is that this recipe isn't earning its complexity over the plain baseline.

### 2. Second seed on box-only KD (and ideally the baseline)
The current +0.010 mAP50 gap is small enough that it needs a second seed before you'd want to defend it as a real effect rather than run-to-run variance. One more ~3 hour run, seed 123, same config otherwise.

### 3. (Lower priority) Freeze-vs-no-freeze on Crack500
Still a carried-over assumption from DeepCrack, never directly tested on Crack500. Worth doing eventually for completeness, not blocking.

### 4. (Optional, deprioritized) Everything else from earlier lists
- nb6 batch-size (32 vs 16) — never run, not urgent.
- YOLO26 migration — separate track, only relevant once the Crack500 recipe itself is settled.
- Numpy logit-diff sanity check between `teacher_logits_box/` and `teacher_logits_centroid/` — nice to have for the writeup, not blocking now that the box vs centroid *training* result already came back clearly different (the old "identical" result is resolved either way).

---

## 🛑 Standing checklist — run this before trusting any new notebook's output

This project has hit the same class of bug three separate times (missing logits, stale override values, silent checkpoint fallback). Before trusting any new result:

- [ ] `[KD] logit files: N` — must be > 0, and a real count (hundreds/thousands), not a stale cached number.
- [ ] `[KD] ✓ KD losses computed` (or equivalent) actually appears in the log, with `mask_kd` nonzero and `feature`/`boundary` absent for this recipe.
- [ ] The printed `EXPERIMENT_NAME` / `Config actually in effect` line shows the temperature/weight you *think* you set — not a stale value from a variable that didn't get updated.
- [ ] No checkpoint-resolution fallback silently grabbing "any `best.pt`" if the exact experiment's isn't found — should fail loudly instead.
- [ ] Eval set (image/instance count in the val summary line) matches what you're comparing it against.

---

## One-line status
Box-only mask-KD beats box+centroid decisively and edges out the no-KD baseline in-domain, but that in-domain edge is small and unseeded. The project's real test — does it help out-of-distribution — is still unanswered and is the next thing to run.
