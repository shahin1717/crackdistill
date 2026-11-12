# Budgeted label distillation notebooks

Plan: `docs/report/plan.pdf`. Built by `scripts/build_label_distill_notebooks.py`; do not edit by hand.

**Question:** with masks for only p% of the train photos and boxes for the rest, how close does SAM bring
YOLOv11n-seg to the fully labelled model (A4: val mask mAP50 0.540)?

| Notebook | Training labels | Attach | ~T4 time |
| :--- | :--- | :--- | :---: |
| `R1_baseline_p25_seed0` | GT masks on 478 crops only | `distill-datasetforme` | ~1.5 h |
| `R2_sam_labels_p25_seed0` | GT on 478 + SAM masks on 1418 (SAM tuned on the 478) | `distill-datasetforme` | ~4 h |
| `R3_sam_labels_p0_seed0` | zero-shot SAM masks on all 1896 | `distill-datasetforme` | ~3.5 h |
| `R4_self_labels_p25_seed0` | GT on 478 + R1 student's masks on 1418 | `distill-datasetforme` + R1's `best.pt` | ~3.5 h |
| `R5a_baseline_p10_seed0` | GT masks on 201 crops only | `distill-datasetforme` | ~1 h |
| `R5b_sam_labels_p10_seed0` | GT on 201 + SAM masks on 1695 | `distill-datasetforme` | ~4 h |

**Order:** R1, R2 and R3 first (independent, run in parallel). R4 after R1 (it needs R1's `best.pt`: attach the R1
notebook output, or upload the file as a dataset). R5a/R5b last.

**Check in the log**
- `[Dataset] crack500_yolo split sizes: {'train': 1896, 'val': 348, 'test': 1124}`
- `[Subset] 62 photos, 478 crops, sha256 a6ec6eb8eeee` (25%) or `25 photos, 201 crops, sha256 f36b958206f0` (10%)
- R2/R5b: `[epoch N] ... 'f1': ...` lines of the teacher, then `[Teacher] fine-tuned decoder ...`
- R2-R5b: `[Labels] GT crops: ..., predicted: ..., label files: 1896` and the predicted-mask quality line
- `FINAL EVALUATION SUMMARY` (val mask mAP50 on GT)

**Download from Output:** `results/*.json` (metrics + `label_meta.json`), `runs/crack_distill/<run>/weights/best.pt`,
and for R2/R5b `runs/teacher_ft/history.json`.

**Decision rule (fixed in advance):** positive if R2 beats R1 by >= 0.02 val mask mAP50 with the same sign on test
pixel F1, and R2 beats R4. Headline number: the gap between R2 and A4.
