# Loss-stage notebooks

Step 2 of `docs/report/plan.pdf`: which KD term to add to the label-distilled student, and its weight.
Built by `scripts/build_loss_notebooks.py`; do not edit by hand.

Every run starts from the **R2 seed-0 student** and trains 40 more epochs on the R2 labels, with flip /
rotation augmentation (the cached teacher cannot follow mosaic). Only the extra loss term changes.

| Notebook | Extra loss term | ~T4 time |
| :--- | :--- | :---: |
| `L0_control_seed0` | none (control) | ~45 min |
| `L1_mask_kd_seed0` | mask KD: KL to the teacher's soft logits | ~1 h |
| `L2_boundary_seed0` | boundary: BCE to the teacher's soft mask, weighted to edge pixels | ~1 h |
| `L3_feature_kd_seed0` | feature KD: SAM encoder features into the neck (P3 to P5) | ~1.5 h |
| `L4_mask_feature_kd_seed0` | mask KD + feature KD together | ~1.5 h |

**Seed 0 results (val mask mAP50):** L0 0.549, L1 0.555, L2 0.549, L3 0.558. No term reached the +0.01 threshold.
`L0_control` and `L3_feature_kd` also exist as `_seed1` and `_seed2` (same R2 seed-0 start, other training seed) to
check whether L3's +0.009 is more than seed noise.

**Attach to every notebook:** `distill-datasetforme` and the **output of the R2 seed-0 notebook**
(`r2-sam-labels-p25-seed0`). It provides the student (`best.pt`) and the fine-tuned teacher
(`runs/teacher_ft/sam2_crack_decoder.pt`). Attach no other notebook output.

**Order:** L0 and L1 first. L3 is the heaviest (it caches ~8 GB of teacher features in `/tmp`) and the least tested.

**Weights:** each term is rescaled to 25% of the task loss, measured on the first 100 batches and then
frozen. The value is printed as `[KD] weight of '<term>' fixed at ...`.

**Check in the log**
- `[Student] starts from ...sam_labels_p25_seed0_150ep/weights/best.pt`
- `[Subset] 62 photos, 478 crops, sha256 a6ec6eb8eeee`
- `[Labels] predicted-mask quality vs hidden GT: ... 'f1': 0.80...` (the R2 teacher)
- L1 to L3: `[Teacher] cached logit files: ...`, `[KD] ✓ KD losses computed: ...`, `[KD] weight of ... fixed at ...`
- `FINAL EVALUATION SUMMARY` (val mask mAP50 on GT)

**Read the result against L0, not against R2 (0.538):** a term helps if it beats L0 by at least 0.01 val mask mAP50.
Next, for the best term: half and double the weight; then the combination of the terms that helped; then 3 seeds.

**Download from Output:** `results/*.json` and `runs/segment/crack_distill/<run>/weights/best.pt`.
