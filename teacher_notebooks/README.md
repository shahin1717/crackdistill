# Teacher notebooks

Is the teacher better than the student? KD is worth running only if it is. Built by `scripts/build_teacher_notebooks.py`.

| Notebook | Does | Attach | Time (T4) | Download |
| :--- | :--- | :--- | :---: | :--- |
| `T1_sam2_teacher_finetune.ipynb` | SAM 2.1 zero-shot → fine-tune decoder on cracks → gate → re-cache logits | `distill-datasetforme` | ~1.5–2 h | `runs/teacher_ft/`, `teacher_logits_ft.zip` (if gate passed) |
| `T2_sam3_zeroshot.ipynb` | SAM 3 zero-shot, text prompt "crack" (bake-off, optional) | `distill-datasetforme` + secret `HF_TOKEN` | ~20 min | printed table, `runs/teacher_sam3/` |

**Gate:** fine-tuned teacher val pixel F1 > 0.80 (the student is ~0.73). If it fails, KD on this teacher is not worth running.

**Metrics** (val crops, prompts = one box per GT component ≥ 50 px, exactly as the KD logits were made):
pixel precision / recall / F1, mean per-image IoU, thin-crack recall (local width ≤ 4 px).

**Check in the log:** `[Dataset] ... {'traincrop': 1896, 'valcrop': 348, 'testcrop': 1124}` and the `[Gate]` line.
