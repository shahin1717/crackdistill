#!/usr/bin/env python3
"""
Build the teacher notebooks in teacher_notebooks/ (separate from the student training notebooks).

  T1_sam2_teacher_finetune.ipynb  SAM 2.1 zero-shot -> fine-tune decoder on cracks -> gate (val F1 > 0.80) -> re-cache logits
  T2_sam3_zeroshot.ipynb          SAM 3 zero-shot with the text prompt "crack", same metrics (bake-off; optional)

Usage: python scripts/build_teacher_notebooks.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.build_final_notebooks import make_cell, make_nb

OUT = ROOT / "teacher_notebooks"
GATE_F1 = 0.80
SPLITS = {"traincrop": 1896, "valcrop": 348, "testcrop": 1124}

teacher_code = (ROOT / "scripts/sam2_teacher_finetune.py").read_text()

LINK = f"""# ── Link raw Crack500 (the folder holding traincrop/ valcrop/ testcrop/ with .jpg + .png masks) ──
import os, json, sys
from pathlib import Path
sys.path.insert(0, ".")
raw = sorted(Path(r) for r, d, f in os.walk("/kaggle/input", followlinks=True) if "traincrop" in d)
assert raw, "[FATAL] raw Crack500 (a folder containing traincrop/) not found in /kaggle/input: attach distill-datasetforme"
CRACK500 = raw[0]
counts = {{s: len(list((CRACK500 / s).glob("*.jpg"))) for s in {list(SPLITS)!r}}}
print(f"[Dataset] {{CRACK500}} {{counts}}")
assert counts == {SPLITS!r}, f"[FATAL] not the canonical Crack500 crops: {{counts}}"
"""


def t1():
    return make_nb([
        make_cell("markdown", f"""# T1: SAM 2.1 teacher, zero-shot vs crack fine-tuned
Is the teacher better than the student? KD can only help if it is.

1. **Zero-shot** SAM 2.1 Hiera-L on Crack500 val crops, box-prompted per GT component (exactly how the KD logits were made).
2. **Fine-tune** the mask decoder + prompt encoder on the 1896 train crops (image encoder frozen, 6 epochs, flips, box jitter),
   keep the epoch with the best val pixel F1.
3. **Gate:** fine-tuned val pixel F1 must be **> {GATE_F1}** (student is ~0.73). Test crops are scored once, for the report.
4. **If the gate passes:** re-cache the train logits with the fine-tuned teacher (`teacher_logits_ft.zip`, same format as before).

**Attach:** `distill-datasetforme` (raw Crack500). **GPU:** T4. **Time:** ~1.5–2 h.
**Download from Output:** `teacher_ft/` (history.json, eval_*.json, sam2_crack_decoder.pt) and `teacher_logits_ft.zip` if written."""),
        make_cell("code", """!mkdir -p scripts checkpoints runs
!pip install -q SAM-2 || pip install -q git+https://github.com/facebookresearch/sam2.git
!wget -q -nc https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2_hiera_large.pt
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
        make_cell("code", f"%%writefile scripts/sam2_teacher_finetune.py\n{teacher_code}"),
        make_cell("code", LINK),
        make_cell("code", """# ── Step 1: zero-shot teacher on val ──
!python scripts/sam2_teacher_finetune.py eval --crack500 {CRACK500} --split valcrop --out runs/teacher_ft"""),
        make_cell("code", """# ── Step 2: fine-tune decoder + prompt encoder (best val F1 kept) ──
!python scripts/sam2_teacher_finetune.py finetune --crack500 {CRACK500} --out runs/teacher_ft --epochs 6 --lr 1e-4"""),
        make_cell("code", """# ── Step 3: score both teachers on test once (report only; selection used val) ──
W = "runs/teacher_ft/sam2_crack_decoder.pt"
!python scripts/sam2_teacher_finetune.py eval --crack500 {CRACK500} --split valcrop --weights {W} --out runs/teacher_ft
!python scripts/sam2_teacher_finetune.py eval --crack500 {CRACK500} --split testcrop --out runs/teacher_ft
!python scripts/sam2_teacher_finetune.py eval --crack500 {CRACK500} --split testcrop --weights {W} --out runs/teacher_ft"""),
        make_cell("code", f"""# ── Gate + summary ──
import pandas as pd
rows = {{p.stem.replace("eval_", ""): json.loads(p.read_text()) for p in sorted(Path("runs/teacher_ft").glob("eval_*.json"))}}
df = pd.DataFrame(rows).T[["images", "precision", "recall", "f1", "mean_iou", "thin_recall"]]
print(df.to_string(float_format=lambda x: f"{{x:.3f}}"))
print(pd.DataFrame(json.loads(Path("runs/teacher_ft/history.json").read_text())).to_string(float_format=lambda x: f"{{x:.3f}}"))
ft_val_f1 = rows["sam2_crack_decoder_valcrop"]["f1"]
GATE_PASSED = ft_val_f1 > {GATE_F1}
print(f"\\n[Gate] fine-tuned teacher val pixel F1 = {{ft_val_f1:.3f}} -> {{'PASS: re-cache logits and run KD' if GATE_PASSED else 'FAIL: teacher not clearly better than the student (~0.73); KD on this teacher is not worth running'}}")"""),
        make_cell("code", """# ── Step 4 (only if the gate passed): re-cache train logits with the fine-tuned teacher ──
if GATE_PASSED:
    !python scripts/sam2_teacher_finetune.py logits --crack500 {CRACK500} --weights {W} --out teacher_logits_ft
    n = len(list(Path("teacher_logits_ft").glob("crack500_*_logits.npy")))
    print(f"[Logits] {n} files (old cache: 1896 Crack500 train crops)")
    !cd teacher_logits_ft/.. && zip -qr teacher_logits_ft.zip teacher_logits_ft && ls -lh teacher_logits_ft.zip
else:
    print("[Logits] skipped: gate failed")"""),
    ])


def t2():
    return make_nb([
        make_cell("markdown", """# T2: SAM 3 zero-shot (teacher bake-off, optional)
SAM 3 segments by concept: the text prompt `"crack"` returns all crack instances, **no GT boxes needed**
(so it is a harder setting than T1's box prompts, but it is what pseudo-labelling unlabelled images would use).
Same metrics and val crops as T1, at several score thresholds.

**Before running:** request access to `facebook/sam3` on Hugging Face, then add a Kaggle secret `HF_TOKEN`
(Add-ons → Secrets). **Attach:** `distill-datasetforme`. **GPU:** T4. **Time:** ~20 min.
SAM 3 is new; if the install or API differs, the error will be in the first two cells. Send it back and I will fix it."""),
        make_cell("code", """!mkdir -p scripts runs
!pip install -q git+https://github.com/facebookresearch/sam3.git
!pip install -q einops decord pycocotools"""),
        make_cell("code", f"%%writefile scripts/sam2_teacher_finetune.py\n{teacher_code}"),
        make_cell("code", LINK),
        make_cell("code", """# ── Load SAM 3 ──
import torch, numpy as np, torch.nn.functional as F
from PIL import Image
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
login(UserSecretsClient().get_secret("HF_TOKEN"))
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.05)"""),
        make_cell("code", """# ── Zero-shot val: text prompt, union of instances above each score threshold ──
from scripts.sam2_teacher_finetune import pairs, read_pair, new_acc, accumulate, summarize
PROMPTS, THRESH = ("crack", "pavement crack"), (0.1, 0.2, 0.3, 0.5)
acc = {(p, t): new_acc() for p in PROMPTS for t in THRESH}
for i, (img_path, mask_path) in enumerate(pairs(CRACK500 / "valcrop"), 1):
    img, gt = read_pair(img_path, mask_path)
    state = processor.set_image(Image.fromarray(img))
    for p in PROMPTS:
        out = processor.set_text_prompt(state=state, prompt=p)
        scores = torch.as_tensor(out["scores"]).float().flatten().cpu()
        m = torch.as_tensor(out["masks"]).float().cpu()
        m = m.reshape(len(scores), 1, *m.shape[-2:]) if len(scores) else torch.zeros(0, 1, *gt.shape)
        if m.shape[-2:] != gt.shape:
            m = F.interpolate(m, size=gt.shape, mode="bilinear")
        m = m[:, 0] > 0.5
        for t in THRESH:
            keep = m[scores >= t]
            accumulate(acc[(p, t)], keep.any(0).numpy() if len(keep) else np.zeros(gt.shape, bool), gt)
    if i % 50 == 0:
        print(f"  {i} images", flush=True)

import pandas as pd
res = {f"{p} @ {t}": summarize(a) for (p, t), a in acc.items()}
df = pd.DataFrame(res).T[["images", "precision", "recall", "f1", "mean_iou", "thin_recall"]]
print(df.to_string(float_format=lambda x: f"{x:.3f}"))
Path("runs/teacher_sam3").mkdir(parents=True, exist_ok=True)
Path("runs/teacher_sam3/eval_sam3_valcrop.json").write_text(json.dumps(res, indent=2) + "\\n")
print("Compare against T1's zero-shot SAM 2 row (box prompts) and the gate (val F1 > 0.80).")"""),
    ])


def main():
    OUT.mkdir(exist_ok=True)
    for name, nb in [("T1_sam2_teacher_finetune.ipynb", t1()), ("T2_sam3_zeroshot.ipynb", t2())]:
        (OUT / name).write_text(json.dumps(nb, indent=1) + "\n")
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
