#!/usr/bin/env python3
"""
Build the budgeted label-distillation notebooks in label_distill_notebooks/ (docs/report/plan.pdf).

Masks are kept for p% of the train photos; the other crops keep only boxes, and SAM 2.1 (or, as a control, the
student itself) turns those boxes into masks. Every run trains YOLOv11n-seg with the A4 recipe (Ultralytics
default augmentation) and is validated on the GT val set.

  R1  25% masks only                          lower bound
  R2  25% masks + SAM masks (SAM tuned on the 25%)   main result
  R3  0% masks, zero-shot SAM masks on all crops
  R4  25% masks + masks predicted by the R1 student  control: is SAM needed?
  R5a 10% masks only        R5b 10% masks + SAM masks

Usage: python scripts/build_label_distill_notebooks.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.build_final_notebooks import (CFG_BASELINE, SUBSET_EXPECTED, ULTRA_AUG, generate_training_notebook,
                                           make_cell)

OUT = ROOT / "label_distill_notebooks"
PSEUDO = "data/datasets/crack500_pseudo"
CFG = {**CFG_BASELINE, **ULTRA_AUG}
N_TRAIN = 1896
R1_ARM = "baseline_ultra_aug_p25"

# (file, arm, title, mask fraction, label source for the rest: None = drop them, "sam", "student")
RUNS = [
    ("R1_baseline_p25", R1_ARM, "25% masks only (lower bound)", 0.25, None),
    ("R2_sam_labels_p25", "sam_labels_p25", "25% masks + SAM masks on the rest (main result)", 0.25, "sam"),
    ("R3_sam_labels_p0", "sam_labels_p0", "0% masks: zero-shot SAM masks on all crops", 0.0, "sam"),
    ("R4_self_labels_p25", "self_labels_p25", "25% masks + the R1 student's own masks on the rest (control)", 0.25, "student"),
    ("R5a_baseline_p10", "baseline_ultra_aug_p10", "10% masks only", 0.1, None),
    ("R5b_sam_labels_p10", "sam_labels_p10", "10% masks + SAM masks on the rest", 0.1, "sam"),
]


def label_cells(fraction, source):
    """Cells that build crack500_pseudo: GT masks for the labelled crops, box-prompted masks for the rest."""
    pct = round(fraction * 100)
    labeled = f"data/datasets/crack500_yolo/train_photos{pct}.txt"
    src = """# ── Step 2b: budgeted label distillation: build the training labels ──
import json, sys
sys.path.insert(0, ".")
assert Path("data/datasets/crack500/traincrop").exists(), "[FATAL] raw Crack500 (traincrop/ with .png masks) is required: attach distill-datasetforme"
"""
    if fraction:
        exp = SUBSET_EXPECTED[fraction]
        src += f"""
# Crops that keep their GT masks: {pct}% of the train photos (same nested subset as the E runs)
from scripts.make_train_subset import write_subset
subset = write_subset("data/datasets/crack500_yolo", {fraction}, seed=0)
print(f"[Subset] {{subset['photos']}} photos, {{subset['crops']}} crops, sha256 {{subset['sha256'][:12]}}")
assert (subset["photos"], subset["crops"], subset["sha256"][:12]) == {exp!r}, f"[FATAL] unexpected subset: {{subset}}"
"""
    labeled_arg = f"--labeled-list {labeled} " if fraction else ""
    if source == "sam":
        src += """
!pip install -q SAM-2 || pip install -q git+https://github.com/facebookresearch/sam2.git
# SAM 2.1 checkpoint (~900 MB). A failed download leaves an empty file, so check the size and fall back to the HF hub.
import shutil
ckpt = Path("checkpoints/sam2_hiera_large.pt")
ckpt.parent.mkdir(exist_ok=True)
if not ckpt.exists() or ckpt.stat().st_size < 8e8:
    ckpt.unlink(missing_ok=True)
    !wget -nv https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O {ckpt}
if not ckpt.exists() or ckpt.stat().st_size < 8e8:
    from huggingface_hub import hf_hub_download
    ckpt.unlink(missing_ok=True)
    shutil.copy(hf_hub_download("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt"), ckpt)
assert ckpt.stat().st_size > 8e8, f"[FATAL] SAM 2.1 checkpoint download failed ({ckpt.stat().st_size} bytes)"
print(f"[Teacher] checkpoint {ckpt} {ckpt.stat().st_size / 1e6:.0f} MB")
"""
        if fraction:
            src += f"""
# Teacher: SAM 2.1 decoder fine-tuned on the labelled crops only (best val F1 kept; zero-shot if tuning never helps)
!python scripts/sam2_teacher_finetune.py finetune --crack500 data/datasets/crack500 --train-list {labeled} --out runs/teacher_ft --epochs 10
assert Path("runs/teacher_ft/history.json").exists(), "[FATAL] teacher fine-tuning crashed: see the traceback above"
W = Path("runs/teacher_ft/sam2_crack_decoder.pt")
WEIGHTS = f"--weights {{W}}" if W.exists() else ""
print(f"[Teacher] {{'fine-tuned decoder ' + str(W) if W.exists() else 'fine-tuning never beat zero-shot on val: using zero-shot SAM'}}")
!python scripts/make_pseudo_labels.py --source sam {{WEIGHTS}} {labeled_arg}--out {PSEUDO}
"""
        else:
            src += f"""
# Teacher: zero-shot SAM 2.1 (no masks are used anywhere in training)
!python scripts/make_pseudo_labels.py --source sam --out {PSEUDO}
"""
    else:
        src += f"""
# Label source: the R1 student ({R1_ARM}); attach the R1 notebook output or a dataset holding its best.pt
pts = [p for p in Path("/kaggle/input").rglob("*.pt") if "sam2" not in p.name and not p.name.startswith("yolo")]
cands = sorted(p for p in pts if "{R1_ARM}" in str(p) and p.name == "best.pt") or sorted(pts)
assert len(cands) == 1, f"[FATAL] need exactly one R1 checkpoint in /kaggle/input, found: {{cands}}"
STUDENT = cands[0]
print(f"[Student] {{STUDENT}}")
!python scripts/make_pseudo_labels.py --source student --student {{STUDENT}} {labeled_arg}--out {PSEUDO}
"""
    n_gt = SUBSET_EXPECTED[fraction][1] if fraction else 0
    src += f"""
meta = json.loads(Path("{PSEUDO}/meta.json").read_text())
n_labels = len(list(Path("{PSEUDO}/labels/train").glob("*.txt")))
print(f"[Labels] GT crops: {{meta['gt_crops']}}, predicted: {{meta['pseudo_crops']}} (empty: {{meta['pseudo_empty']}}), label files: {{n_labels}}")
print(f"[Labels] predicted-mask quality vs hidden GT: {{meta['pseudo_quality_vs_hidden_gt']}}")
assert meta["gt_crops"] == {n_gt} and n_labels == {N_TRAIN}, f"[FATAL] unexpected label set: {{meta}}"
# The trainer must read the labels written above, not the GT labels of the source dataset
from ultralytics.data.utils import check_det_dataset
train_dir = check_det_dataset("{PSEUDO}/dataset.yaml")["train"]
print(f"[Labels] trainer reads images from {{train_dir}}")
assert str(Path("{PSEUDO}").resolve()) in str(train_dir), f"[FATAL] trainer resolved to another dataset: {{train_dir}}"
Path("/kaggle/working/results").mkdir(parents=True, exist_ok=True)
Path("/kaggle/working/results/label_meta.json").write_text(json.dumps(meta, indent=2))
"""
    files = ["sam2_teacher_finetune", "make_pseudo_labels"] + (["make_train_subset"] if fraction else [])
    writes = [make_cell("code", f"%%writefile scripts/{f}.py\n{(ROOT / 'scripts' / f'{f}.py').read_text()}") for f in files]
    return writes + [make_cell("code", src)]


def notebook(arm, title, fraction, source):
    if source is None:
        return generate_training_notebook(arm, title, "A4 recipe on the labelled crops only.", CFG, seed=0,
                                          prompt_type="none", arm=arm, train_fraction=fraction)
    desc = ("A4 recipe on all 1896 train crops. GT masks are used only for the labelled crops; the other crops get "
            "masks predicted from their boxes (Step 2b). Val and test labels are GT.")
    nb = generate_training_notebook(arm, title, desc, CFG, seed=0, prompt_type="none", arm=arm)
    cells, old = [], '"path": "data/datasets/crack500_yolo", "format"'
    for c in nb["cells"]:
        text = "".join(c["source"])
        if text.startswith("# ── Step 3"):
            assert old in text
            c = make_cell("code", text.replace(old, f'"path": "{PSEUDO}/dataset.yaml", "format"'))
        cells.append(c)
        if text.startswith("# ── Step 2"):
            cells += label_cells(fraction, source)
    assert len(cells) > len(nb["cells"])
    nb["cells"] = cells
    return nb


README = f"""# Budgeted label distillation notebooks

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
- `[Dataset] crack500_yolo split sizes: {{'train': 1896, 'val': 348, 'test': 1124}}`
- `[Subset] 62 photos, 478 crops, sha256 a6ec6eb8eeee` (25%) or `25 photos, 201 crops, sha256 f36b958206f0` (10%)
- R2/R5b: `[epoch N] ... 'f1': ...` lines of the teacher, then `[Teacher] fine-tuned decoder ...`
- R2-R5b: `[Labels] GT crops: ..., predicted: ..., label files: 1896` and the predicted-mask quality line
- `FINAL EVALUATION SUMMARY` (val mask mAP50 on GT)

**Download from Output:** `results/*.json` (metrics + `label_meta.json`), `runs/crack_distill/<run>/weights/best.pt`,
and for R2/R5b `runs/teacher_ft/history.json`.

**Decision rule (fixed in advance):** positive if R2 beats R1 by >= 0.02 val mask mAP50 with the same sign on test
pixel F1, and R2 beats R4. Headline number: the gap between R2 and A4.
"""


def main():
    OUT.mkdir(exist_ok=True)
    for name, arm, title, fraction, source in RUNS:
        path = OUT / f"{name}_seed0.ipynb"
        path.write_text(json.dumps(notebook(arm, title, fraction, source), indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")
    (OUT / "README.md").write_text(README)


if __name__ == "__main__":
    main()
