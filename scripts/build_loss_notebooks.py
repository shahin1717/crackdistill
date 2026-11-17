#!/usr/bin/env python3
"""
Build the Kaggle notebooks of the loss stage (Step 2 of docs/report/plan.pdf): which KD term to add to the
label-distilled student (R2), and with which weight.

Every run starts from the R2 seed-0 student and trains 40 more epochs on the R2 labels (GT masks on 25% of the
photos, SAM masks on the rest), with flip / 90-degree rotation augmentation that the cached teacher can follow.
The runs differ only in the extra loss term:

  L0  none (control: the same fine-tuning with the task loss only)
  L1  mask KD      KL to the teacher's soft logits, inside each crack's box, where the teacher agrees with the label
  L2  boundary     BCE to the teacher's soft mask, weighted towards uncertain (edge) pixels
  L3  feature KD   channel-wise distillation of SAM encoder features into the neck (P3, P4, P5)

The weight of a term is set by scale (KDScaleCalibrator): the term is rescaled to 25% of the task loss, measured
on the first 100 batches and then frozen.

Usage:  python scripts/build_loss_notebooks.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.build_final_notebooks import CFG_MASK_KD, D4, GATED, SUBSET_EXPECTED, generate_training_notebook, make_cell
from scripts.build_label_distill_notebooks import SAM_SETUP

OUT = ROOT / "loss_notebooks"
PSEUDO = "data/datasets/crack500_pseudo"
KD_DIR = "/tmp/kd_teacher"                      # not in /kaggle/working: ~8 GB of features must not become output
LOGITS_DIR = f"{KD_DIR}/teacher_logits_ft"
R2_RUN = "sam_labels_p25_seed0"
FRACTION, EPOCHS, FRAC = 0.25, 40, 0.25
# Ultralytics arguments of the fine-tuning: a fixed optimizer and a learning rate 10x below the one R2 trained with
FT_ARGS = {"optimizer": "AdamW", "lr0": 2e-4, "warmup_epochs": 1.0}

OFF = {f"distillation.losses.{t}.enabled": False for t in ("mask_kd", "feature", "boundary", "affinity", "tversky")}
BASE = {**CFG_MASK_KD, **OFF, **D4, "train.epochs": EPOCHS}


def term(name, **extra):
    k = f"distillation.losses.{name}"
    return {f"{k}.enabled": True, f"{k}.weight": 1.0, f"{k}.auto_frac": FRAC, **{f"{k}.{a}": v for a, v in extra.items()}}


# (file, arm, title, description, config, needs teacher features)
RUNS = [
    ("L0_control", "ft_control", "R2 + 40 epochs, task loss only (control)",
     "The fine-tuning recipe of the loss stage without any KD term. Every other L run is compared with this one.",
     {**BASE, "distillation.enabled": False}, False),
    ("L1_mask_kd", "ft_mask_kd", "R2 + mask KD",
     "Task loss + KL to the fine-tuned teacher's soft logits, inside each crack's box and only where the teacher "
     "agrees with the training label.",
     {**BASE, **term("mask_kd"), **GATED}, False),
    ("L2_boundary", "ft_boundary", "R2 + boundary loss",
     "Task loss + BCE to the fine-tuned teacher's soft mask, weighted towards the pixels where the teacher is "
     "uncertain (crack edges).",
     {**BASE, **term("boundary")}, False),
    ("L3_feature_kd", "ft_feature_kd", "R2 + feature KD",
     "Task loss + channel-wise distillation of SAM 2.1 encoder features into the neck outputs P3, P4 and P5.",
     {**BASE, **term("feature", method="cwd", temperature=4.0, layers=[16, 19, 22])}, True),
    ("L4_mask_feature_kd", "ft_mask_feature_kd", "R2 + mask KD + feature KD",
     "Task loss + the two terms that came closest to the threshold alone (L1 +0.006, L3 +0.009), each at 25% of the task loss.",
     {**BASE, **term("mask_kd"), **GATED, **term("feature", method="cwd", temperature=4.0, layers=[16, 19, 22])}, True),
]
# Runs repeated with training seeds 1 and 2 (same R2 seed-0 start, same labels): is L3's +0.009 over L0 more than noise?
SEEDED = {"L0_control", "L3_feature_kd"}


def prepare_cell(kd, features):
    """Step 2b: the R2 student and teacher from the attached R2 output, the R2 labels, and the teacher cache."""
    exp = SUBSET_EXPECTED[FRACTION]
    pct = round(FRACTION * 100)
    labeled = f"data/datasets/crack500_yolo/train_photos{pct}.txt"
    kd_args = f"--kd-out {KD_DIR} " + ("--features " if features else "") if kd else ""
    src = f"""# ── Step 2b: R2 student, R2 labels and the teacher cache ──
import json, sys
sys.path.insert(0, ".")
assert Path("data/datasets/crack500/traincrop").exists(), "[FATAL] raw Crack500 (traincrop/ with .png masks) is required: attach distill-datasetforme"

# The R2 seed-0 notebook output must be attached: it holds the student to start from and the fine-tuned teacher
cands = sorted(p for p in Path("/kaggle/input").rglob("best.pt") if "{R2_RUN}" in str(p))
assert len(cands) == 1, f"[FATAL] attach the output of the R2 seed-0 notebook (need exactly one {R2_RUN} best.pt), found: {{cands}}"
STUDENT_INIT = cands[0]
print(f"[Student] starts from {{STUDENT_INIT}}")

from scripts.make_train_subset import write_subset
subset = write_subset("data/datasets/crack500_yolo", {FRACTION}, seed=0)
print(f"[Subset] {{subset['photos']}} photos, {{subset['crops']}} crops, sha256 {{subset['sha256'][:12]}}")
assert (subset["photos"], subset["crops"], subset["sha256"][:12]) == {exp!r}, f"[FATAL] unexpected subset: {{subset}}"
""" + SAM_SETUP + f"""
# Teacher: the decoder R2 fine-tuned. If the R2 output does not hold it, fine-tune again (same recipe, ~25 min).
decoders = sorted(Path("/kaggle/input").rglob("sam2_crack_decoder.pt"))
if decoders:
    W = decoders[0]
else:
    !python scripts/sam2_teacher_finetune.py finetune --crack500 data/datasets/crack500 --train-list {labeled} --out runs/teacher_ft --epochs 10
    W = Path("runs/teacher_ft/sam2_crack_decoder.pt")
assert W.exists(), "[FATAL] no fine-tuned teacher decoder: see the output above"
print(f"[Teacher] fine-tuned decoder {{W}}")
!python scripts/make_pseudo_labels.py --source sam --weights {{W}} --labeled-list {labeled} --out {PSEUDO} {kd_args}

meta = json.loads(Path("{PSEUDO}/meta.json").read_text())
n_labels = len(list(Path("{PSEUDO}/labels/train").glob("*.txt")))
quality = meta["pseudo_quality_vs_hidden_gt"]
print(f"[Labels] GT crops: {{meta['gt_crops']}}, predicted: {{meta['pseudo_crops']}} (empty: {{meta['pseudo_empty']}}), label files: {{n_labels}}")
print(f"[Labels] predicted-mask quality vs hidden GT: {{quality}}")
assert meta["gt_crops"] == {exp[1]} and n_labels == 1896, f"[FATAL] unexpected label set: {{meta}}"
# R2 trained on masks with F1 0.80: much lower means this is not the R2 teacher
assert quality["f1"] > 0.75, f"[FATAL] teacher masks are worse than in R2 (F1 0.80): {{quality}}"
# The trainer must read the labels written above, not the GT labels of the source dataset
from ultralytics.data.utils import check_det_dataset
train_dir = check_det_dataset("{PSEUDO}/dataset.yaml")["train"]
print(f"[Labels] trainer reads images from {{train_dir}}")
assert str(Path("{PSEUDO}").resolve()) in str(train_dir), f"[FATAL] trainer resolved to another dataset: {{train_dir}}"
"""
    if kd:
        src += f"""
n_logits = len(list(Path("{LOGITS_DIR}").glob("*_logits.npy")))
print(f"[Teacher] cached logit files: {{n_logits}} in {LOGITS_DIR}")
assert n_logits == meta["kd_logit_files"] and n_logits > 1600, f"[FATAL] teacher logits are missing: {{n_logits}}"
"""
    if features:
        src += f"""
n_feats = len(list(Path("{KD_DIR}/teacher_features").glob("*_features.npz")))
print(f"[Teacher] cached feature files: {{n_feats}}")
assert n_feats == n_logits, f"[FATAL] teacher features are missing: {{n_feats}} of {{n_logits}}"
"""
    src += """
Path("/kaggle/working/results").mkdir(parents=True, exist_ok=True)
Path("/kaggle/working/results/label_meta.json").write_text(json.dumps(meta, indent=2))
"""
    return src


def cut(text, marker, note):
    """Drop everything from marker to the end of the cell."""
    assert marker in text, marker
    return text[:text.index(marker)] + note + "\n"


def notebook(arm, title, desc, cfg, features, seed=0):
    kd = cfg["distillation.enabled"]
    desc += (f" Starts from the R2 seed-0 student and trains {EPOCHS} epochs on the R2 labels with flip / rotation "
             "augmentation. Val and test labels are GT.")
    nb = generate_training_notebook(arm, title, desc, cfg, seed=seed, prompt_type="box" if kd else "none",
                                    logits_dir=LOGITS_DIR, arm=arm)
    old_data = '"path": "data/datasets/crack500_yolo", "format"'
    old_train = "trainer = KDSegmentationTrainer(cfg)"
    old_cfg = "cfg = override_config(cfg, overrides)"
    cells, seen = [], set()
    for c in nb["cells"]:
        text = "".join(c["source"]).replace("_150ep", f"_ft{EPOCHS}ep").replace("Epochs=150", f"Epochs={EPOCHS}")
        if text.startswith("# ── Step 1") and kd:
            # never link logits from an attached dataset: those are the zero-shot teacher's, in another instance order
            text = cut(text, "# 2. Link precomputed teacher logits", "# 2. Teacher logits are generated in Step 2b")
            seen.add(1)
        if text.startswith("# ── Step 2") and kd:
            text = cut(text, "# 3. Verify Logits", "# 3. Teacher logits are generated and checked in Step 2b")
            seen.add(2)
        if text.startswith("# ── Step 3"):
            assert old_data in text and old_train in text and old_cfg in text
            text = text.replace(old_data, f'"path": "{PSEUDO}/dataset.yaml", "format"')
            text = text.replace(old_cfg, f'overrides["student.backbone"] = str(STUDENT_INIT)\n{old_cfg}')
            text = text.replace(old_train, f"trainer = KDSegmentationTrainer(cfg, overrides={FT_ARGS!r})")
            seen.add(3)
        cells.append(make_cell(c["cell_type"], text))
        if text.startswith("# ── Step 2"):
            for f in ("sam2_teacher_finetune", "make_pseudo_labels", "make_train_subset"):
                cells.append(make_cell("code", f"%%writefile scripts/{f}.py\n{(ROOT / 'scripts' / f'{f}.py').read_text()}"))
            cells.append(make_cell("code", prepare_cell(kd, features)))
    assert seen == ({1, 2, 3} if kd else {3}), seen
    nb["cells"] = cells
    return nb


README = f"""# Loss-stage notebooks

Step 2 of `docs/report/plan.pdf`: which KD term to add to the label-distilled student, and its weight.
Built by `scripts/build_loss_notebooks.py`; do not edit by hand.

Every run starts from the **R2 seed-0 student** and trains {EPOCHS} more epochs on the R2 labels, with flip /
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

**Weights:** each term is rescaled to {round(FRAC * 100)}% of the task loss, measured on the first 100 batches and then
frozen. The value is printed as `[KD] weight of '<term>' fixed at ...`.

**Check in the log**
- `[Student] starts from ...{R2_RUN}_150ep/weights/best.pt`
- `[Subset] 62 photos, 478 crops, sha256 a6ec6eb8eeee`
- `[Labels] predicted-mask quality vs hidden GT: ... 'f1': 0.80...` (the R2 teacher)
- L1 to L3: `[Teacher] cached logit files: ...`, `[KD] ✓ KD losses computed: ...`, `[KD] weight of ... fixed at ...`
- `FINAL EVALUATION SUMMARY` (val mask mAP50 on GT)

**Read the result against L0, not against R2 (0.538):** a term helps if it beats L0 by at least 0.01 val mask mAP50.
Next, for the best term: half and double the weight; then the combination of the terms that helped; then 3 seeds.

**Download from Output:** `results/*.json` and `runs/segment/crack_distill/<run>/weights/best.pt`.
"""


def main():
    OUT.mkdir(exist_ok=True)
    for name, arm, title, desc, cfg, features in RUNS:
        for seed in (0, 1, 2) if name in SEEDED else (0,):
            path = OUT / f"{name}_seed{seed}.ipynb"
            path.write_text(json.dumps(notebook(arm, title, desc, cfg, features, seed), indent=1, ensure_ascii=False) + "\n")
            print(f"wrote {path.relative_to(ROOT)}")
    (OUT / "README.md").write_text(README)


if __name__ == "__main__":
    main()
