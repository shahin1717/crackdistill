# Datasets — what to get, where from, how to import

*Checked 2026-10-14. Sources and formats were verified against the official repos/archives unless marked
UNVERIFIED. Design context: [`KD_DESIGN_v2.md`](KD_DESIGN_v2.md).*

## What you need for what

| Purpose | Data | Where it is | Action |
| :--- | :--- | :--- | :--- |
| Training on Kaggle (Stage A, C) | Crack500 raw + SAM 2 teacher logits | your Kaggle dataset `distill-datasetforme` (`datasets/crack500`, `teacher_logits/`) | **none** — attach it |
| Final test (Stage D) | Crack500 test crops + 200 full test photos | local `data/datasets/crack500` (same files) | none |
| Zero-shot OOD test (Stage D) | CFD, GAPs384, CRKWH100; DeepCrack as near-OOD | DeepCrack is local; the rest: download below | §2–§4 |
| Optional later (Stage B) | 250 full-size Crack500 **train** photos | an older Kaggle mirror (§1) | only if needed |

## 1. Crack500 and the teacher logits

- **Already done:** `distill-datasetforme` holds `datasets/crack500` (`traincrop`, `valcrop`, `testcrop`, `valdata`,
  `testdata`) and `teacher_logits/` (2,196 `*_logits.npy`). The notebooks find both automatically.
- **Original source:** the FPHBN Google Drive bundle (file id `13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD`,
  `pavement crack datasets-20210103T153625Z-001.zip`, 1.98 GB). Its `CRACK500/` layout is identical to the local
  copy. Repo: <https://github.com/fyangneil/pavement-crack-detection>.
- **Lost the teacher logits?** `python scripts/generate_teacher_logits.py --prompt-type box --logits-dir data/teacher_logits_box --dataset data/datasets/crack500_yolo`
  (needs the SAM 2.1 Large checkpoint; slow — run on Kaggle, not the laptop).
- **Full-size train photos:** not in the current official bundle, but the Kaggle mirror
  `pauldavid22/crack50020220509t090436z001` (~5.8 GB) contains `CRACK500/traindata/traindata` (250 photos +
  masks) from an older release. Whether they match the crops exactly is UNVERIFIED. They would allow
  native-scale random tiles *including crack-free pavement* (fewer false positives on full photos), but every
  tile then needs new teacher logits — a Stage B option, not part of Stage A.

## 2. Zero-shot test sets

| Dataset | Pairs | Size (W×H) | Surface | Labels | Get it | Mask rule (in the converter) |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **CFD** | 118 | 480×320 | city asphalt | 2–5 px | `git clone https://github.com/cuilimeng/CrackForest-dataset.git` (~10 MB) | `.mat` `Segmentation == 2` (3/4 are other segments, not crack); 201–329 have no GT |
| **GAPs384** | 509 crops | 540×640 / 540×440 | German asphalt | regions | FPHBN bundle `GAPS384/croppedimg.zip`, `croppedgt.zip` | PNG ≥ 128 |
| **CRKWH100** | 100 | 512×512 | asphalt (line camera) | ~1 px | <https://github.com/qinnzou/DeepCrack> README links — OneDrive (browser) or Baidu code `zfoo` | auto polarity (UNVERIFIED format); score with a 2–4 px tolerance |
| DeepCrack (near-OOD) | 237 test | 544×384 | concrete/asphalt/stone | full width | local `data/datasets/deepcrack`, or `curl -LO https://github.com/yhlleo/DeepCrack/raw/master/dataset/DeepCrack.zip` (68 MB) | PNG ≥ 128 |
| AEL (optional) | 58 | varies | asphalt | width | FPHBN bundle `AEL/img.zip`, `gt.zip` | **inverted**: PNG < 128; `Im_GT_X[or].jpg` ↔ `GT_X.png` |
| CrackTree200 (optional) | 206 | 800×600 | asphalt | 1 px | FPHBN bundle `cracktree200/…` | PNG ≥ 128; overlaps CrackTree260 — don't report both |

Not used: CrackSeg9k, Khanh11k, OmniCrack30k — they contain Crack500/CFD/DeepCrack/GAPs images in random
splits, so they leak into these test sets.

## 3. Download recipes

**CFD and DeepCrack** — the two commands in the table.

**GAPs384 / AEL / CrackTree200 without downloading 2 GB** — pull only the inner zips (~80 MB) over HTTP range
requests (listing and a small member were tested; large members not end-to-end):

```bash
pip install remotezip
python - <<'EOF'
from remotezip import RemoteZip
URL = "https://drive.usercontent.google.com/download?id=13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD&export=download&confirm=t"
P = "pavement crack datasets/"
want = ["GAPS384/croppedimg.zip", "GAPS384/croppedgt.zip", "AEL/img.zip", "AEL/gt.zip",
        "cracktree200/cracktree200rgb.zip", "cracktree200/cracktree200_gt.zip"]
with RemoteZip(URL) as z:
    for w in want:
        z.extract(P + w, "raw/fphbn")
EOF
cd "raw/fphbn/pavement crack datasets/GAPS384" && unzip -q croppedimg.zip && unzip -q croppedgt.zip
```

**Fallback (whole bundle):** `pip install -U gdown && gdown 13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD -O fphbn.zip`
(or `curl -L "<URL above>" -o fphbn.zip`). On "Too many users have viewed or downloaded this file": wait up to
24 h, or open the link in a signed-in browser, "Add shortcut to Drive"/download there.

## 4. Convert to image/mask pairs

```bash
conda activate distill
python scripts/prepare_ood_datasets.py cfd       --src raw/CrackForest-dataset                     --out data/ood/cfd
python scripts/prepare_ood_datasets.py gaps384   --src "raw/fphbn/pavement crack datasets/GAPS384"  --out data/ood/gaps384
python scripts/prepare_ood_datasets.py crkwh100  --src raw/CRKWH100                                 --out data/ood/crkwh100
python scripts/prepare_ood_datasets.py deepcrack --src data/datasets/deepcrack                      --out data/ood/deepcrack
```

Each run writes `images/*.png`, `masks/*.png` (0/255, same size as the image, EXIF ignored) and `meta.json`
(pair count, crack fraction) and fails loudly on a size mismatch. Tested on synthetic files in each format and on
the real DeepCrack test split (237 pairs, median crack fraction 3.4%). If CRKWH100's folder names differ from
`image/`·`gt/`, rename them — that layout is UNVERIFIED.

## 5. Importing to Kaggle (to evaluate there instead of on the laptop)

**Web:** kaggle.com → **Datasets → New Dataset** → drag in `ood.zip` (zip of `data/ood/`) → keep it **Private** →
Create. In a notebook, **+ Add Input** → it appears under `/kaggle/input/datasets/<you>/<slug>/` (older notebooks:
`/kaggle/input/<slug>/`).

**CLI:**

```bash
pip install kaggle          # API token: kaggle.com → Settings → API → Create New Token → ~/.kaggle/kaggle.json
kaggle datasets init -p data/ood
# edit data/ood/dataset-metadata.json: "title": "crack-ood-eval", "id": "<you>/crack-ood-eval",
#                                      "licenses": [{"name": "other"}]
kaggle datasets create -p data/ood -r zip              # private by default; -r zip keeps subfolders
kaggle datasets version -p data/ood -m "add CRKWH100" -r zip   # later updates
```

Without `-r zip` the CLI silently skips subfolders. Trained checkpoints (`best.pt`) can be uploaded the same way
as a second private dataset for an evaluation notebook.

## 6. Licenses

All of these are research-only or carry no license: CFD and DeepCrack are non-commercial research; GAPs is
academic use only (registration); Zou's sets are "for academic research"; Crack500, AEL, CrackTree have no
license file. Kaggle mirrors labelled CC0/Apache/MIT are relabelling, not a new license. Keep any upload
**private**; if something is ever published, leave out GAPs-derived images and cite every source.
