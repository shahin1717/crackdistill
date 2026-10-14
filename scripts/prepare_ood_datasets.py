#!/usr/bin/env python3
"""
Convert a downloaded crack dataset into image/mask PNG pairs for zero-shot evaluation.

Output: <out>/images/<stem>.png, <out>/masks/<stem>.png (0 / 255, same size), <out>/meta.json.
Images are opened with PIL, which does not apply EXIF rotation, and every pair must have equal sizes.
Where each dataset comes from and why each rule is what it is: docs/DATASETS.md.

Usage:
  python scripts/prepare_ood_datasets.py cfd       --src raw/CrackForest-dataset --out data/ood/cfd
  python scripts/prepare_ood_datasets.py gaps384   --src raw/GAPS384             --out data/ood/gaps384
  python scripts/prepare_ood_datasets.py deepcrack --src data/datasets/deepcrack --out data/ood/deepcrack
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _gray(path):
    return np.array(Image.open(path).convert("L"))


def _cfd_mask(path):
    """CFD .mat: Segmentation 1 = background, 2 = crack (3/4 are other segment ids, not crack); FPHBN PNG: > 0."""
    if path.suffix == ".mat":
        import scipy.io as sio
        return sio.loadmat(path, squeeze_me=True, struct_as_record=False)["groundTruth"].Segmentation == 2
    return _gray(path) > 0


def _auto_polarity(path):
    """Thin-label sets with unverified polarity: crack is the minority class."""
    m = _gray(path) >= 128
    return ~m if m.mean() > 0.5 else m


def _pairs_by_stem(img_dir, gt_dir, gt_stem=lambda s: s):
    gts = {p.stem: p for p in Path(gt_dir).rglob("*") if p.suffix.lower() in IMG_EXT | {".mat"}}
    for img in sorted(p for p in Path(img_dir).rglob("*") if p.suffix.lower() in IMG_EXT):
        gt = gts.get(gt_stem(img.stem))
        if gt is not None:
            yield img, gt


def _ael_gt_stem(stem):
    """'Im_GT_AIGLE_RN_C18aor' -> 'GT_AIGLE_RN_C18a'."""
    s = stem[3:] if stem.startswith("Im_") else stem
    return s[:-2] if s.endswith("or") else s


def _first_dir(src, *names):
    for n in names:
        if (Path(src) / n).is_dir():
            return Path(src) / n
    raise FileNotFoundError(f"none of {names} under {src}")


# name -> (pairs(src), mask(gt_path), thin 1-px labels?)
DATASETS = {
    "cfd": (lambda s: _pairs_by_stem(_first_dir(s, "image", "cfd_image"), _first_dir(s, "groundTruth", "seg_gt")),
            _cfd_mask, False),
    "gaps384": (lambda s: _pairs_by_stem(_first_dir(s, "croppedimg"), _first_dir(s, "croppedgt")),
                lambda p: _gray(p) >= 128, False),
    "ael": (lambda s: _pairs_by_stem(_first_dir(s, "img"), _first_dir(s, "gt"), _ael_gt_stem),
            lambda p: _gray(p) < 128, False),
    "cracktree200": (lambda s: _pairs_by_stem(_first_dir(s, "cracktree200rgb"), _first_dir(s, "cracktree200_gt")),
                     lambda p: _gray(p) >= 128, True),
    "deepcrack": (lambda s: _pairs_by_stem(_first_dir(s, "test_img"), _first_dir(s, "test_lab")),
                  lambda p: _gray(p) >= 128, False),
    "crkwh100": (lambda s: _pairs_by_stem(_first_dir(s, "image", "images", "img"), _first_dir(s, "gt", "label", "labels")),
                 _auto_polarity, True),
}


def prepare(name, src, out):
    pairs_fn, mask_fn, thin = DATASETS[name]
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    n, fracs = 0, []
    for img_path, gt_path in pairs_fn(Path(src)):
        img = Image.open(img_path).convert("RGB")
        mask = mask_fn(gt_path)
        if img.size != (mask.shape[1], mask.shape[0]):
            raise ValueError(f"size mismatch {img_path} {img.size} vs {gt_path} {mask.shape[::-1]}")
        stem = img_path.stem if name != "ael" else _ael_gt_stem(img_path.stem)
        img.save(out / "images" / f"{stem}.png")
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L").save(out / "masks" / f"{stem}.png")
        fracs.append(float(mask.mean()))
        n += 1
    if n == 0:
        raise FileNotFoundError(f"no image/mask pairs found for {name} under {src}")
    meta = {"dataset": name, "source": str(src), "n_pairs": n, "thin_labels": thin,
            "crack_fraction_median": float(np.median(fracs)), "crack_fraction_max": float(np.max(fracs))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if meta["crack_fraction_max"] > 0.5:
        print(f"[Warning] some {name} masks are >50% crack — check the polarity rule", file=sys.stderr)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name", choices=sorted(DATASETS))
    ap.add_argument("--src", required=True, help="Extracted raw dataset folder")
    ap.add_argument("--out", required=True, help="Output folder, e.g. data/ood/<name>")
    a = ap.parse_args()
    print(json.dumps(prepare(a.name, a.src, a.out), indent=2))


if __name__ == "__main__":
    main()
