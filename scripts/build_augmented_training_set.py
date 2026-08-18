#!/usr/bin/env python3
"""
Builds data/datasets/crack500_yolo_augmented/ = the existing crack500_yolo
training set (1896 x 640x360 tiles) PLUS the mosaic composites mined by
mine_negative_and_mosaic_tiles.py, converted to YOLO-seg polygon labels via
convert_crack500.py's existing mask->polygon function. Val/test splits are
copied through unchanged (identical to crack500_yolo) so evaluation stays
comparable and uncontaminated.

Run (from ~/distill, after mine_negative_and_mosaic_tiles.py):
  python scripts/build_augmented_training_set.py
"""

import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
from scripts.convert_crack500 import binary_mask_to_yolo_instances

SRC_YOLO = ROOT / "data/datasets/crack500_yolo"
MINED_DIR = ROOT / "data/datasets/crack500_ood_mined"
OUT_DIR = ROOT / "data/datasets/crack500_yolo_augmented"


def copy_base_dataset():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    shutil.copytree(SRC_YOLO, OUT_DIR)
    print(f"[Base] Copied {SRC_YOLO} -> {OUT_DIR}")


def add_mosaics():
    mosaic_img_dir = MINED_DIR / "mosaic_images"
    mosaic_mask_dir = MINED_DIR / "mosaic_masks"
    out_img_dir = OUT_DIR / "images/train"
    out_lbl_dir = OUT_DIR / "labels/train"

    n_added = 0
    n_skipped_no_instances = 0
    for img_path in sorted(mosaic_img_dir.glob("*.jpg")):
        stem = img_path.stem  # e.g. 20160222_081011_mosaic
        mask_path = mosaic_mask_dir / f"{stem}.png"
        if not mask_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_lines = binary_mask_to_yolo_instances(str(mask_path), w, h)
        if not label_lines:
            n_skipped_no_instances += 1
            continue
        shutil.copy(img_path, out_img_dir / img_path.name)
        (out_lbl_dir / f"{stem}.txt").write_text("\n".join(label_lines) + "\n")
        n_added += 1

    print(f"[Mosaics] Added {n_added} composite images+labels to {out_img_dir}")
    if n_skipped_no_instances:
        print(f"[Mosaics] Skipped {n_skipped_no_instances} composites with 0 valid instances "
              f"(below MIN_AREA after connected-components)")


def add_negatives():
    neg_img_dir = MINED_DIR / "negative_images"
    neg_lbl_dir = MINED_DIR / "negative_labels"
    if not neg_img_dir.exists() or not any(neg_img_dir.iterdir()):
        print("[Negatives] None available (0 mined) — skipped. See possibleOODimprovements.md.")
        return
    out_img_dir = OUT_DIR / "images/train"
    out_lbl_dir = OUT_DIR / "labels/train"
    n = 0
    for img_path in sorted(neg_img_dir.glob("*.jpg")):
        shutil.copy(img_path, out_img_dir / img_path.name)
        lbl_src = neg_lbl_dir / f"{img_path.stem}.txt"
        shutil.copy(lbl_src, out_lbl_dir / lbl_src.name)
        n += 1
    print(f"[Negatives] Added {n} background-only crops (CAUTION: small sample, "
          f"sourced from photos that overlap the val/test split by filename — treat as a "
          f"pilot only, not a production fix).")


def main():
    copy_base_dataset()
    add_mosaics()
    add_negatives()
    n_train_imgs = len(list((OUT_DIR / "images/train").glob("*.jpg")))
    print(f"[Done] {OUT_DIR} ready — {n_train_imgs} total train images "
          f"(base {len(list((SRC_YOLO / 'images/train').glob('*.jpg')))} + additions)")


if __name__ == "__main__":
    main()
