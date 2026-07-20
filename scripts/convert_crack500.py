#!/usr/bin/env python3
"""
Crack500 → YOLO seg format converter
=====================================
Crack500 structure:
  crack500/
  ├── traincrop/   ← 00001.jpg + 00001.png (binary mask, same stem)
  ├── valcrop/
  ├── testcrop/
  ├── train.txt    ← list of image filenames (optional)
  ├── val.txt
  └── test.txt

Output (YOLO seg format, ready for ultralytics):
  crack500_yolo/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── labels/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── dataset.yaml

Each .txt label: one line per connected crack instance
  0 x1 y1 x2 y2 ... (normalized polygon, class 0 = crack)

Usage:
  python scripts/convert_crack500.py \
      --src ~/distill/data/datasets/crack500 \
      --dst ~/distill/data/datasets/crack500_yolo
"""

import os
import cv2
import numpy as np
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


CLASS_ID = 0        # single class: crack
MIN_AREA = 50       # minimum pixel area to keep an instance
MIN_POINTS = 6      # minimum polygon points (3 coordinate pairs)


def binary_mask_to_yolo_instances(mask_path: str, img_w: int, img_h: int) -> list[str]:
    """
    Read binary PNG mask → split into instances via connectedComponents
    → convert each to normalized YOLO seg polygon string.

    Returns list of label lines (one per instance).
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # Threshold (Crack500 masks are 0/255)
    binary = (mask > 127).astype(np.uint8)

    # Separate touching cracks into individual instances
    num_labels, labels_map = cv2.connectedComponents(binary)

    label_lines = []
    for label_id in range(1, num_labels):      # 0 = background
        instance = (labels_map == label_id).astype(np.uint8)

        if instance.sum() < MIN_AREA:
            continue

        # Find contours for this instance
        contours, _ = cv2.findContours(
            instance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if len(contour) < MIN_POINTS // 2:
                continue

            # Flatten and normalize to [0, 1]
            pts = contour.squeeze()
            if pts.ndim == 1:
                pts = pts.reshape(1, 2)

            # Simplify contour slightly to reduce file size
            epsilon = 0.002 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True).squeeze()
            if simplified.ndim == 1:
                simplified = simplified.reshape(1, 2)
            if len(simplified) < 3:
                simplified = pts

            norm = []
            for x, y in simplified:
                norm.append(x / img_w)
                norm.append(y / img_h)

            if len(norm) < MIN_POINTS:
                continue

            coords_str = " ".join(f"{v:.6f}" for v in norm)
            label_lines.append(f"{CLASS_ID} {coords_str}")

    return label_lines


def process_split(src_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path, split_name: str):
    """Process one split (train/val/test)."""

    # Crack500 stores images+masks together in traincrop/valcrop/testcrop
    crop_dir = src_dir / f"{split_name}crop"
    if not crop_dir.exists():
        # Try alternate names
        for candidate in [src_dir / split_name, src_dir / f"{split_name}data"]:
            if candidate.exists():
                crop_dir = candidate
                break
        else:
            print(f"  [WARNING] Could not find directory for split '{split_name}', skipping.")
            return 0

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Find all images (jpg/jpeg/png that are NOT masks)
    all_files = sorted(crop_dir.iterdir())
    # Crack500: image = .jpg, mask = same stem + .png
    image_files = [f for f in all_files if f.suffix.lower() in ('.jpg', '.jpeg')
                   and ':Zone.Identifier' not in f.name]

    if not image_files:
        # Some versions store as .png images too — distinguish by paired files
        png_files = [f for f in all_files if f.suffix.lower() == '.png'
                     and ':Zone.Identifier' not in f.name]
        # If .jpg exists for a stem → .png is mask. If no .jpg → .png is image.
        jpg_stems = {f.stem for f in all_files if f.suffix.lower() in ('.jpg', '.jpeg')}
        image_files = [f for f in png_files if f.stem not in jpg_stems]

    converted = 0
    skipped   = 0

    for img_path in tqdm(image_files, desc=f"  {split_name}", leave=False):
        stem = img_path.stem

        # Find corresponding mask (.png with same stem)
        mask_path = crop_dir / f"{stem}.png"
        if not mask_path.exists():
            # Try .bmp
            mask_path = crop_dir / f"{stem}.bmp"
        if not mask_path.exists():
            skipped += 1
            continue

        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        # Convert mask to YOLO seg labels
        label_lines = binary_mask_to_yolo_instances(str(mask_path), w, h)

        # Copy image
        dst_img_path = dst_img_dir / img_path.name
        shutil.copy2(img_path, dst_img_path)

        # Write label file (even if empty — YOLO needs it)
        dst_lbl_path = dst_lbl_dir / f"{stem}.txt"
        with open(dst_lbl_path, "w") as f:
            f.write("\n".join(label_lines))

        converted += 1

    print(f"  {split_name}: {converted} images converted, {skipped} skipped")
    return converted


def write_dataset_yaml(dst: Path, num_train: int, num_val: int, num_test: int):
    """Write ultralytics-compatible dataset.yaml."""
    yaml_content = f"""# Crack500 — YOLO seg format
# Auto-generated by convert_crack500.py

path: {dst.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: crack

# Stats
# train: ~{num_train} images
# val:   ~{num_val} images
# test:  ~{num_test} images
"""
    with open(dst / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    print(f"\n  dataset.yaml written to {dst / 'dataset.yaml'}")


def verify_conversion(dst: Path):
    """Quick sanity check on converted dataset."""
    print("\n[Verify] Checking converted dataset...")
    issues = 0
    for split in ["train", "val", "test"]:
        img_dir = dst / "images" / split
        lbl_dir = dst / "labels" / split
        if not img_dir.exists():
            continue

        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        lbls = list(lbl_dir.glob("*.txt"))

        # Check counts match
        if len(imgs) != len(lbls):
            print(f"  [!] {split}: {len(imgs)} images vs {len(lbls)} labels — mismatch!")
            issues += 1
        else:
            print(f"  {split}: {len(imgs)} images ✓")

        # Check a few labels are non-empty
        non_empty = sum(1 for l in lbls if l.stat().st_size > 0)
        empty     = len(lbls) - non_empty
        print(f"    labels with cracks: {non_empty} | empty (no crack): {empty}")

        if non_empty == 0:
            print(f"  [!] {split}: ALL labels are empty — check mask paths!")
            issues += 1

    if issues == 0:
        print("\n  ✓ Dataset looks good!")
    else:
        print(f"\n  ✗ {issues} issue(s) found — check output above.")

    return issues == 0


def main():
    parser = argparse.ArgumentParser(description="Convert Crack500 to YOLO seg format")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to crack500 root dir (contains traincrop/, valcrop/, testcrop/)"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Output directory (default: <src>_yolo)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run sanity check after conversion"
    )
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve() if args.dst else src.parent / f"{src.name}_yolo"

    print(f"[Convert] Source: {src}")
    print(f"[Convert] Output: {dst}")
    print()

    if not src.exists():
        print(f"ERROR: Source directory not found: {src}")
        return

    counts = {}
    for split in ["train", "val", "test"]:
        n = process_split(
            src_dir    = src,
            dst_img_dir= dst / "images" / split,
            dst_lbl_dir= dst / "labels" / split,
            split_name = split,
        )
        counts[split] = n

    write_dataset_yaml(dst, counts["train"], counts["val"], counts["test"])

    if args.verify:
        verify_conversion(dst)

    print(f"\n[Done] Converted dataset at: {dst}")
    print(f"\nNext step — test YOLO11 loads it:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('yolo11n-seg.pt')")
    print(f"  model.train(data='{dst}/dataset.yaml', epochs=1, imgsz=512)")


if __name__ == "__main__":
    main()
# (appended — nothing, file is complete)
