#!/usr/bin/env python3
"""
Crack500 Uncropped Test/Val → YOLO seg format converter
======================================================
Converts the original uncropped validation and test sets of Crack500.
Handles EXIF orientation for images by rotating the corresponding masks.

Source directories:
  data/datasets/crack500/valdata/   ← contains {stem}.jpg and {stem}_mask.png
  data/datasets/crack500/testdata/  ← contains {stem}.jpg and {stem}_mask.png

Output (YOLO seg format):
  data/datasets/crack500_uncropped_yolo/
  ├── images/
  │   ├── val/
  │   └── test/
  ├── labels/
  │   ├── val/
  │   └── test/
  └── dataset.yaml
"""

import os
import cv2
import numpy as np
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image


CLASS_ID = 0        # single class: crack
MIN_AREA = 50       # minimum pixel area to keep an instance
MIN_POINTS = 6      # minimum polygon points (3 coordinate pairs)


def get_exif_rotation(img_path: Path):
    """Retrieve EXIF orientation tag from image."""
    try:
        with Image.open(img_path) as im:
            exif = im.getexif()
            if exif:
                return exif.get(274)  # 274 is the Orientation tag
    except Exception:
        pass
    return None


def rotate_mask_to_match_image(mask: np.ndarray, exif_orientation: int) -> np.ndarray:
    """Rotate mask array to match image rotation applied by cv2.imread based on EXIF."""
    if exif_orientation == 6:
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    elif exif_orientation == 8:
        return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif exif_orientation == 3:
        return cv2.rotate(mask, cv2.ROTATE_180)
    return mask


def binary_mask_to_yolo_instances(mask_path: str, img_w: int, img_h: int, exif_orientation: int = None) -> list[str]:
    """
    Read binary PNG mask → rotate based on EXIF → split into instances via connectedComponents
    → convert each to normalized YOLO seg polygon string.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    if exif_orientation:
        mask = rotate_mask_to_match_image(mask, exif_orientation)

    # Threshold (Crack500 masks are binary 0/255)
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


def process_split(src_dir: Path, dst_dir: Path, split_name: str):
    """Process uncropped val or test split."""
    split_dir = src_dir / f"{split_name}data"
    if not split_dir.exists():
        print(f"  [Warning] Directory {split_dir} does not exist, skipping split {split_name}.")
        return 0

    dst_img_dir = dst_dir / "images" / split_name
    dst_lbl_dir = dst_dir / "labels" / split_name

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files (jpg/jpeg/png that do not contain '_mask')
    all_files = sorted(split_dir.iterdir())
    image_files = [
        f for f in all_files 
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        and '_mask' not in f.name.lower()
        and ':Zone.Identifier' not in f.name
    ]

    converted = 0
    skipped = 0

    for img_path in tqdm(image_files, desc=f"  {split_name}", leave=False):
        stem = img_path.stem

        # Find mask (stem + "_mask.png")
        mask_path = split_dir / f"{stem}_mask.png"
        if not mask_path.exists():
            skipped += 1
            continue

        # Read image to get dimensions (matches how cv2.imread auto-rotates it based on EXIF)
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        # Get EXIF rotation from image
        exif_orientation = get_exif_rotation(img_path)

        # Convert mask to YOLO seg labels (rotating it to match)
        label_lines = binary_mask_to_yolo_instances(str(mask_path), w, h, exif_orientation)

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


def main():
    parser = argparse.ArgumentParser(description="Convert Crack500 Uncropped splits to YOLO seg format")
    parser.add_argument(
        "--src",
        type=str,
        default="data/datasets/crack500",
        help="Path to crack500 root dir"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/datasets/crack500_uncropped_yolo",
        help="Output directory"
    )
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    print(f"[Convert] Source: {src}")
    print(f"[Convert] Output: {dst}")
    print()

    if not src.exists():
        print(f"ERROR: Source directory not found: {src}")
        return

    if dst.exists():
        print(f"[Warning] Output directory exists, clearing: {dst}")
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in ["val", "test"]:
        n = process_split(src, dst, split)
        counts[split] = n

    # Write dataset_uncropped.yaml
    yaml_content = f"""# Crack500 Uncropped — YOLO seg format
# Auto-generated by convert_crack500_uncropped.py

path: {dst.resolve()}
train: images/val
val:   images/val
test:  images/test

nc: 1
names:
  0: crack

# Stats
# val:   ~{counts.get('val', 0)} images (uncropped)
# test:  ~{counts.get('test', 0)} images (uncropped)
"""
    with open(dst / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    print(f"\n  dataset.yaml written to {dst / 'dataset.yaml'}")
    print(f"[Done] Converted uncropped splits successfully.")


if __name__ == "__main__":
    main()
