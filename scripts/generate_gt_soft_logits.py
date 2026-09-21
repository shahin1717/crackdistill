#!/usr/bin/env python3
"""
Generate Gaussian-Softened Ground-Truth Pseudo-Logits (GT-Soft Control Arm).
Addresses Problem P2-2:
- Generates soft target logits directly from ground-truth instance annotations using 2D Gaussian smoothing (sigma=2.0).
- Establishes the empirical baseline required to prove that SAM 2 distillation gains stem from genuine foundation model priors rather than simple soft-boundary regularization.
- Saves soft targets to data/teacher_logits_gt_soft/ in identical (M, 256, 256) float32 numpy format as SAM 2 logits.

Usage:
  python scripts/generate_gt_soft_logits.py --data data/datasets/crack500_yolo --out data/teacher_logits_gt_soft --sigma 2.0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from tqdm import tqdm


def generate_soft_logit_from_mask(
    mask: np.ndarray,
    sigma: float = 2.0,
    target_res: Optional[int] = None,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Convert a single binary mask into a Gaussian-softened pseudo-logit.

    Args:
        mask (np.ndarray): Binary mask (2D array, 0 or 1/255).
        sigma (float): Gaussian blur standard deviation.
        target_res (int, optional): Optional square resolution to resize to.
        eps (float): Numerical safety bound for inverse sigmoid.

    Returns:
        np.ndarray: Soft logit 2D array, float32.
    """
    if target_res is not None and mask.shape[:2] != (target_res, target_res):
        m = cv2.resize(mask.astype(np.float32), (target_res, target_res), interpolation=cv2.INTER_LINEAR)
    else:
        m = mask.astype(np.float32)

    if m.max() > 1.0:
        m = m / 255.0

    ksize = int(math_ceil(sigma * 6)) | 1
    ksize = max(5, min(ksize, 31))
    soft_prob = cv2.GaussianBlur(m, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    soft_prob = np.clip(soft_prob, eps, 1.0 - eps)
    return np.log(soft_prob / (1.0 - soft_prob)).astype(np.float32)


def generate_gt_soft_for_image(
    label_path: Path,
    img_h: int = 640,
    img_w: int = 640,
    sigma: float = 2.0,
    target_res: int = 256,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Convert YOLO segmentation polygon labels into Gaussian-softened pseudo-logits.

    Args:
        label_path: Path to YOLO label .txt file containing polygon instances.
        img_h: Image canvas height.
        img_w: Image canvas width.
        sigma: Gaussian smoothing parameter (default 2.0).
        target_res: Output square resolution for teacher logits (default 256).
        eps: Probability clipping parameter for logit inversion.

    Returns:
        np.ndarray: Soft logit tensor of shape (M, target_res, target_res), float32.
    """
    if not label_path.exists():
        return np.zeros((0, target_res, target_res), dtype=np.float32)

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    instances = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            coords = [float(x) for x in parts[1:]]
            pts = np.array(coords).reshape(-1, 2)
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h
            inst = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(inst, [pts.astype(np.int32)], 1)
            if inst.sum() >= 20:
                instances.append(inst)
        except Exception:
            continue

    if not instances:
        return np.zeros((0, target_res, target_res), dtype=np.float32)

    logits_list = [
        generate_soft_logit_from_mask(inst, sigma=sigma, target_res=target_res, eps=eps)
        for inst in instances
    ]
    return np.stack(logits_list, axis=0)


def math_ceil(x: float) -> int:
    return int(np.ceil(x))


def main():
    parser = argparse.ArgumentParser(description="Generate GT-Soft control logits from ground-truth labels.")
    parser.add_argument("--data", type=str, default="data/datasets/crack500_yolo", help="Path to YOLO dataset directory")
    parser.add_argument("--out", type=str, default="data/teacher_logits_gt_soft", help="Output directory for GT-soft logits")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing sigma (default: 2.0)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to generate (default: train)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    labels_dir = data_dir / "labels" / args.split
    images_dir = data_dir / "images" / args.split
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not labels_dir.exists():
        print(f"Labels directory '{labels_dir}' does not exist.")
        return

    label_files = sorted(list(labels_dir.glob("*.txt")))
    print(f"Generating GT-Soft pseudo-logits (sigma={args.sigma}) for {len(label_files)} images in '{labels_dir}'...")

    generated = 0
    skipped = 0

    for lf in tqdm(label_files, desc="GT-Soft generation"):
        out_file = out_dir / f"{lf.stem}.npy"
        if out_file.exists():
            skipped += 1
            continue

        # Check image dimensions if image file is present
        img_h, img_w = 640, 640
        for ext in [".jpg", ".png", ".bmp"]:
            img_p = images_dir / f"{lf.stem}{ext}"
            if img_p.exists():
                im = cv2.imread(str(img_p))
                if im is not None:
                    img_h, img_w = im.shape[:2]
                break

        logits = generate_gt_soft_for_image(lf, img_h=img_h, img_w=img_w, sigma=args.sigma, target_res=256)
        if logits.shape[0] > 0:
            np.save(str(out_file), logits)
            generated += 1

    print(f"✓ GT-Soft logits completed: {generated} generated, {skipped} skipped, {len(label_files)} total.")


if __name__ == "__main__":
    main()
