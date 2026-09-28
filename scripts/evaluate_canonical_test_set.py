#!/usr/bin/env python3
"""
Evaluate trained checkpoints on the untouched canonical Crack500 test sets.
Addresses Problem P2-3:
- Evaluates on the held-out 1,124-image in-domain test set (crack500/testcrop).
- Evaluates on the held-out 200-image full-resolution uncropped test set (crack500/testdata).
- Implements strict split isolation: ensures test data is evaluated once without hyperparameter feedback loops.
- Generates structured evaluation report with SHA256 provenance tracking.

Usage:
  python scripts/evaluate_canonical_test_set.py --weights path/to/best.pt --arm mask_kd --seed 42
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.checkpoint import resolve_checkpoint, compute_file_sha256, get_checkpoint_manifest
from inference.tiled_inference import tiled_predict_image_gaussian


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice similarity coefficient between two binary masks."""
    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / total)


compute_dice_score = compute_dice


def load_canonical_crops(dataset_dir: Path, split: str = "test") -> list:
    """Load metadata for canonical image crops in the dataset split."""
    dataset_dir = Path(dataset_dir)
    img_dir = dataset_dir / "images" / split
    lbl_dir = dataset_dir / "labels" / split

    crops = []
    if not img_dir.exists():
        return crops

    for p in sorted(img_dir.glob("*.*")):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            lbl_p = lbl_dir / f"{p.stem}.txt"
            crops.append({
                "stem": p.stem,
                "image_path": p,
                "label_path": lbl_p if lbl_p.exists() else None,
            })
    return crops


def list_uncropped_scenes(uncropped_dir: Path) -> list:
    """Full-resolution test photos (any .jpg/.JPG/.png case), excluding mask files."""
    return sorted(
        p for p in Path(uncropped_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.stem.endswith(("_mask", "_lab"))
    )


def load_scene(img_path: Path):
    """
    Load a photo and its GT mask in the same pixel frame, or (None, None).

    Crack500 masks are annotated on the raw sensor pixels, while ~65% of the test photos carry an
    EXIF rotation that cv2.imread would apply by default; read the raw pixels so they match.
    """
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    gt_mask_path = img_path.parent / f"{img_path.stem}_mask.png"
    if not gt_mask_path.exists():
        gt_mask_path = img_path.parent / f"{img_path.stem}.png"
    gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE) if gt_mask_path.exists() else None
    if img_bgr is None or gt_mask is None or gt_mask.shape[:2] != img_bgr.shape[:2]:
        return None, None
    return img_bgr, (gt_mask > 127).astype(np.uint8)


def evaluate_checkpoint_on_test_set(
    checkpoint_path: Path,
    data_yaml: str = "data/datasets/crack500_yolo/dataset.yaml",
    uncropped_dir: Path = Path("data/datasets/crack500/testdata"),
    output_dir: Path = Path("results"),
    conf: float = 0.25,
    arm: str = None,
    seed: int = None,
) -> dict:
    """
    Run full canonical test set evaluation on a trained YOLOv11 checkpoint.
    """
    from ultralytics import YOLO

    resolved_ckpt = resolve_checkpoint(checkpoint_path)
    sha256 = compute_file_sha256(resolved_ckpt)
    model = YOLO(str(resolved_ckpt))

    print("=" * 60)
    print(f"CANONICAL TEST SET BENCHMARK")
    print(f"Checkpoint: {resolved_ckpt}")
    print(f"SHA256    : {sha256}")
    print("=" * 60)

    # 1. In-Domain Cropped Canonical Test Set (1,124 images)
    print("\n[1/2] Evaluating In-Domain Canonical Test Set (split='test')...")
    test_metrics = {}
    try:
        val_res = model.val(data=data_yaml, split="test", verbose=True)
        test_metrics["in_domain_mask_mAP50"] = float(val_res.seg.map50)
        test_metrics["in_domain_mask_mAP50_95"] = float(val_res.seg.map)
        test_metrics["in_domain_box_mAP50"] = float(val_res.box.map50)
        test_metrics["in_domain_box_mAP50_95"] = float(val_res.box.map)
    except Exception as e:
        print(f"[Warning] In-domain test val evaluation failed or test split missing: {e}")

    # 2. Out-of-Distribution Uncropped Test Set (200 images, tiled sliding-window)
    print("\n[2/2] Evaluating Full-Resolution Uncropped Test Set with Gaussian Tiled Inference...")
    tiled_dices = []
    direct_dices = []

    test_images = list_uncropped_scenes(uncropped_dir) if uncropped_dir.exists() else []
    skipped = 0

    if test_images:
        print(f"Found {len(test_images)} uncropped test scenes in '{uncropped_dir}'. Evaluating...")
        for img_path in tqdm(test_images, desc="Tiled Test Eval"):
            img_bgr, gt_binary = load_scene(img_path)
            if img_bgr is None:
                skipped += 1
                continue
            h, w = img_bgr.shape[:2]

            # Direct Resize Prediction (retina_masks: masks mapped back to the original frame, letterbox removed)
            r_dir = model.predict(img_bgr, imgsz=512, conf=conf, verbose=False, retina_masks=True)[0]
            pred_dir = np.zeros((h, w), dtype=np.uint8)
            if r_dir.masks is not None and len(r_dir.masks) > 0:
                pred_dir = (r_dir.masks.data.cpu().numpy().max(0) > 0.5).astype(np.uint8)
            direct_dices.append(compute_dice(pred_dir, gt_binary))

            # Gaussian Tiled Prediction (512x512 patches, 384 stride, 2D Gaussian apodization)
            pred_tiled = tiled_predict_image_gaussian(model, img_bgr, tile_size=512, stride=384, conf=conf)
            tiled_dices.append(compute_dice(pred_tiled, gt_binary))

    mean_direct_dice = float(np.mean(direct_dices)) if direct_dices else 0.0
    mean_tiled_dice = float(np.mean(tiled_dices)) if tiled_dices else 0.0

    test_metrics["n_uncropped_scenes"] = len(tiled_dices)
    test_metrics["n_uncropped_skipped"] = skipped if test_images else 0
    test_metrics["ood_uncropped_direct_dice"] = mean_direct_dice
    test_metrics["ood_uncropped_tiled_dice"] = mean_tiled_dice
    test_metrics["delta_tiled_dice"] = mean_tiled_dice - mean_direct_dice
    test_metrics["arm"] = arm
    test_metrics["seed"] = seed
    test_metrics["checkpoint_path"] = str(resolved_ckpt)
    test_metrics["sha256"] = sha256
    test_metrics["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print("\n" + "=" * 60)
    print("FINAL CANONICAL TEST BENCHMARK RESULTS:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
    print("=" * 60)

    # Save structured summary
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{arm}_seed{seed}" if arm is not None and seed is not None else resolved_ckpt.parent.parent.name
    out_file = output_dir / f"canonical_test_benchmark_{tag}_{sha256[:8]}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"✓ Saved canonical test benchmark to: {out_file}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on canonical Crack500 test sets.")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--data", type=str, default="data/datasets/crack500_yolo/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--uncropped", type=str, default="data/datasets/crack500/testdata", help="Directory containing uncropped test images")
    parser.add_argument("--out", type=str, default="results/test", help="Output directory for test results")
    parser.add_argument("--arm", type=str, default=None, help="Experimental arm (baseline, mask_kd, gt_soft) for aggregation")
    parser.add_argument("--seed", type=int, default=None, help="Training seed of this checkpoint, for seed-paired aggregation")
    args = parser.parse_args()

    evaluate_checkpoint_on_test_set(
        checkpoint_path=Path(args.weights),
        data_yaml=args.data,
        uncropped_dir=Path(args.uncropped),
        output_dir=Path(args.out),
        arm=args.arm,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
