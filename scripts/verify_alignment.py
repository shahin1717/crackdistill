#!/usr/bin/env python3
"""
scripts/verify_alignment.py

Geometric Coordinate Alignment Diagnostic for CrackDistill (Problem P0-1).
Verifies that letterbox coordinate warping maps SAM 2 teacher logits into
the exact physical pixel coordinates of the YOLO letterbox training canvas.

Usage:
    python scripts/verify_alignment.py [--max-samples 50] [--output-dir reports/overlays]
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to sys.path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))


def compute_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute Intersection-over-Union between two binary or soft mask tensors."""
    intersection = (mask1 * mask2).sum().item()
    union = (mask1 + mask2 > 0).float().sum().item()
    if union < eps:
        return 1.0 if intersection < eps else 0.0
    return float(intersection / (union + eps))


def main():
    parser = argparse.ArgumentParser(description="Verify Letterbox Coordinate Alignment between SAM and YOLO.")
    parser.add_argument("--dataset-yaml", type=str, default="data/datasets/crack500_yolo/dataset.yaml",
                        help="Path to YOLO dataset yaml.")
    parser.add_argument("--logits-dir", type=str, default="data/teacher_logits",
                        help="Directory containing precomputed teacher logits.")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Maximum samples to evaluate.")
    parser.add_argument("--output-dir", type=str, default="reports/overlays",
                        help="Output directory for visual overlay verification.")
    parser.add_argument("--min-median-iou", type=float, default=0.50,
                        help="Minimum acceptable median IoU threshold for warped alignment.")
    args = parser.parse_args()

    logits_path = Path(args.logits_dir).resolve()
    if not logits_path.exists():
        for alt in [ROOT / "data/teacher_logits_box", ROOT / "data/teacher_logits"]:
            if alt.exists():
                logits_path = alt
                break

    print("=" * 70)
    print("🔬 CrackDistill: Geometric Alignment Verification (P0-1)")
    print(f"  Dataset:    {args.dataset_yaml}")
    print(f"  Logits Dir: {logits_path}")
    print("=" * 70)

    try:
        from ultralytics.data.dataset import YOLODataset
    except ImportError:
        print("ERROR: ultralytics is required. Run in the distill environment.")
        sys.exit(1)

    ds_yaml_path = Path(args.dataset_yaml).resolve()
    if not ds_yaml_path.exists():
        print(f"ERROR: Dataset yaml not found at {ds_yaml_path}")
        sys.exit(1)

    with open(ds_yaml_path) as f:
        data_cfg = yaml.safe_load(f)

    # Load dataset in training mode with task='segment' and augment=False to test letterbox geometry
    img_train_dir = ds_yaml_path.parent / "images/train"
    dataset = YOLODataset(
        img_path=str(img_train_dir),
        data=data_cfg,
        imgsz=512,
        augment=False,
        task="segment"
    )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    unwarped_ious = []
    warped_ious = []
    valid_samples = 0
    visual_samples = []

    print(f"\nScanning dataset ({len(dataset)} items) for samples with precomputed logits...")

    for idx in range(len(dataset)):
        if valid_samples >= args.max_samples:
            break

        sample = dataset[idx]
        im_file = Path(sample.get("im_file", ""))
        stem = im_file.stem

        # Look for logits
        logit_file = None
        for prefix in [f"crack500_{stem}", f"deepcrack_{stem}", stem]:
            cand = logits_path / f"{prefix}_logits.npy"
            if cand.exists():
                logit_file = cand
                break

        if logit_file is None:
            continue

        # Load teacher logits: shape (N_teacher, H_sam, W_sam)
        try:
            sam_raw = np.load(str(logit_file))
            if sam_raw.ndim == 2:
                sam_raw = np.expand_dims(sam_raw, axis=0)
            sam_logits = torch.from_numpy(sam_raw).float()
        except Exception:
            continue

        gt_masks = sample.get("masks", None)
        if gt_masks is None or len(gt_masks) == 0:
            continue

        orig_h, orig_w = sample.get("ori_shape", (360, 640))
        (r_w, r_h), (pad_w, pad_h) = sample.get("ratio_pad", ((0.8, 0.8), (0.0, 112.0)))
        img_tensor = sample.get("img", None)  # (3, 512, 512)
        imgsz = img_tensor.shape[-2:] if img_tensor is not None else (512, 512)

        target_h, target_w = 256, 256
        scale_y = target_h / float(imgsz[0])
        scale_x = target_w / float(imgsz[1])

        # Compute letterbox padding in target space
        t_top = int(round(pad_h * scale_y))
        t_left = int(round(pad_w * scale_x))
        t_unpad_h = int(round(float(orig_h) * r_h * scale_y))
        t_unpad_w = int(round(float(orig_w) * r_w * scale_x))

        t_top = max(0, min(t_top, target_h - 1))
        t_left = max(0, min(t_left, target_w - 1))
        t_unpad_h = max(1, min(t_unpad_h, target_h - t_top))
        t_unpad_w = max(1, min(t_unpad_w, target_w - t_left))
        t_bottom = max(0, target_h - t_top - t_unpad_h)
        t_right = max(0, target_w - t_left - t_unpad_w)

        num_gt = gt_masks.shape[0]
        num_sam = sam_logits.shape[0]

        # Greedy matching based on warped IoU to find true instance pairings
        for g_idx in range(num_gt):
            gt_inst = gt_masks[g_idx].unsqueeze(0).unsqueeze(0)
            gt_resized = F.interpolate(gt_inst.float(), size=(target_h, target_w), mode="nearest").squeeze()
            gt_bin = (gt_resized > 0.5).float()

            if gt_bin.sum() < 15:
                continue

            best_w_iou = -1.0
            best_u_iou = -1.0
            best_sam_warped = None
            best_sam_unwarped = None

            for s_idx in range(num_sam):
                sam_inst = sam_logits[s_idx].unsqueeze(0).unsqueeze(0)

                # 1. Unwarped
                sam_unwarped = F.interpolate(sam_inst, size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze()
                sam_bin_u = (torch.sigmoid(sam_unwarped) > 0.35).float()
                u_iou = compute_mask_iou(sam_bin_u, gt_bin)

                # 2. Warped
                sam_unpad = F.interpolate(sam_inst, size=(t_unpad_h, t_unpad_w), mode="bilinear", align_corners=False)
                sam_warped = F.pad(sam_unpad, (t_left, t_right, t_top, t_bottom), mode="constant", value=-20.0).squeeze()
                sam_bin_w = (torch.sigmoid(sam_warped) > 0.35).float()
                w_iou = compute_mask_iou(sam_bin_w, gt_bin)

                if w_iou > best_w_iou:
                    best_w_iou = w_iou
                    best_u_iou = u_iou
                    best_sam_warped = sam_bin_w
                    best_sam_unwarped = sam_bin_u

            if best_w_iou >= 0.0:
                unwarped_ious.append(best_u_iou)
                warped_ious.append(best_w_iou)

                if len(visual_samples) < 6 and best_w_iou > 0.60:
                    visual_samples.append({
                        "stem": stem,
                        "img": img_tensor.permute(1, 2, 0).cpu().numpy(),
                        "gt": gt_bin.cpu().numpy(),
                        "unwarped": best_sam_unwarped.cpu().numpy(),
                        "warped": best_sam_warped.cpu().numpy(),
                        "iou_old": best_u_iou,
                        "iou_new": best_w_iou,
                    })

        valid_samples += 1

    if not warped_ious:
        print("ERROR: No valid instances evaluated. Check dataset and logits paths.")
        sys.exit(1)

    unwarped_arr = np.array(unwarped_ious)
    warped_arr = np.array(warped_ious)

    print("\n" + "=" * 70)
    print("📊 ALIGNMENT EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total instances evaluated: {len(warped_arr)} (across {valid_samples} samples)")
    print("-" * 70)
    print(f"Metric                       Unwarped (Old)       Warped (New / Fixed)")
    print(f"Mean IoU:                   {unwarped_arr.mean():.4f}               {warped_arr.mean():.4f}")
    print(f"Median IoU:                 {np.median(unwarped_arr):.4f}               {np.median(warped_arr):.4f}")
    print(f"5th Percentile IoU:         {np.percentile(unwarped_arr, 5):.4f}               {np.percentile(warped_arr, 5):.4f}")
    print(f"Fraction with IoU >= 0.70:  {(unwarped_arr >= 0.70).mean()*100:.1f}%                {(warped_arr >= 0.70).mean()*100:.1f}%")
    print(f"Fraction with IoU >= 0.85:  {(unwarped_arr >= 0.85).mean()*100:.1f}%                {(warped_arr >= 0.85).mean()*100:.1f}%")
    print("=" * 70)

    # Generate Visual Verification Overlay Grid
    if visual_samples:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n_rows = len(visual_samples)
            fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
            if n_rows == 1:
                axes = np.expand_dims(axes, 0)

            for r, s in enumerate(visual_samples):
                # 1. Letterboxed image
                img_disp = (s["img"] * 255).astype(np.uint8)
                axes[r, 0].imshow(img_disp)
                axes[r, 0].set_title(f"{s['stem']}\nLetterboxed YOLO Input (512x512)", fontsize=10)
                axes[r, 0].axis("off")

                # 2. GT Mask
                axes[r, 1].imshow(s["gt"], cmap="gray")
                axes[r, 1].set_title("Ground Truth Mask\n(In Letterbox Canvas)", fontsize=10)
                axes[r, 1].axis("off")

                # 3. Old Unwarped SAM
                axes[r, 2].imshow(s["unwarped"], cmap="autumn")
                axes[r, 2].set_title(f"Old SAM (Unwarped)\nIoU: {s['iou_old']:.4f} (MISALIGNED)", fontsize=10, color="red")
                axes[r, 2].axis("off")

                # 4. New Warped SAM
                axes[r, 3].imshow(s["warped"], cmap="spring")
                axes[r, 3].set_title(f"New SAM (Warped)\nIoU: {s['iou_new']:.4f} (ALIGNED ✓)", fontsize=10, color="green")
                axes[r, 3].axis("off")

            plt.tight_layout()
            out_img_path = out_dir / "alignment_verification.png"
            plt.savefig(str(out_img_path), dpi=150)
            plt.close()
            print(f"\n✓ Saved side-by-side visual overlays to: {out_img_path}")
        except Exception as e:
            print(f"Warning: Could not save visual overlays: {e}")

    # Acceptance Assertion
    median_warped = float(np.median(warped_arr))
    if median_warped >= args.min_median_iou:
        print(f"\n✅ PASS: Median Warped IoU ({median_warped:.4f}) meets acceptance threshold (>= {args.min_median_iou})!")
        return 0
    else:
        print(f"\n❌ FAIL: Median Warped IoU ({median_warped:.4f}) below threshold ({args.min_median_iou})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
