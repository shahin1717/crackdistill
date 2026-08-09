#!/usr/bin/env python3
"""
Generate SAM 2 teacher logits for crack500 training images.
Run ONCE before training. Saves .npy logit files to data/teacher_logits/

Usage (from ~/distill):
  python scripts/generate_teacher_logits.py           # full dataset
  python scripts/generate_teacher_logits.py --resume  # skip already done
  python scripts/generate_teacher_logits.py --max 50  # only first N images
"""

import argparse
import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path

# ── project root on path ─────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# ── fixed paths (relative to ~/distill) ─────────────────────
DATASET_DIR = ROOT / "data/datasets/crack500"
LOGITS_DIR  = ROOT / "data/teacher_logits"
SAM2_CKPT   = ROOT / "checkpoints/sam2_hiera_large.pt"
SAM2_CFG    = "configs/sam2.1/sam2.1_hiera_l.yaml"


def mask_to_bbox(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r1, r2 = np.where(rows)[0][[0, -1]]
    c1, c2 = np.where(cols)[0][[0, -1]]
    return np.array([c1, r1, c2, r2], dtype=np.float32)


def mask_to_centroid(mask: np.ndarray):
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        h, w = mask.shape[:2]
        if 0 <= cX < w and 0 <= cY < h and mask[cY, cX] > 0:
            return np.array([[cX, cY]], dtype=np.float32)
    # Fallback to maximum of distance transform (guaranteed to be inside mask)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
    cX, cY = max_loc
    return np.array([[cX, cY]], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip images that already have logits")
    parser.add_argument("--max", type=int, default=None,
                        help="Max images to process (for testing)")
    parser.add_argument("--img-dir", type=str, default=None,
                        help="Path to directory of training images")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Path to directory of training masks")
    parser.add_argument("--prefix", type=str, default="crack500_",
                        help="Prefix to use for saved logits files")
    parser.add_argument("--prompt-type", type=str, default="box_centroid",
                        choices=["box", "box_centroid"],
                        help="Prompt type to use (box or box_centroid)")
    parser.add_argument("--logits-dir", type=str, default=None,
                        help="Override output logits directory")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to YOLO dataset or raw dataset directory")
    args = parser.parse_args()

    if args.dataset and not args.img_dir:
        ds_path = Path(args.dataset).expanduser().resolve()
        if (ds_path / "images/train").exists():
            args.img_dir = str(ds_path / "images/train")
            args.mask_dir = str(ds_path / "masks/train") if (ds_path / "masks/train").exists() else str(ds_path / "images/train")
        elif ds_path.exists():
            args.img_dir = str(ds_path)
            args.mask_dir = str(ds_path)

    if args.logits_dir:
        logits_dir = Path(args.logits_dir).expanduser().resolve()
    else:
        logits_dir = ROOT / "data/teacher_logits"
        
    orig_logits_dir = logits_dir
    # On Kaggle, redirect to /tmp/ to avoid exceeding 20GB disk limit
    if "/kaggle/" in str(logits_dir):
        logits_dir = Path("/tmp") / logits_dir.name
        logits_dir.mkdir(parents=True, exist_ok=True)
        try:
            if os.path.lexists(orig_logits_dir):
                if os.path.islink(orig_logits_dir):
                    os.unlink(orig_logits_dir)
                elif orig_logits_dir.is_dir() and not list(orig_logits_dir.glob("*.npy")):
                    orig_logits_dir.rmdir()
            if not orig_logits_dir.exists():
                orig_logits_dir.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(logits_dir, orig_logits_dir)
                print(f"[Logits] Created symlink: {orig_logits_dir} -> {logits_dir}")
        except Exception as e:
            print(f"[Logits Warning] Could not symlink {orig_logits_dir} -> {logits_dir}: {e}")
        
    logits_dir.mkdir(parents=True, exist_ok=True)

    if args.img_dir:
        # Single custom dataset mode
        datasets = [{
            "name": "Custom",
            "img_dir": Path(args.img_dir).expanduser().resolve(),
            "mask_dir": Path(args.mask_dir).expanduser().resolve() if args.mask_dir else Path(args.img_dir).expanduser().resolve(),
            "prefix": args.prefix
        }]
    else:
        # Multi-dataset auto-mode
        datasets = [
            {
                "name": "Crack500",
                "img_dir": ROOT / "data/datasets/crack500/traincrop",
                "mask_dir": ROOT / "data/datasets/crack500/traincrop",
                "prefix": "crack500_"
            },
            {
                "name": "DeepCrack",
                "img_dir": ROOT / "data/datasets/deepcrack/train_img",
                "mask_dir": ROOT / "data/datasets/deepcrack/train_lab",
                "prefix": "deepcrack_"
            }
        ]

    # Filter out non-existent datasets
    active_datasets = []
    for d in datasets:
        if d["img_dir"].exists():
            active_datasets.append(d)
        else:
            if args.img_dir:
                print(f"ERROR: Image directory not found: {d['img_dir']}")
                return
            else:
                print(f"Skipping {d['name']} logits generation: directory {d['img_dir']} does not exist.")

    if not active_datasets:
        print("No active datasets to process. Exiting.")
        return

    # ── Load SAM 2 ───────────────────────────────────────────
    print(f"Loading SAM 2 from {SAM2_CKPT}...")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model     = build_sam2(SAM2_CFG, str(SAM2_CKPT), device="cuda")
    predictor = SAM2ImagePredictor(model)
    print("SAM 2 loaded ✓\n")

    for d in active_datasets:
        img_dir = d["img_dir"]
        mask_dir = d["mask_dir"]
        prefix = d["prefix"]

        # ── Process images ───────────────────────────────────────
        # Check for jpg/jpeg files first to avoid loading PNG masks as images in Crack500 traincrop
        images = sorted([
            f for f in img_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg')
            and ':Zone.Identifier' not in f.name
        ])
        if not images:
            images = sorted([
                f for f in img_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
                and ':Zone.Identifier' not in f.name
            ])
        if args.max:
            images = images[:args.max]

        print(f"\n>>> Processing {d['name']} ({len(images)} images) <<<")
        print(f"  Images: {img_dir}")
        print(f"  Masks:  {mask_dir}")
        print(f"  Prefix: {prefix}\n")

        generated = 0
        skipped   = 0
        failed    = 0

        for img_path in images:
            image_id   = f"{prefix}{img_path.stem}"
            logit_file = logits_dir / f"{image_id}_logits.npy"

            # Skip if already done
            if args.resume and logit_file.exists():
                skipped += 1
                continue

            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                failed += 1
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load instance masks from binary PNG/BMP file or YOLO label .txt file
            instance_masks = []

            mask_path = mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                mask_path = mask_dir / f"{img_path.stem}.bmp"
            
            # Check raw dataset fallback paths (e.g. train_lab / traincrop)
            if not mask_path.exists():
                for alt_name in ["train_lab", "traincrop", "masks"]:
                    alt_path = img_dir.parent.parent / alt_name / f"{img_path.stem}.png"
                    if alt_path.exists():
                        mask_path = alt_path
                        break

            # Check YOLO txt label fallback
            label_txt_path = img_dir.parent.parent / "labels" / img_dir.name / f"{img_path.stem}.txt"

            if mask_path.exists():
                binary = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if binary is not None:
                    binary = (binary > 127).astype(np.uint8)
                    num_labels, labels_map = cv2.connectedComponents(binary)
                    for label_id in range(1, num_labels):
                        inst = (labels_map == label_id).astype(np.uint8)
                        if inst.sum() >= 50:
                            instance_masks.append(inst)
            elif label_txt_path.exists():
                img_h, img_w = image.shape[:2]
                with open(label_txt_path, "r") as lf:
                    lines = lf.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 7:  # class + at least 3 (x,y) pairs
                        continue
                    try:
                        coords = [float(x) for x in parts[1:]]
                        pts = np.array(coords).reshape(-1, 2)
                        pts[:, 0] *= img_w
                        pts[:, 1] *= img_h
                        inst = np.zeros((img_h, img_w), dtype=np.uint8)
                        cv2.fillPoly(inst, [pts.astype(np.int32)], 1)
                        if inst.sum() >= 50:
                            instance_masks.append(inst)
                    except Exception:
                        pass

            if not instance_masks:
                failed += 1
                continue

            all_logits = []
            predictor.set_image(image)

            # Capture encoder features
            features = predictor._features
            image_embed = features["image_embed"].cpu().half().numpy()
            feat0 = features["high_res_feats"][0].cpu().half().numpy()
            feat1 = features["high_res_feats"][1].cpu().half().numpy()

            for instance in instance_masks:
                box = mask_to_bbox(instance)
                if box is None:
                    continue

                if args.prompt_type == "box_centroid":
                    centroid = mask_to_centroid(instance)
                    point_labels = np.array([1], dtype=np.int32)
                    with torch.no_grad():
                        _, _, logits = predictor.predict(
                            point_coords=centroid,
                            point_labels=point_labels,
                            box=box,
                            multimask_output=False,
                        )
                else:
                    with torch.no_grad():
                        _, _, logits = predictor.predict(
                            box=box,
                            multimask_output=False,
                        )
                all_logits.append(logits[0].astype(np.float32))  # (256, 256)

            if all_logits:
                np.save(str(logit_file), np.stack(all_logits, axis=0))
                
                # Save encoder features in a shared features directory to save disk space
                features_dir = logits_dir.parent / "teacher_features"
                features_dir.mkdir(parents=True, exist_ok=True)
                feat_file = features_dir / f"{image_id}_features.npz"
                if not feat_file.exists():
                    np.savez_compressed(
                        str(feat_file),
                        image_embed=image_embed,
                        feat1=feat1
                    )
                generated += 1
            else:
                failed += 1

            total = generated + skipped + failed
            if total % 50 == 0 or total == len(images):
                print(f"  {total}/{len(images)} | "
                      f"generated={generated} skipped={skipped} failed={failed}")

        print(f"\n  Finished {d['name']}: generated={generated}, skipped={skipped}, failed={failed}")

    print("\n[Done] All active datasets processed successfully.")
    print("Next step:")
    print("  python scripts/run_experiments.py --exp full_kd")


if __name__ == "__main__":
    main()