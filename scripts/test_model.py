#!/usr/bin/env python3
"""
Test the KD-trained student model.
Shows predictions on test images with metrics.

Usage:
  # Test on crack500 test set
  python scripts/test_model.py

  # Test on a single image
  python scripts/test_model.py --image path/to/image.jpg

  # Test on a folder
  python scripts/test_model.py --folder path/to/images/
"""

import argparse
import random
import time
import cv2
import numpy as np
from pathlib import Path

MODEL_PATH  = Path.home() / "distill/runs/kd_demo/kd_student/weights/best.pt"
TEST_DIR    = Path.home() / "distill/data/datasets/crack500/testcrop"
OUTPUT_DIR  = Path.home() / "distill/runs/kd_demo/test_results"


def test(source, model_path=MODEL_PATH, max_images=20):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect images
    if isinstance(source, str) and Path(source).is_file():
        images = [Path(source)]
    elif isinstance(source, str) and Path(source).is_dir():
        all_images = [
            f for f in Path(source).iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            and ':Zone' not in f.name
            and not f.name.lower().endswith('_mask.png')
        ]
        images = random.sample(all_images, min(max_images, len(all_images)))
    else:
        images = source[:max_images]

    print(f"\nTesting on {len(images)} images...")
    print(f"Model: {model_path}\n")

    times = []
    detections = []

    for img_path in images:
        t0 = time.perf_counter()
        results = model.predict(
            str(img_path),
            imgsz=512,
            conf=0.25,
            device="cuda",
            verbose=False,
        )
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)

        r = results[0]
        n_instances = len(r.boxes) if r.boxes is not None else 0
        detections.append(n_instances)

        # Save annotated image
        annotated = r.plot()
        out_path  = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), annotated)

        print(f"  {img_path.name:40s}  {n_instances} cracks  {ms:.1f}ms")

    # Summary
    avg_ms  = np.mean(times)
    avg_fps = 1000 / avg_ms
    print(f"\n{'─'*55}")
    print(f"  Images tested:    {len(images)}")
    print(f"  Avg speed:        {avg_ms:.1f} ms  →  {avg_fps:.0f} FPS")
    print(f"  Avg detections:   {np.mean(detections):.1f} instances/image")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'─'*55}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--model",  type=str, default=str(MODEL_PATH))
    parser.add_argument("--n",      type=int, default=20, help="Max images to test")
    parser.add_argument("--val",    action="store_true", help="Run full validation (eval metrics)")
    parser.add_argument("--data",   type=str, default="data/datasets/crack500_uncropped_yolo/dataset.yaml", help="Path to dataset.yaml")
    args = parser.parse_args()

    if args.val:
        from ultralytics import YOLO
        print(f"\nRunning validation evaluation on: {args.data}")
        print(f"Model: {args.model}\n")
        model = YOLO(args.model)
        metrics = model.val(
            data=args.data,
            imgsz=512,
            device="cuda",
            verbose=True,
        )
        print("\n[Validation Metrics]")
        print(f"  mAP50-box:    {metrics.box.map50:.4f}")
        print(f"  mAP50-seg:    {metrics.seg.map50:.4f}")
        print(f"  mAP50-95-seg: {metrics.seg.map:.4f}")
    else:
        source = args.image or args.folder or str(TEST_DIR)
        test(source, model_path=args.model, max_images=args.n)


if __name__ == "__main__":
    main()