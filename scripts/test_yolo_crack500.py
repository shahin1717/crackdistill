#!/usr/bin/env python3
"""
Quick sanity test — verifies YOLO11n-seg trains for 1 epoch on crack500_yolo.
Run this after convert_crack500.py to confirm everything works end-to-end.

Usage:
  python scripts/test_yolo_crack500.py --data ~/distill/data/datasets/crack500_yolo/dataset.yaml
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset.yaml (output of convert_crack500.py)"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz",  type=int, default=512)
    parser.add_argument("--batch",  type=int, default=4)
    args = parser.parse_args()

    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.exists():
        print(f"ERROR: dataset.yaml not found at {data_yaml}")
        print("Run convert_crack500.py first.")
        return

    print(f"[Test] Loading YOLO11n-seg...")
    from ultralytics import YOLO
    model = YOLO("yolo11n-seg.pt")

    print(f"[Test] Training for {args.epochs} epoch(s) on {data_yaml}")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cuda",
        amp=True,
        verbose=True,
        project="runs/test",
        name="crack500_smoke_test",
    )

    print("\n[Test] ✓ Smoke test passed!")
    print(f"  Results saved to: runs/test/crack500_smoke_test/")
    print(f"\nNow run full training:")
    print(f"  python scripts/run_experiments.py --exp baseline_finetune")
    print(f"  python scripts/run_experiments.py --exp full_kd")


if __name__ == "__main__":
    main()
