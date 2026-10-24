#!/usr/bin/env python3
"""
Training subset for the data-efficiency curve: a fixed fraction of whole parent photos (all crops of a photo stay
together, so no photo is split across "seen" and "unseen"). Subsets are nested — the 25% photos are the first
quarter of the same shuffled order the 50% subset uses — and identical for every arm.

Writes <dataset>/train_photos<pct>.txt (absolute image paths) and <dataset>/dataset_photos<pct>.yaml
(same val/test as dataset.yaml).

Usage:
  python scripts/make_train_subset.py --dataset data/datasets/crack500_yolo --fraction 0.25
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parent_photo(stem):
    """'20160222_080933_361_1281' -> '20160222_080933' (crop stems end in _<x>_<y>)."""
    return stem.rsplit("_", 2)[0]


def select_photos(stems, fraction, seed=0):
    photos = sorted({parent_photo(s) for s in stems})
    random.Random(seed).shuffle(photos)
    return set(photos[:round(fraction * len(photos))])


def write_subset(dataset_dir, fraction, seed=0):
    dataset_dir = Path(dataset_dir).resolve()
    imgs = sorted(p for p in (dataset_dir / "images" / "train").iterdir() if p.suffix.lower() in IMG_EXT)
    keep = select_photos([p.stem for p in imgs], fraction, seed)
    chosen = [p for p in imgs if parent_photo(p.stem) in keep]
    tag = f"photos{round(fraction * 100)}"
    (dataset_dir / f"train_{tag}.txt").write_text("".join(f"{p}\n" for p in chosen))
    yaml_path = dataset_dir / f"dataset_{tag}.yaml"
    yaml_path.write_text(f"path: {dataset_dir}\ntrain: train_{tag}.txt\nval: images/val\ntest: images/test\n\n"
                         "nc: 1\nnames:\n  0: crack\n")
    sha = hashlib.sha256("\n".join(p.stem for p in chosen).encode()).hexdigest()
    return {"fraction": fraction, "seed": seed, "photos": len(keep), "crops": len(chosen),
            "sha256": sha, "yaml": str(yaml_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="data/datasets/crack500_yolo")
    ap.add_argument("--fraction", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(write_subset(a.dataset, a.fraction, a.seed), indent=2))


if __name__ == "__main__":
    main()
