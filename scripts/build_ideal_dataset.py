#!/usr/bin/env python3
"""
Crack500 + DeepCrack training set at a matched crack scale, split by scene so nothing leaks.

- Crack500: the canonical crops and labels from crack500_yolo, unchanged (1896 / 348 / 1124).
- DeepCrack (544x384): cracks are ~3x thinner in pixels than in Crack500 crops (median width 4.4 px vs
  ~14 px after the x0.8 training resize), so images are upscaled x2 and cut into six 512x512 tiles
  (3 x 2, overlapping). Tiles without cracks are kept: Crack500 crops all contain a crack, so these are
  the only crack-free training examples.
- DeepCrack splits are by scene ("11289-3" -> scene "11289"): train scenes 80/20 into train/val with seed 0;
  test images whose scene also appears in train (11142, 11289, IMG_6536) are dropped from test.

Output: data/datasets/crack_ideal_yolo/{images,labels}/{train,val,test}, dataset.yaml, meta.json.
Usage: python scripts/build_ideal_dataset.py [--out data/datasets/crack_ideal_yolo]
"""

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.convert_crack500 import binary_mask_to_yolo_instances

SCALE, TILE = 2.0, 512
XS, YS = (0, 288, 576), (0, 256)   # 1088x768 canvas -> 3 x 2 overlapping tiles
SPLITS = ("train", "val", "test")


def scene(stem):
    return stem.split("-")[0]


def deepcrack_pairs(root, split):
    labs = {p.stem: p for p in (root / f"{split}_lab").iterdir()}
    return [(p, labs[p.stem]) for p in sorted((root / f"{split}_img").iterdir()) if p.stem in labs]


def deepcrack_splits(root, val_frac=0.2, seed=0):
    train = deepcrack_pairs(root, "train")
    scenes = sorted({scene(i.stem) for i, _ in train})
    random.Random(seed).shuffle(scenes)
    val_scenes = set(scenes[:round(val_frac * len(scenes))])
    test = [(i, l) for i, l in deepcrack_pairs(root, "test") if scene(i.stem) not in set(scenes)]
    return {"train": [p for p in train if scene(p[0].stem) not in val_scenes],
            "val": [p for p in train if scene(p[0].stem) in val_scenes], "test": test}


def write_tiles(img_path, lab_path, out, split, tmp):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    lab = cv2.imread(str(lab_path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    assert img.shape[:2] == lab.shape, img_path
    h, w = img.shape[:2]
    big = cv2.resize(img, (round(w * SCALE), round(h * SCALE)), interpolation=cv2.INTER_CUBIC)
    mask = (cv2.resize(lab, big.shape[1::-1], interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8) * 255
    n_empty = 0
    for y in YS:
        for x in XS:
            stem = f"deepcrack_{img_path.stem}_{x}_{y}"
            tile, mtile = big[y:y + TILE, x:x + TILE], mask[y:y + TILE, x:x + TILE]
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
            mpath = tmp / f"{stem}.png"
            cv2.imwrite(str(mpath), mtile)
            lines = binary_mask_to_yolo_instances(str(mpath), TILE, TILE)
            (out / "labels" / split / f"{stem}.txt").write_text("".join(f"{l}\n" for l in lines))
            n_empty += not lines
    return n_empty


def build(crack500_yolo, deepcrack, out):
    crack500_yolo, deepcrack, out = Path(crack500_yolo), Path(deepcrack), Path(out)
    if out.exists():
        shutil.rmtree(out)
    meta = {"crack500": {}, "deepcrack_images": {}, "deepcrack_tiles": {}, "deepcrack_empty_tiles": {}}
    for s in SPLITS:
        for kind in ("images", "labels"):
            (out / kind / s).mkdir(parents=True)
        imgs = sorted((crack500_yolo / "images" / s).glob("*.jpg"))
        for p in imgs:
            shutil.copy2(p, out / "images" / s / f"crack500_{p.name}")
            shutil.copy2(crack500_yolo / "labels" / s / f"{p.stem}.txt", out / "labels" / s / f"crack500_{p.stem}.txt")
        meta["crack500"][s] = len(imgs)
    with tempfile.TemporaryDirectory() as tmp:
        for s, pairs in deepcrack_splits(deepcrack).items():
            meta["deepcrack_images"][s] = len(pairs)
            meta["deepcrack_empty_tiles"][s] = sum(write_tiles(i, l, out, s, Path(tmp)) for i, l in pairs)
            meta["deepcrack_tiles"][s] = len(pairs) * len(XS) * len(YS)
    (out / "dataset.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
                                      "nc: 1\nnames:\n  0: crack\n")
    meta.update(scale=SCALE, tile=TILE, rule="DeepCrack x2, 6 tiles/image, split by scene, seed 0")
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--crack500-yolo", default="data/datasets/crack500_yolo")
    ap.add_argument("--deepcrack", default="data/datasets/deepcrack")
    ap.add_argument("--out", default="data/datasets/crack_ideal_yolo")
    a = ap.parse_args()
    print(json.dumps(build(a.crack500_yolo, a.deepcrack, a.out), indent=2))


if __name__ == "__main__":
    main()
