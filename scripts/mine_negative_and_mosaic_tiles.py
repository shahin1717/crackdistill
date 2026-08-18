#!/usr/bin/env python3
"""
Mines two new training assets from the existing Crack500 traincrop grid,
without any new data collection or labeling:

1. Negative (background-only) tiles: crack500_yolo/labels/train has 0 empty
   label files today (verified) — every surviving traincrop tile contains a
   crack, because Crack500's authors discarded background-only grid cells
   before shipping the dataset. This script identifies which grid cells
   *survive* per source photo (from the {stem}_{x}_{y}.jpg filenames, a
   640x360 stride grid) and, for photos where source pixels for unused
   cells are recoverable from valdata/testdata (which DO ship full photos),
   extracts those unused cells as genuine background-only training crops.

2. Mosaic composites: wherever 2+ surviving traincrop tiles are grid-adjacent
   for the same source photo, stitches them (image + mask) into one larger
   composite for native-resolution SAM2 teacher-logit generation.

Run (from ~/distill):
  python scripts/mine_negative_and_mosaic_tiles.py
"""

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
TRAINCROP_DIR = ROOT / "data/datasets/crack500/traincrop"
VALDATA_DIRS = [ROOT / "data/datasets/crack500/valdata", ROOT / "data/datasets/crack500/testdata"]

STEM_RE = re.compile(r"^(.+?)_(\d+)_(\d+)\.jpg$")


def parse_traincrop_grid():
    """Group traincrop tiles by source stem -> list of (x, y) offsets present."""
    stems = defaultdict(list)
    for f in TRAINCROP_DIR.glob("*.jpg"):
        m = STEM_RE.match(f.name)
        if not m:
            continue
        stem, x, y = m.group(1), int(m.group(2)), int(m.group(3))
        stems[stem].append((x, y))
    return stems


def tile_shape_for_stem(stem, coords):
    """
    Tile pixel dimensions are NOT constant across the dataset — Crack500 mixes
    landscape and portrait source photos, and the crop grid follows each photo's
    own orientation. Determine (tile_w, tile_h) per-stem from an actual loaded tile
    rather than assuming a single global 640x360.
    """
    x, y = coords[0]
    p = TRAINCROP_DIR / f"{stem}_{x}_{y}.jpg"
    img = cv2.imread(str(p), cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def find_source_photo(stem):
    """Locate a full-resolution photo for `stem` in valdata/testdata (same date-prefixed camera series is NOT assumed; only an exact filename match counts)."""
    for d in VALDATA_DIRS:
        cand = d / f"{stem}.jpg"
        if cand.exists():
            return cand
    return None


def full_grid_positions(w, h, tile_w, tile_h):
    """All (x, y) top-left offsets a tile_w x tile_h grid would occupy over a w x h photo."""
    xs = list(range(1, max(2, w - tile_w + 2), tile_w))
    ys = list(range(1, max(2, h - tile_h + 2), tile_h))
    return [(x, y) for y in ys for x in xs if x + tile_w - 1 <= w and y + tile_h - 1 <= h]


def mine_negative_tiles(stems, out_img_dir, out_lbl_dir, max_per_stem=2):
    """
    For stems whose exact-filename full photo exists in valdata/testdata (i.e. a photo
    that is ALSO a val/test image — safe here only because we read PIXELS from an
    unrelated grid cell never used as a val/test crop, not because we reuse val/test
    *evaluation regions*), extract unused grid cells as background-only negative crops.
    NOTE: by construction this only fires for stems with an exact filename collision, which
    in this dataset is empty (train and val/test stems are disjoint camera captures) —
    documented here so the gap is explicit rather than silently producing zero output.
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_stems_with_source = 0
    for stem, coords in stems.items():
        photo_path = find_source_photo(stem)
        if photo_path is None:
            continue
        tile_shape = tile_shape_for_stem(stem, coords)
        if tile_shape is None:
            continue
        tile_w, tile_h = tile_shape
        n_stems_with_source += 1
        img = cv2.imread(str(photo_path), cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        used = set(coords)
        unused = [c for c in full_grid_positions(w, h, tile_w, tile_h) if c not in used]
        for i, (x, y) in enumerate(unused[:max_per_stem]):
            tile = img[y - 1:y - 1 + tile_h, x - 1:x - 1 + tile_w]
            if tile.shape[:2] != (tile_h, tile_w):
                continue
            out_name = f"{stem}_neg_{x}_{y}"
            cv2.imwrite(str(out_img_dir / f"{out_name}.jpg"), tile)
            (out_lbl_dir / f"{out_name}.txt").write_text("")  # empty label = background-only
            n_written += 1
    return n_written, n_stems_with_source


def stitch_mosaics(stems, out_img_dir, out_mask_dir, min_tiles=2):
    """Stitch grid-adjacent surviving tiles per stem into larger composites (image + mask)."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    n_composites = 0
    for stem, coords in stems.items():
        if len(coords) < min_tiles:
            continue
        tile_shape = tile_shape_for_stem(stem, coords)
        if tile_shape is None:
            continue
        tile_w, tile_h = tile_shape
        xs = sorted(set(c[0] for c in coords))
        ys = sorted(set(c[1] for c in coords))
        # Find the largest fully-covered rectangular block of grid cells present.
        best = None
        for y0 in ys:
            for y1 in ys:
                if y1 < y0:
                    continue
                for x0 in xs:
                    for x1 in xs:
                        if x1 < x0:
                            continue
                        needed = [(x, y) for y in ys if y0 <= y <= y1 for x in xs if x0 <= x <= x1]
                        if all(c in coords for c in needed) and len(needed) >= min_tiles:
                            area = (len(set(x for x, _ in needed))) * (len(set(y for _, y in needed)))
                            if best is None or area > best[0]:
                                best = (area, x0, x1, y0, y1)
        if best is None:
            continue
        _, x0, x1, y0, y1 = best
        block_xs = [x for x in xs if x0 <= x <= x1]
        block_ys = [y for y in ys if y0 <= y <= y1]
        canvas_w = len(block_xs) * tile_w
        canvas_h = len(block_ys) * tile_h
        canvas_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        wrote_any = False
        for j, y in enumerate(block_ys):
            for i, x in enumerate(block_xs):
                tile_img_p = TRAINCROP_DIR / f"{stem}_{x}_{y}.jpg"
                tile_mask_p = TRAINCROP_DIR / f"{stem}_{x}_{y}.png"
                tile_img = cv2.imread(str(tile_img_p), cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
                tile_mask = cv2.imread(str(tile_mask_p), cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_GRAYSCALE)
                if tile_img is None or tile_img.shape[:2] != (tile_h, tile_w):
                    continue
                canvas_img[j * tile_h:(j + 1) * tile_h, i * tile_w:(i + 1) * tile_w] = tile_img
                wrote_any = True
                if tile_mask is not None and tile_mask.shape[:2] == (tile_h, tile_w):
                    canvas_mask[j * tile_h:(j + 1) * tile_h, i * tile_w:(i + 1) * tile_w] = tile_mask
        if not wrote_any:
            continue
        cv2.imwrite(str(out_img_dir / f"{stem}_mosaic.jpg"), canvas_img)
        cv2.imwrite(str(out_mask_dir / f"{stem}_mosaic.png"), canvas_mask)
        n_composites += 1
    return n_composites


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(ROOT / "data/datasets/crack500_ood_mined"))
    parser.add_argument("--min-mosaic-tiles", type=int, default=2)
    parser.add_argument("--max-neg-per-stem", type=int, default=2)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    stems = parse_traincrop_grid()
    print(f"[Grid] Parsed {len(stems)} unique source stems from {TRAINCROP_DIR}")

    n_mosaics = stitch_mosaics(
        stems,
        out_root / "mosaic_images",
        out_root / "mosaic_masks",
        min_tiles=args.min_mosaic_tiles,
    )
    print(f"[Mosaics] Stitched {n_mosaics} multi-tile composites -> {out_root / 'mosaic_images'}")

    n_neg, n_with_source = mine_negative_tiles(
        stems,
        out_root / "negative_images",
        out_root / "negative_labels",
        max_per_stem=args.max_neg_per_stem,
    )
    print(f"[Negatives] {n_with_source}/{len(stems)} stems had a matching full-res source photo in valdata/testdata")
    print(f"[Negatives] Wrote {n_neg} background-only crops -> {out_root / 'negative_images'}")
    if n_neg == 0:
        print("[Negatives] 0 written: train-split stems have no filename match in valdata/testdata "
              "(expected — Crack500's train and val/test photos are disjoint camera captures). "
              "Negative tiles need an external/new photo source; see OODimprovements/README.md.")


if __name__ == "__main__":
    main()
