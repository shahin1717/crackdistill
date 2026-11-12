#!/usr/bin/env python3
"""
Teacher bake-off: zero-shot SAM models of different sizes on Crack500 val crops, all with the same box prompts
(one box per GT component >= 50 px, as used for the KD logits).

  SAM 2.1: tiny, small, base_plus, large        SAM 1: vit_b, vit_l, vit_h   (needs the segment-anything package)

Per model: pixel precision / recall / F1, mean IoU, thin-crack recall, parameters, ms per image.
Checkpoints are downloaded to checkpoints/ when missing. A model that fails is reported and skipped.

Usage:
  python scripts/sam_bakeoff.py --crack500 data/datasets/crack500 --out runs/teacher_bakeoff
  python scripts/sam_bakeoff.py --models sam2.1_tiny sam2.1_large --max 20
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.sam2_teacher_finetune import (SAM2_MODELS, accumulate, decode, instances, load_predictor, new_acc, pairs,
                                           read_pair, set_image, summarize)

SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
SAM1_URL = "https://dl.fbaipublicfiles.com/segment_anything/"
# Hugging Face copies, used when the Meta host fails (its TLS certificate was expired on 2026-11-12)
SAM2_HF = "https://huggingface.co/facebook/{repo}/resolve/main/"
SAM1_HF = "https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/"
SAM1_MODELS = {"sam1_vit_b": ("vit_b", "sam_vit_b_01ec64.pth"), "sam1_vit_l": ("vit_l", "sam_vit_l_0b3195.pth"),
               "sam1_vit_h": ("vit_h", "sam_vit_h_4b8939.pth")}
MODELS = [*SAM2_MODELS, *SAM1_MODELS]


def fetch(urls, dst, min_bytes=3e7):
    """Download the first working url to dst unless a complete file is already there; a failed download must not
    leave an empty file."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        if dst.exists() and dst.stat().st_size >= min_bytes:
            break
        dst.unlink(missing_ok=True)
        print(f"  downloading {url}", flush=True)
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            print(f"  download failed ({type(e).__name__}: {e})", flush=True)
    assert dst.exists() and dst.stat().st_size >= min_bytes, f"could not download {dst.name}"
    return dst


def sam2_mask_fn(name, ckpt_dir, device):
    cfg, fname = SAM2_MODELS[name]
    # the large checkpoint is stored under its historical name by the other scripts
    old = Path(ckpt_dir) / "sam2_hiera_large.pt"
    ckpt = old if name == "sam2.1_large" and old.exists() and old.stat().st_size > 8e8 else fetch(
        [SAM2_URL + fname, SAM2_HF.format(repo=fname[:-3].replace("_", "-")) + fname], Path(ckpt_dir) / fname)
    predictor = load_predictor(ckpt, device=device, cfg=cfg)

    def mask_fn(img, boxes):
        set_image(predictor, img)
        return (decode(predictor, boxes)[1][:, 0] > 0).any(0).cpu().numpy()
    return mask_fn, predictor.model


def sam1_mask_fn(name, ckpt_dir, device):
    from segment_anything import SamPredictor, sam_model_registry
    arch, fname = SAM1_MODELS[name]
    model = sam_model_registry[arch](checkpoint=str(fetch([SAM1_URL + fname, SAM1_HF + fname], Path(ckpt_dir) / fname))).to(device).eval()
    predictor = SamPredictor(model)

    def mask_fn(img, boxes):
        predictor.set_image(img)
        b = predictor.transform.apply_boxes_torch(torch.as_tensor(boxes, device=device), img.shape[:2])
        masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=b, multimask_output=False)
        return masks[:, 0].any(0).cpu().numpy()
    return mask_fn, model


@torch.no_grad()
def evaluate_fn(mask_fn, split_dir, max_images=None):
    acc, seconds = new_acc(), 0.0
    for img_path, mask_path in pairs(split_dir, max_images):
        img, gt = read_pair(img_path, mask_path)
        _, boxes = instances(gt)
        pred = np.zeros(gt.shape, bool)
        if len(boxes):
            t0 = time.time()
            pred = mask_fn(img, boxes)
            seconds += time.time() - t0
        accumulate(acc, pred, gt)
    return {**summarize(acc), "ms_per_image": 1000 * seconds / max(len(acc["ious"]), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--crack500", default="data/datasets/crack500")
    ap.add_argument("--split", default="valcrop")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out", default="runs/teacher_bakeoff")
    ap.add_argument("--max", type=int, default=None, help="limit images (smoke tests)")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in a.models:
        print(f"[{name}]", flush=True)
        try:
            mask_fn, model = (sam2_mask_fn if name in SAM2_MODELS else sam1_mask_fn)(name, a.ckpt_dir, device)
            results[name] = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6,
                             **evaluate_fn(mask_fn, Path(a.crack500) / a.split, a.max)}
            print(f"  {json.dumps(results[name])}", flush=True)
            del mask_fn, model
            torch.cuda.empty_cache()
        except Exception as e:   # one broken download or install must not lose the other models
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  FAILED {results[name]['error']}", flush=True)
        (out / f"bakeoff_{a.split}.json").write_text(json.dumps(results, indent=2) + "\n")
    ok = {k: v for k, v in results.items() if "error" not in v}
    print(f"\n{'model':<18}{'params M':>9}{'ms/img':>8}{'prec':>7}{'recall':>7}{'F1':>7}{'IoU':>7}{'thin R':>7}")
    for k, v in sorted(ok.items(), key=lambda kv: -kv[1]["f1"]):
        print(f"{k:<18}{v['params_M']:>9.1f}{v['ms_per_image']:>8.0f}{v['precision']:>7.3f}{v['recall']:>7.3f}"
              f"{v['f1']:>7.3f}{v['mean_iou']:>7.3f}{v['thin_recall']:>7.3f}")
    if ok:
        print(f"\n[Best zero-shot teacher by pixel F1] {max(ok, key=lambda k: ok[k]['f1'])}")


if __name__ == "__main__":
    main()
