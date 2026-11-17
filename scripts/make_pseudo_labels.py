#!/usr/bin/env python3
"""
Budgeted label distillation: build a Crack500 training set where only some crops keep their GT masks and the
rest get masks predicted from their boxes.

  labelled crops    (--labeled-list, e.g. train_photos25.txt)  -> GT label copied
  unlabelled crops  (everything else in train)                 -> boxes of the GT components (>= 50 px) only;
      --source sam      SAM 2.1 box prompts (zero-shot, or --weights from sam2_teacher_finetune.py finetune)
      --source student  a trained YOLO-seg student (--student best.pt), prediction kept inside the boxes

Val and test labels are always GT. The quality of the predicted masks against the hidden GT is saved in meta.json.

--kd-out DIR (SAM only) also caches what the KD losses need, for every train crop that has label lines:
  DIR/teacher_logits_ft/crack500_<stem>_logits.npy     K x 256 x 256 float16, one teacher map per label line
  DIR/teacher_features/crack500_<stem>_features.npz    image_embed, feat1 (float16), only with --features
Output: <out>/{images/{train,val,test} (links to the source images), labels/{train,val,test}, dataset.yaml, meta.json}

Usage:
  python scripts/make_pseudo_labels.py --source sam --weights runs/teacher_ft/sam2_crack_decoder.pt \\
      --labeled-list data/datasets/crack500_yolo/train_photos25.txt --out data/datasets/crack500_pseudo
  python scripts/make_pseudo_labels.py --source sam --out data/datasets/crack500_pseudo        # 0% masks, zero-shot
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.convert_crack500 import binary_mask_to_yolo_instances
from scripts.sam2_teacher_finetune import (accumulate, decode, instances, jitter, load_predictor, new_acc, pairs,
                                           read_pair, read_stems, set_image, summarize)


def boxes_mask(boxes, shape):
    m = np.zeros(shape, bool)
    for x1, y1, x2, y2 in boxes.astype(int):
        m[y1:y2 + 1, x1:x2 + 1] = True
    return m


@torch.no_grad()
def sam_predict(predictor, img, boxes):
    """Union mask at full resolution and the per-box low-res logits (N x 256 x 256)."""
    set_image(predictor, img)
    low, full = decode(predictor, boxes)
    return (full[:, 0] > 0).any(0).cpu().numpy(), low[:, 0].float().cpu().numpy()


def sam_mask(predictor, img, boxes):
    return sam_predict(predictor, img, boxes)[0]


def teacher_maps(low, lines, size=256):
    """
    One teacher logit map per label line (K x size x size, float16), in label-file order, which is how the KD
    trainer pairs teacher maps with instances.

    low: N x size x size box-prompted logits spanning the whole image. Every map is their pixel-wise maximum,
    except that pixels inside another label's polygon are forced to background (-|logit|), so an instance is
    never taught the neighbouring crack as its own foreground.
    """
    union = low.max(0)
    regions = []
    for line in lines:
        pts = np.array(line.split()[1:], dtype=np.float32).reshape(-1, 2) * size
        r = np.zeros((size, size), np.uint8)
        cv2.fillPoly(r, [np.round(pts).astype(np.int32)], 1)
        regions.append(cv2.dilate(r, np.ones((3, 3), np.uint8)).astype(bool))
    maps = []
    for k, own in enumerate(regions):
        other = np.zeros_like(own)
        for j, r in enumerate(regions):
            if j != k:
                other |= r
        maps.append(np.where(other & ~own, -np.abs(union), union))
    return np.stack(maps).astype(np.float16)


def student_mask(model, img_path, boxes, shape, imgsz=512, conf=0.25):
    res = model.predict(str(img_path), imgsz=imgsz, conf=conf, retina_masks=True, verbose=False)[0]
    if res.masks is None:
        return np.zeros(shape, bool)
    return res.masks.data.bool().any(0).cpu().numpy() & boxes_mask(boxes, shape)


def build(crack500, yolo, out, source="sam", ckpt="checkpoints/sam2_hiera_large.pt", weights=None, student=None,
          labeled_list=None, box_jitter=0.0, max_images=None, kd_out=None, features=False):
    crack500, yolo, out = Path(crack500), Path(yolo).resolve(), Path(out)
    labeled = read_stems(labeled_list) if labeled_list else set()
    if out.exists():
        shutil.rmtree(out)
    for s in ("val", "test"):
        shutil.copytree(yolo / "labels" / s, out / "labels" / s, ignore=shutil.ignore_patterns("*.cache"))
    (out / "labels" / "train").mkdir(parents=True)
    # One link per image inside real folders: Ultralytics resolves a linked images/ folder back to the source
    # dataset and would then read the source (GT) labels instead of the ones written here
    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True)
        for img in (yolo / "images" / s).iterdir():
            os.symlink(img, out / "images" / s / img.name)

    assert not kd_out or source == "sam", "--kd-out caches SAM outputs: use --source sam"
    logits_dir = feats_dir = None
    if kd_out:
        logits_dir = Path(kd_out) / "teacher_logits_ft"
        feats_dir = Path(kd_out) / "teacher_features"
        for d in (logits_dir, feats_dir):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
    n_logits = 0
    if source == "sam":
        predictor = load_predictor(ckpt, weights)
    else:
        from ultralytics import YOLO
        model = YOLO(str(student))
    acc, n_gt, n_pseudo, n_empty = new_acc(), 0, 0, 0
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        todo = pairs(crack500 / "traincrop")
        for img_path, mask_path in todo:
            dst = out / "labels" / "train" / f"{img_path.stem}.txt"
            is_gt = img_path.stem in labeled
            if is_gt:
                shutil.copy2(yolo / "labels" / "train" / dst.name, dst)
                n_gt += 1
                if not kd_out:
                    continue
            elif max_images and n_pseudo >= max_images:   # smoke tests: the remaining crops get no label file
                continue
            img, gt = read_pair(img_path, mask_path)
            _, boxes = instances(gt)
            pred, low = np.zeros(gt.shape, bool), None
            if len(boxes):
                if box_jitter and not is_gt:
                    boxes = jitter(boxes, *gt.shape, frac=box_jitter)
                if source == "sam":
                    pred, low = sam_predict(predictor, img, boxes)
                else:
                    pred = student_mask(model, img_path, boxes, gt.shape)
            if is_gt:
                lines = dst.read_text().splitlines()
            else:
                accumulate(acc, pred, gt)
                png = Path(tmp) / "m.png"
                cv2.imwrite(str(png), pred.astype(np.uint8) * 255)
                lines = binary_mask_to_yolo_instances(str(png), gt.shape[1], gt.shape[0])
                dst.write_text("".join(f"{l}\n" for l in lines))
                n_pseudo += 1
                n_empty += not lines
                if n_pseudo % 200 == 0:
                    print(f"  {n_pseudo} pseudo-labelled", flush=True)
            lines = [l for l in lines if l.strip()]
            if kd_out and low is not None and lines:
                np.save(logits_dir / f"crack500_{img_path.stem}_logits.npy", teacher_maps(low, lines))
                n_logits += 1
                if features:
                    f = predictor._features
                    np.savez(feats_dir / f"crack500_{img_path.stem}_features.npz",
                             image_embed=f["image_embed"][0].half().cpu().numpy(),
                             feat1=f["high_res_feats"][1][0].half().cpu().numpy())
    (out / "dataset.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
                                      "nc: 1\nnames:\n  0: crack\n")
    meta = {"source": source, "weights": str(weights or student or "zero-shot"), "box_jitter": box_jitter,
            "gt_crops": n_gt, "pseudo_crops": n_pseudo, "pseudo_empty": n_empty, "kd_logit_files": n_logits,
            "pseudo_quality_vs_hidden_gt": summarize(acc) if n_pseudo else None}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=["sam", "student"], default="sam")
    ap.add_argument("--crack500", default="data/datasets/crack500")
    ap.add_argument("--yolo", default="data/datasets/crack500_yolo")
    ap.add_argument("--out", default="data/datasets/crack500_pseudo")
    ap.add_argument("--ckpt", default="checkpoints/sam2_hiera_large.pt")
    ap.add_argument("--weights", default=None, help="fine-tuned SAM decoder; omit for zero-shot")
    ap.add_argument("--student", default=None, help="YOLO-seg best.pt (for --source student)")
    ap.add_argument("--labeled-list", default=None, help="crops that keep GT masks; omit for 0%% masks")
    ap.add_argument("--box-jitter", type=float, default=0.0)
    ap.add_argument("--max", type=int, default=None, help="limit pseudo-labelled crops (smoke tests)")
    ap.add_argument("--kd-out", default=None, help="also cache teacher logits (and features) for the KD losses here")
    ap.add_argument("--features", action="store_true", help="with --kd-out: cache SAM encoder features too")
    a = ap.parse_args()
    print(json.dumps(build(a.crack500, a.yolo, a.out, a.source, a.ckpt, a.weights, a.student, a.labeled_list,
                           a.box_jitter, a.max, a.kd_out, a.features), indent=2))


if __name__ == "__main__":
    main()
