#!/usr/bin/env python3
"""
Budgeted label distillation: build a Crack500 training set where only some crops keep their GT masks and the
rest get masks predicted from their boxes.

  labelled crops    (--labeled-list, e.g. train_photos25.txt)  -> GT label copied
  unlabelled crops  (everything else in train)                 -> boxes of the GT components (>= 50 px) only;
      --source sam      SAM 2.1 box prompts (zero-shot, or --weights from sam2_teacher_finetune.py finetune)
      --source student  a trained YOLO-seg student (--student best.pt), prediction kept inside the boxes

Val and test labels are always GT. The quality of the predicted masks against the hidden GT is saved in meta.json.
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
def sam_mask(predictor, img, boxes):
    set_image(predictor, img)
    return (decode(predictor, boxes)[1][:, 0] > 0).any(0).cpu().numpy()


def student_mask(model, img_path, boxes, shape, imgsz=512, conf=0.25):
    res = model.predict(str(img_path), imgsz=imgsz, conf=conf, retina_masks=True, verbose=False)[0]
    if res.masks is None:
        return np.zeros(shape, bool)
    return res.masks.data.bool().any(0).cpu().numpy() & boxes_mask(boxes, shape)


def build(crack500, yolo, out, source="sam", ckpt="checkpoints/sam2_hiera_large.pt", weights=None, student=None,
          labeled_list=None, box_jitter=0.0, max_images=None):
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
            if img_path.stem in labeled:
                shutil.copy2(yolo / "labels" / "train" / dst.name, dst)
                n_gt += 1
                continue
            if max_images and n_pseudo >= max_images:   # smoke tests: the remaining crops get no label file
                continue
            img, gt = read_pair(img_path, mask_path)
            _, boxes = instances(gt)
            pred = np.zeros(gt.shape, bool)
            if len(boxes):
                if box_jitter:
                    boxes = jitter(boxes, *gt.shape, frac=box_jitter)
                pred = sam_mask(predictor, img, boxes) if source == "sam" else student_mask(model, img_path, boxes, gt.shape)
            accumulate(acc, pred, gt)
            png = Path(tmp) / "m.png"
            cv2.imwrite(str(png), pred.astype(np.uint8) * 255)
            lines = binary_mask_to_yolo_instances(str(png), gt.shape[1], gt.shape[0])
            dst.write_text("".join(f"{l}\n" for l in lines))
            n_pseudo += 1
            n_empty += not lines
            if n_pseudo % 200 == 0:
                print(f"  {n_pseudo} pseudo-labelled", flush=True)
    (out / "dataset.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
                                      "nc: 1\nnames:\n  0: crack\n")
    meta = {"source": source, "weights": str(weights or student or "zero-shot"), "box_jitter": box_jitter,
            "gt_crops": n_gt, "pseudo_crops": n_pseudo, "pseudo_empty": n_empty,
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
    a = ap.parse_args()
    print(json.dumps(build(a.crack500, a.yolo, a.out, a.source, a.ckpt, a.weights, a.student, a.labeled_list,
                           a.box_jitter, a.max), indent=2))


if __name__ == "__main__":
    main()
