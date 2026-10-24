#!/usr/bin/env python3
"""
Scale-matched evaluation of trained checkpoints (inference only; light enough for the laptop GPU).
  test crops : COCO mask mAP (Ultralytics val, split=test) + pixel F1/IoU at a threshold tuned on the val crops
  full photos: x0.8 (the training scale), 512 tiles, 50% overlap, max-combined; pixel F1/IoU at a threshold
               tuned on the 50 val photos; also split into crop-source photos and the 52 IMG_* other-camera photos
Score map per image = max over instances of (instance mask x confidence), so the threshold acts on confidence.
Usage: python scripts/evaluate_scale_matched.py results/test/out.json name=ckpt.pt [name=ckpt.pt ...]
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

C5 = "data/datasets/crack500"
YAML = "data/datasets/crack500_yolo/dataset.yaml"
SCALE, TILE, STRIDE, MAX_DET = 0.8, 512, 256, 50
THRS = [round(0.05 * i, 2) for i in range(1, 13)]


def imread(p, flag=cv2.IMREAD_COLOR):
    return cv2.imread(str(p), flag | cv2.IMREAD_IGNORE_ORIENTATION)


def score_map(r, shape):
    if r.masks is None or len(r.boxes) == 0:
        return np.zeros(shape, np.uint8)
    m = (r.masks.data.float() * r.boxes.conf.float()[:, None, None]).max(0).values
    m = (m * 255).byte().cpu().numpy()
    return m if m.shape == shape else cv2.resize(m, shape[::-1], interpolation=cv2.INTER_LINEAR)


def predict(model, imgs):
    return model.predict(imgs, imgsz=TILE, conf=THRS[0], retina_masks=True, half=True, max_det=MAX_DET, verbose=False)


def tiled(model, img):
    h0, w0 = img.shape[:2]
    small = cv2.resize(img, (round(w0 * SCALE), round(h0 * SCALE)), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    H = max(TILE, int(np.ceil((h - TILE) / STRIDE)) * STRIDE + TILE)
    W = max(TILE, int(np.ceil((w - TILE) / STRIDE)) * STRIDE + TILE)
    canvas = np.full((H, W, 3), 114, np.uint8)
    canvas[:h, :w] = small
    pos = [(y, x) for y in range(0, H - TILE + 1, STRIDE) for x in range(0, W - TILE + 1, STRIDE)]
    out = np.zeros((H, W), np.uint8)
    for i in range(0, len(pos), 16):
        res = predict(model, [canvas[y:y + TILE, x:x + TILE] for y, x in pos[i:i + 16]])
        for (y, x), r in zip(pos[i:i + 16], res):
            np.maximum(out[y:y + TILE, x:x + TILE], score_map(r, (TILE, TILE)), out=out[y:y + TILE, x:x + TILE])
    return cv2.resize(out[:h, :w], (w0, h0), interpolation=cv2.INTER_LINEAR)


def add(counts, smap, gt):
    for t in THRS:
        p = smap >= round(t * 255)
        counts[t] += ((p & gt).sum(), (p & ~gt).sum(), (~p & gt).sum())


def scores(c):
    tp, fp, fn = (int(v) for v in c)
    return {"F1": 2 * tp / max(1, 2 * tp + fp + fn), "IoU": tp / max(1, tp + fp + fn),
            "P": tp / max(1, tp + fp), "R": tp / max(1, tp + fn)}


def new_counts():
    return {t: np.zeros(3, np.int64) for t in THRS}


def crops(model, folder):
    counts = new_counts()
    imgs = sorted(Path(folder).glob("*.jpg"))
    for i in range(0, len(imgs), 16):
        batch = imgs[i:i + 16]
        arrs = [imread(p) for p in batch]
        for p, a, r in zip(batch, arrs, predict(model, arrs)):
            gt = imread(p.with_suffix(".png"), cv2.IMREAD_GRAYSCALE) > 127
            assert gt.shape == a.shape[:2], p
            add(counts, score_map(r, gt.shape), gt)
    return counts, len(imgs)


def photos(model, folder):
    groups = {"all": new_counts(), "crop_source": new_counts(), "IMG_other_cameras": new_counts()}
    n = 0
    for mf in sorted(glob.glob(f"{folder}/*_mask.png")):
        stem = mf[:-len("_mask.png")]
        img = imread(next(p for p in (stem + ".jpg", stem + ".JPG") if Path(p).exists()))
        gt = imread(mf, cv2.IMREAD_GRAYSCALE) > 127
        assert gt.shape == img.shape[:2], stem
        smap = tiled(model, img)
        add(groups["all"], smap, gt)
        add(groups["IMG_other_cameras" if Path(stem).name.startswith("IMG_") else "crop_source"], smap, gt)
        n += 1
    return groups, n


def tuned(val_counts, test_counts):
    t = max(THRS, key=lambda t: scores(val_counts[t])["F1"])
    return {"thr": t, "val_F1": scores(val_counts[t])["F1"], **{f"test_{k}": v for k, v in scores(test_counts[t]).items()}}


out_path, results = sys.argv[1], {}
for spec in sys.argv[2:]:
    name, ckpt = spec.split("=", 1)
    model = YOLO(ckpt)
    v = model.val(data=YAML, split="test", imgsz=TILE, batch=8, workers=2, plots=False, verbose=False)
    res = {"ckpt": ckpt, "test_crops_mask_mAP50": float(v.seg.map50), "test_crops_mask_mAP50_95": float(v.seg.map),
           "test_crops_box_mAP50": float(v.box.map50)}
    model = YOLO(ckpt)
    vc, nv = crops(model, f"{C5}/valcrop")
    tc, nt = crops(model, f"{C5}/testcrop")
    res["crops"] = {"n_val": nv, "n_test": nt, **tuned(vc, tc)}
    vp, nvp = photos(model, f"{C5}/valdata")
    tp_, ntp = photos(model, f"{C5}/testdata")
    res["photos"] = {"n_val": nvp, "n_test": ntp,
                     **{g: tuned(vp["all"], tp_[g]) for g in tp_}}
    results[name] = res
    print(name, json.dumps(res, indent=1), flush=True)
    Path(out_path).write_text(json.dumps(results, indent=2))

print("\n{:22s} {:>8s} {:>8s} | {:>7s} {:>7s} {:>5s} | {:>7s} {:>7s} {:>5s} {:>6s} {:>6s}".format(
    "model", "mAP50", "mAP50-95", "cropF1", "cropIoU", "thr", "photoF1", "phIoU", "thr", "P", "R"))
for name, r in results.items():
    c, p = r["crops"], r["photos"]["all"]
    print(f"{name:22s} {r['test_crops_mask_mAP50']:8.3f} {r['test_crops_mask_mAP50_95']:8.3f} | "
          f"{c['test_F1']:7.3f} {c['test_IoU']:7.3f} {c['thr']:5.2f} | {p['test_F1']:7.3f} {p['test_IoU']:7.3f} "
          f"{p['thr']:5.2f} {p['test_P']:6.3f} {p['test_R']:6.3f}")
