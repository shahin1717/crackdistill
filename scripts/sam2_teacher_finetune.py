#!/usr/bin/env python3
"""
Check and fine-tune the SAM 2.1 teacher on cracks, with the same box prompts used to cache the KD logits.

  eval      box-prompted teacher vs GT on a Crack500 crop split: pixel P/R/F1, mean IoU, thin-crack recall
  finetune  train the mask decoder + prompt encoder on traincrop (image encoder frozen), keep the best val F1
  logits    re-cache teacher logits: crack500_<stem>_logits.npy, M x 256 x 256, same instances and order as
            scripts/generate_teacher_logits.py (connected components >= 50 px, one box each)

Instances are the GT connected components (>= 50 px); the prediction for an image is the union of its instance masks.

Usage:
  python scripts/sam2_teacher_finetune.py eval --crack500 data/datasets/crack500 --split valcrop
  python scripts/sam2_teacher_finetune.py finetune --crack500 data/datasets/crack500 --out runs/teacher_ft
  python scripts/sam2_teacher_finetune.py finetune --train-list data/datasets/crack500_yolo/train_photos25.txt
  python scripts/sam2_teacher_finetune.py eval --split valcrop --weights runs/teacher_ft/sam2_crack_decoder.pt
  python scripts/sam2_teacher_finetune.py logits --weights runs/teacher_ft/sam2_crack_decoder.pt --out data/teacher_logits_ft
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
MIN_AREA = 50
THIN_PX = 4   # a GT pixel is "thin" if the local crack width around it is <= 4 px


def load_predictor(ckpt, weights=None, device=None):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SAM2ImagePredictor(build_sam2(SAM2_CFG, str(ckpt), device=device))
    if weights:
        state = torch.load(weights, map_location=device)
        predictor.model.sam_mask_decoder.load_state_dict(state["sam_mask_decoder"])
        predictor.model.sam_prompt_encoder.load_state_dict(state["sam_prompt_encoder"])
    predictor.model.eval()
    return predictor


def read_stems(list_file):
    """Stems from a file of image paths or stems, one per line (e.g. train_photos25.txt from make_train_subset.py)."""
    return {Path(line.strip()).stem for line in Path(list_file).read_text().splitlines() if line.strip()}


def pairs(split_dir, max_images=None, stems=None):
    out = [(p, p.with_suffix(".png")) for p in sorted(Path(split_dir).glob("*.jpg")) if p.with_suffix(".png").exists()]
    if stems is not None:
        out = [x for x in out if x[0].stem in stems]
    return out[:max_images] if max_images else out


def read_pair(img_path, mask_path):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    gt = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    return img, gt


def instances(gt):
    """GT connected components >= MIN_AREA px and their xyxy boxes, in label order (as generate_teacher_logits.py)."""
    n, lab = cv2.connectedComponents(gt)
    masks, boxes = [], []
    for i in range(1, n):
        m = (lab == i).astype(np.uint8)
        if m.sum() < MIN_AREA:
            continue
        ys, xs = np.nonzero(m)
        masks.append(m)
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    return masks, np.array(boxes, dtype=np.float32).reshape(-1, 4)


def set_image(predictor, img):
    with torch.autocast("cuda", dtype=torch.float16, enabled=predictor.device.type == "cuda"):
        predictor.set_image(img)
    f = predictor._features
    predictor._features = {"image_embed": f["image_embed"].float(), "high_res_feats": [x.float() for x in f["high_res_feats"]]}


def decode(predictor, boxes):
    """Box prompts -> (low-res 256x256 logits, full-res logits), both N x 1 x H x W. Same path as
    SAM2ImagePredictor._predict, but differentiable."""
    _, _, _, box = predictor._prep_prompts(None, None, boxes, None, normalize_coords=True)
    labels = torch.tensor([[2, 3]], dtype=torch.int, device=box.device).repeat(box.shape[0], 1)
    model = predictor.model
    sparse, dense = model.sam_prompt_encoder(points=(box.reshape(-1, 2, 2), labels), boxes=None, masks=None)
    low, _, _, _ = model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"],
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False, repeat_image=box.shape[0] > 1,
        high_res_features=predictor._features["high_res_feats"])
    return low, predictor._transforms.postprocess_masks(low, predictor._orig_hw[-1])


def thin_mask(gt):
    dist = cv2.distanceTransform(gt, cv2.DIST_L2, 3)
    local_width = 2 * cv2.dilate(dist, np.ones((5, 5), np.uint8))
    return gt.astype(bool) & (local_width <= THIN_PX)


def accumulate(acc, pred, gt):
    """Add one image (pred: bool HxW, gt: uint8 HxW) to the running pixel counts."""
    g, thin = gt.astype(bool), thin_mask(gt)
    acc["tp"] += int((pred & g).sum()); acc["fp"] += int((pred & ~g).sum()); acc["fn"] += int((~pred & g).sum())
    acc["thin_tp"] += int((pred & thin).sum()); acc["thin_n"] += int(thin.sum())
    union = (pred | g).sum()
    acc["ious"].append(float((pred & g).sum() / union) if union else 1.0)
    return acc


def new_acc():
    return {"tp": 0, "fp": 0, "fn": 0, "thin_tp": 0, "thin_n": 0, "ious": []}


def summarize(acc):
    tp, fp, fn = acc["tp"], acc["fp"], acc["fn"]
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"images": len(acc["ious"]), "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9),
            "mean_iou": float(np.mean(acc["ious"])), "thin_recall": acc["thin_tp"] / max(acc["thin_n"], 1)}


@torch.no_grad()
def evaluate(predictor, split_dir, max_images=None):
    acc = new_acc()
    for img_path, mask_path in pairs(split_dir, max_images):
        img, gt = read_pair(img_path, mask_path)
        _, boxes = instances(gt)
        pred = np.zeros(gt.shape, bool)
        if len(boxes):
            set_image(predictor, img)
            pred = (decode(predictor, boxes)[1][:, 0] > 0).any(0).cpu().numpy()
        accumulate(acc, pred, gt)
    return summarize(acc)


def jitter(boxes, h, w, frac=0.05):
    bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    noise = np.random.uniform(-frac, frac, boxes.shape) * np.stack([bw, bh, bw, bh], 1)
    return np.clip(boxes + noise, 0, [w - 1, h - 1, w - 1, h - 1]).astype(np.float32)


def finetune(predictor, train_dir, val_dir, out, epochs=6, lr=1e-4, accum=4, max_train=None, max_val=None, seed=0,
             stems=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model = predictor.model
    params = [*model.sam_mask_decoder.parameters(), *model.sam_prompt_encoder.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    train = pairs(train_dir, max_train, stems)
    print(f"[finetune] {len(train)} train crops", flush=True)
    steps = epochs * math.ceil(len(train) / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / steps, 1))))

    history = [{"epoch": 0, **evaluate(predictor, val_dir, max_val)}]
    best = history[0]["f1"]
    print(f"[epoch 0] zero-shot val {history[0]}", flush=True)
    for epoch in range(1, epochs + 1):
        model.sam_mask_decoder.train(); model.sam_prompt_encoder.train()
        random.shuffle(train)
        t0, losses = time.time(), []
        opt.zero_grad()
        for i, (img_path, mask_path) in enumerate(train, 1):
            img, gt = read_pair(img_path, mask_path)
            if random.random() < 0.5:
                img, gt = img[:, ::-1], gt[:, ::-1]
            if random.random() < 0.5:
                img, gt = img[::-1], gt[::-1]
            img, gt = np.ascontiguousarray(img), np.ascontiguousarray(gt)
            masks, boxes = instances(gt)
            if not masks:
                continue
            set_image(predictor, img)
            logits = decode(predictor, jitter(boxes, *gt.shape))[1][:, 0]
            target = torch.as_tensor(np.stack(masks), dtype=torch.float32, device=logits.device)
            prob = logits.sigmoid()
            dice = 1 - (2 * (prob * target).sum((1, 2)) + 1) / (prob.sum((1, 2)) + target.sum((1, 2)) + 1)
            loss = F.binary_cross_entropy_with_logits(logits, target) + dice.mean()
            (loss / accum).backward()
            losses.append(loss.item())
            if i % accum == 0 or i == len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if i % 200 == 0:
                print(f"  epoch {epoch} {i}/{len(train)} loss {np.mean(losses[-200:]):.4f}", flush=True)
        model.eval()
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "minutes": (time.time() - t0) / 60,
               **evaluate(predictor, val_dir, max_val)}
        history.append(row)
        print(f"[epoch {epoch}] {row}", flush=True)
        if row["f1"] > best:
            best = row["f1"]
            torch.save({"sam_mask_decoder": model.sam_mask_decoder.state_dict(),
                        "sam_prompt_encoder": model.sam_prompt_encoder.state_dict(), "epoch": epoch, "val": row},
                       out / "sam2_crack_decoder.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return history


@torch.no_grad()
def cache_logits(predictor, img_dir, out_dir, prefix="crack500_", max_images=None):
    """Same instances, prompts and output format as generate_teacher_logits.py --prompt-type box."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for img_path, mask_path in pairs(img_dir, max_images):
        img, gt = read_pair(img_path, mask_path)
        _, boxes = instances(gt)
        if not len(boxes):
            continue
        predictor.set_image(img)
        logits = [predictor.predict(box=b, multimask_output=False)[2][0].astype(np.float32) for b in boxes]
        np.save(out_dir / f"{prefix}{img_path.stem}_logits.npy", np.stack(logits))
        written += 1
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["eval", "finetune", "logits"])
    ap.add_argument("--crack500", default="data/datasets/crack500")
    ap.add_argument("--ckpt", default="checkpoints/sam2_hiera_large.pt")
    ap.add_argument("--weights", default=None, help="fine-tuned decoder (from finetune); omit for zero-shot")
    ap.add_argument("--split", default="valcrop")
    ap.add_argument("--out", default="runs/teacher_ft")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-list", default=None, help="finetune only on these crops (file of image paths or stems)")
    ap.add_argument("--max", type=int, default=None, help="limit images (smoke tests)")
    ap.add_argument("--max-val", type=int, default=None)
    a = ap.parse_args()
    root = Path(a.crack500)
    predictor = load_predictor(a.ckpt, a.weights)
    if a.cmd == "eval":
        res = {"split": a.split, "weights": a.weights or "zero-shot", **evaluate(predictor, root / a.split, a.max)}
        print(json.dumps(res, indent=2))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        tag = Path(a.weights).stem if a.weights else "zeroshot"
        (Path(a.out) / f"eval_{tag}_{a.split}.json").write_text(json.dumps(res, indent=2) + "\n")
    elif a.cmd == "finetune":
        finetune(predictor, root / "traincrop", root / "valcrop", a.out, a.epochs, a.lr, max_train=a.max, max_val=a.max_val,
                 stems=read_stems(a.train_list) if a.train_list else None)
    else:
        print(f"[logits] wrote {cache_logits(predictor, root / 'traincrop', a.out, max_images=a.max)} files to {a.out}")


if __name__ == "__main__":
    main()
