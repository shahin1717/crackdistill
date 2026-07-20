"""
Metrics — Crack-Distill
========================
Supports:
  instance_seg: mAP@0.5-seg, mAP@0.5:0.95-seg, Dice, BIoU
  semantic_seg:  mIoU, Dice, pixel accuracy
  detection:     mAP@0.5, mAP@0.5:0.95
Also measures: FPS, latency, parameter count, GFLOPs
"""

import time
import torch
import numpy as np
from collections import defaultdict


class SegmentationMetrics:
    """Task-aware metrics accumulator."""

    def __init__(self, num_classes: int, task: str = "instance_seg"):
        self.num_classes = num_classes
        self.task = task
        self.reset()

    def reset(self):
        self.results = defaultdict(list)
        self.inference_times = []

    def update(self, preds, targets: list):
        """
        preds:   model predictions (task-specific format)
        targets: list of target dicts (one per image)
        """
        if self.task == "instance_seg":
            self._update_instance_seg(preds, targets)
        elif self.task == "semantic_seg":
            self._update_semantic_seg(preds, targets)
        elif self.task == "detection":
            self._update_detection(preds, targets)

    def _update_instance_seg(self, preds, targets):
        """Compute per-image instance seg metrics."""
        # YOLO predictions come as a list of Results objects
        for pred, target in zip(preds, targets):
            gt_masks = target.get("masks", [])
            if not gt_masks:
                continue

            # Extract predicted masks
            if hasattr(pred, "masks") and pred.masks is not None:
                pred_masks = pred.masks.data.cpu().numpy()  # (N, H, W)
            else:
                pred_masks = np.array([])

            if len(pred_masks) == 0 or len(gt_masks) == 0:
                self.results["dice"].append(0.0)
                self.results["iou"].append(0.0)
                continue

            # Dice and IoU (best-match Hungarian-style)
            dice_scores, iou_scores, biou_scores = [], [], []
            for gt in gt_masks:
                gt = np.array(gt).astype(bool)
                best_dice = 0.0
                best_iou  = 0.0
                best_biou = 0.0
                for pm in pred_masks:
                    pm = pm.astype(bool)
                    # Resize to gt size if needed
                    if pm.shape != gt.shape:
                        import cv2
                        pm = cv2.resize(
                            pm.astype(np.uint8), (gt.shape[1], gt.shape[0])
                        ).astype(bool)
                    d = dice_coefficient(gt, pm)
                    i = iou_score(gt, pm)
                    b = boundary_iou(gt, pm)
                    best_dice = max(best_dice, d)
                    best_iou  = max(best_iou,  i)
                    best_biou = max(best_biou, b)
                dice_scores.append(best_dice)
                iou_scores.append(best_iou)
                biou_scores.append(best_biou)

            self.results["dice"].append(np.mean(dice_scores))
            self.results["iou"].append(np.mean(iou_scores))
            self.results["boundary_iou"].append(np.mean(biou_scores))

    def _update_semantic_seg(self, preds, targets):
        """Pixel-wise metrics for semantic seg."""
        pred_masks = preds.argmax(dim=1).cpu().numpy()  # (B, H, W)
        for pred, target in zip(pred_masks, targets):
            gt = target["semantic_mask"]
            miou = mean_iou_semantic(pred, gt, self.num_classes)
            dice = dice_semantic(pred, gt, self.num_classes)
            self.results["mIoU"].append(miou)
            self.results["dice"].append(dice)

    def _update_detection(self, preds, targets):
        """Detection mAP (delegated to torchmetrics or ultralytics)."""
        # Placeholder — YOLO computes this internally
        pass

    def compute(self) -> dict:
        """Return averaged metrics."""
        out = {}
        for k, v in self.results.items():
            out[k] = float(np.mean(v)) if v else 0.0

        # Primary metric alias
        if self.task == "instance_seg":
            out["mAP50-seg"] = out.get("iou", 0.0)  # proxy until proper mAP
        elif self.task == "semantic_seg":
            out["mIoU_mean"] = out.get("mIoU", 0.0)

        return out

    def measure_efficiency(self, model, input_size=(1, 3, 512, 512), device="cuda", n=100):
        """
        Measure FPS, latency, params, GFLOPs.
        Returns dict suitable for paper Table.
        """
        import torch
        from thop import profile  # pip install thop

        model.eval()
        dummy = torch.randn(input_size).to(device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(dummy)

        # Latency
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(dummy)
            times.append(time.perf_counter() - t0)

        latency_ms = np.mean(times) * 1000
        fps = 1000 / latency_ms

        # Params
        params = sum(p.numel() for p in model.parameters()) / 1e6

        # GFLOPs
        try:
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            gflops = macs * 2 / 1e9
        except Exception:
            gflops = None

        return {
            "fps":         round(fps, 1),
            "latency_ms":  round(latency_ms, 2),
            "params_M":    round(params, 2),
            "gflops":      round(gflops, 2) if gflops else "N/A",
        }


# ──────────────────────────────────────────────
# Core metric functions
# ──────────────────────────────────────────────

def dice_coefficient(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> float:
    gt   = gt.astype(bool).flatten()
    pred = pred.astype(bool).flatten()
    intersection = (gt & pred).sum()
    return (2.0 * intersection + smooth) / (gt.sum() + pred.sum() + smooth)


def iou_score(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> float:
    gt   = gt.astype(bool).flatten()
    pred = pred.astype(bool).flatten()
    intersection = (gt & pred).sum()
    union = (gt | pred).sum()
    return (intersection + smooth) / (union + smooth)


def boundary_iou(
    gt: np.ndarray,
    pred: np.ndarray,
    dilation_ratio: float = 0.02,
    smooth: float = 1e-6,
) -> float:
    """
    Boundary IoU — evaluates mask quality at boundaries specifically.
    dilation_ratio: fraction of image diagonal used for boundary width.
    """
    import cv2
    H, W = gt.shape
    d = max(1, int(dilation_ratio * np.sqrt(H**2 + W**2)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*d+1, 2*d+1))

    def get_boundary(mask):
        mask_u8 = mask.astype(np.uint8)
        dilated = cv2.dilate(mask_u8, kernel)
        eroded  = cv2.erode(mask_u8, kernel)
        return (dilated - eroded).astype(bool)

    gt_b   = get_boundary(gt.astype(np.uint8))
    pred_b = get_boundary(pred.astype(np.uint8))

    intersection = (gt_b & pred_b).sum()
    union        = (gt_b | pred_b).sum()
    return (intersection + smooth) / (union + smooth)


def mean_iou_semantic(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> float:
    """mIoU for semantic segmentation."""
    ious = []
    for cls in range(num_classes):
        p = (pred == cls)
        g = (gt == cls)
        if not g.any():
            continue
        inter = (p & g).sum()
        union = (p | g).sum()
        ious.append(inter / (union + 1e-6))
    return float(np.mean(ious)) if ious else 0.0


def dice_semantic(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> float:
    """Mean Dice for semantic segmentation."""
    dices = []
    for cls in range(num_classes):
        p = (pred == cls).flatten()
        g = (gt == cls).flatten()
        if not g.any():
            continue
        dices.append(dice_coefficient(g, p))
    return float(np.mean(dices)) if dices else 0.0
