"""
Dataset — Crack-Distill
========================
Unified dataset class that:
  - Loads COCO / YOLO / binary-mask / VOC formats
  - Converts everything to instance OR semantic targets
  - Injects teacher logits on-the-fly during KD training
  - Handles cv2.connectedComponents for touching crack separation
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ──────────────────────────────────────────────
# Augmentation pipelines
# ──────────────────────────────────────────────

def get_transforms(split: str, img_size: int = 512) -> A.Compose:
    """Consistent augmentations with mask-safe transforms."""

    bbox_params = A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
    )

    if split == "train":
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.RandomRain(p=0.1),                    # robustness test data
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=bbox_params)

    else:  # val / test
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=bbox_params)


# ──────────────────────────────────────────────
# Format loaders
# ──────────────────────────────────────────────

def load_coco_annotations(ann_file: str) -> dict:
    """Load COCO format annotations."""
    with open(ann_file) as f:
        data = json.load(f)
    # Build image_id → annotations lookup
    ann_map = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        ann_map.setdefault(iid, []).append(ann)
    img_map = {img["id"]: img for img in data["images"]}
    return {"images": img_map, "annotations": ann_map, "categories": data["categories"]}


def binary_mask_to_instances(binary_mask: np.ndarray) -> list[np.ndarray]:
    """
    Convert a binary segmentation mask to a list of instance masks
    using connected components. Handles touching cracks.

    Returns: list of (H, W) bool arrays, one per connected component.
    """
    mask_uint8 = binary_mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_uint8)
    instances = []
    for label_id in range(1, num_labels):  # 0 = background
        instance_mask = (labels == label_id)
        # Filter tiny noise (< 50 pixels)
        if instance_mask.sum() >= 50:
            instances.append(instance_mask)
    return instances


def mask_to_polygon(mask: np.ndarray) -> list:
    """Convert binary mask to COCO-style polygon contours."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if contour.size >= 6:
            polygons.append(contour.flatten().tolist())
    return polygons


def mask_to_yolo_seg_line(mask: np.ndarray, class_id: int, img_h: int, img_w: int) -> str:
    """Convert binary mask to YOLO seg label line."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).flatten().tolist()
    # Normalize to [0, 1]
    norm = []
    for i, v in enumerate(contour):
        norm.append(v / img_w if i % 2 == 0 else v / img_h)
    coords_str = " ".join(f"{v:.6f}" for v in norm)
    return f"{class_id} {coords_str}"


# ──────────────────────────────────────────────
# Main Dataset Class
# ──────────────────────────────────────────────

class CrackDistillDataset(Dataset):
    """
    Unified dataset for Crack-Distill pipeline.

    task: "instance_seg" | "semantic_seg" | "detection"
    Format conversion happens automatically.
    Teacher logits are loaded on-the-fly if available.
    """

    def __init__(
        self,
        cfg,
        split: str = "train",        # train | val | test
        task: str = "instance_seg",
        logits_dir: Optional[str] = None,
    ):
        self.cfg = cfg
        self.split = split
        self.task = task
        self.img_size = cfg.data.image_size
        self.logits_dir = Path(logits_dir or cfg.teacher.logits_dir)
        self.transform = get_transforms(split, self.img_size)

        # Load all datasets specified in config
        self.samples = []
        for ds_cfg in cfg.data.datasets:
            samples = self._load_dataset(ds_cfg)
            self.samples.extend(samples)

        # Apply split
        np.random.seed(cfg.project.seed)
        indices = np.random.permutation(len(self.samples))
        n = len(indices)
        n_train = int(n * cfg.data.train_split)
        n_val   = int(n * cfg.data.val_split)

        if split == "train":
            self.samples = [self.samples[i] for i in indices[:n_train]]
        elif split == "val":
            self.samples = [self.samples[i] for i in indices[n_train:n_train+n_val]]
        else:  # test
            self.samples = [self.samples[i] for i in indices[n_train+n_val:]]

        # Apply low-data fraction if set
        if hasattr(cfg.data, "train_fraction") and split == "train":
            k = max(1, int(len(self.samples) * cfg.data.train_fraction))
            self.samples = self.samples[:k]

        print(f"[Dataset] {split}: {len(self.samples)} samples (task={task})")

    def _load_dataset(self, ds_cfg) -> list:
        """Load samples from one dataset config entry."""
        # Support both ConfigNode (dot access) and plain dict (bracket access)
        if hasattr(ds_cfg, "format"):
            fmt  = ds_cfg.format
            path = Path(str(ds_cfg.path)).expanduser()
            name = ds_cfg.name
        else:
            fmt  = ds_cfg.get("format", "coco")
            path = Path(ds_cfg["path"]).expanduser()
            name = ds_cfg["name"]

        if fmt == "coco":
            return self._load_coco(path, name)
        elif fmt == "yolo":
            return self._load_yolo(path, name)
        elif fmt == "binary_mask":
            return self._load_binary_mask(path, name)
        else:
            raise ValueError(f"Unknown format: {fmt}")

    def _load_coco(self, path: Path, name: str) -> list:
        ann_file = path / "annotations" / "instances.json"
        if not ann_file.exists():
            # Try common COCO directory structures
            for candidate in path.rglob("instances*.json"):
                ann_file = candidate
                break
        coco = load_coco_annotations(str(ann_file))
        samples = []
        for img_id, img_info in coco["images"].items():
            img_path = path / "images" / img_info["file_name"]
            anns = coco["annotations"].get(img_id, [])
            samples.append({
                "image_path": str(img_path),
                "image_id": f"{name}_{img_id}",
                "annotations": anns,
                "format": "coco",
                "dataset": name,
            })
        return samples

    def _load_yolo(self, path: Path, name: str) -> list:
        img_dir = path / "images"
        lbl_dir = path / "labels"
        samples = []
        for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            samples.append({
                "image_path": str(img_path),
                "image_id": f"{name}_{img_path.stem}",
                "label_path": str(lbl_path) if lbl_path.exists() else None,
                "format": "yolo",
                "dataset": name,
            })
        return samples

    def _load_binary_mask(self, path: Path, name: str) -> list:
        img_dir  = path / "images"
        mask_dir = path / "masks"
        samples = []
        for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            mask_path = mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                mask_path = mask_dir / (img_path.stem + ".bmp")
            samples.append({
                "image_path": str(img_path),
                "image_id": f"{name}_{img_path.stem}",
                "mask_path": str(mask_path) if mask_path.exists() else None,
                "format": "binary_mask",
                "dataset": name,
            })
        return samples

    def _load_instance_targets(self, sample: dict) -> dict:
        """
        Load and return per-instance masks and bboxes.
        Handles all input formats → unified output.
        """
        fmt = sample["format"]
        targets = {"masks": [], "bboxes": [], "class_ids": []}

        if fmt == "coco":
            for ann in sample.get("annotations", []):
                if "segmentation" not in ann or not ann["segmentation"]:
                    continue
                # Convert polygon to mask
                img = cv2.imread(sample["image_path"])
                H, W = img.shape[:2]
                mask = np.zeros((H, W), dtype=np.uint8)
                for seg in ann["segmentation"]:
                    pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts], 1)
                targets["masks"].append(mask.astype(bool))
                x, y, w, h = ann["bbox"]
                targets["bboxes"].append([x, y, x+w, y+h])
                targets["class_ids"].append(ann["category_id"] - 1)

        elif fmt == "yolo":
            img = cv2.imread(sample["image_path"])
            H, W = img.shape[:2]
            lbl_path = sample.get("label_path")
            if lbl_path and os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        coords = list(map(float, parts[1:]))
                        # YOLO seg: class x1 y1 x2 y2 ... (normalized polygon)
                        pts = np.array(coords).reshape(-1, 2)
                        pts[:, 0] *= W
                        pts[:, 1] *= H
                        pts = pts.astype(np.int32)
                        mask = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillPoly(mask, [pts], 1)
                        targets["masks"].append(mask.astype(bool))
                        targets["class_ids"].append(cls_id)

        elif fmt == "binary_mask":
            mask_path = sample.get("mask_path")
            if mask_path and os.path.exists(mask_path):
                binary = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                binary = (binary > 127).astype(np.uint8)
                # Split into individual instances via connected components
                instances = binary_mask_to_instances(binary)
                for inst_mask in instances:
                    targets["masks"].append(inst_mask)
                    targets["class_ids"].append(0)  # single class: crack

        return targets

    def _load_semantic_targets(self, sample: dict) -> np.ndarray:
        """Load semantic segmentation mask (H, W) with class IDs."""
        fmt = sample["format"]
        img = cv2.imread(sample["image_path"])
        H, W = img.shape[:2]
        semantic_mask = np.zeros((H, W), dtype=np.int64)

        if fmt == "coco":
            for ann in sample.get("annotations", []):
                if "segmentation" not in ann:
                    continue
                for seg in ann.get("segmentation", []):
                    pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(semantic_mask, [pts], ann["category_id"])

        elif fmt == "binary_mask":
            mask_path = sample.get("mask_path")
            if mask_path and os.path.exists(mask_path):
                binary = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                semantic_mask = (binary > 127).astype(np.int64)

        return semantic_mask

    def _load_teacher_logits(self, image_id: str) -> Optional[np.ndarray]:
        """Load pre-saved SAM logits. Returns None if not available."""
        logit_path = self.logits_dir / f"{image_id}_logits.npy"
        if logit_path.exists():
            return np.load(str(logit_path))
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load targets based on task
        if self.task == "instance_seg":
            targets = self._load_instance_targets(sample)
        elif self.task == "semantic_seg":
            targets = {"semantic_mask": self._load_semantic_targets(sample)}
        elif self.task == "detection":
            targets = self._load_instance_targets(sample)  # bbox only used
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Resize to fixed square — handles portrait/landscape mixed batches
        sz = self.img_size
        image = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_LINEAR)

        image_tensor = torch.from_numpy(
            image.transpose(2, 0, 1).astype(np.float32) / 255.0
        )

        # Load teacher logits (may be None if not yet generated)
        teacher_logits = self._load_teacher_logits(sample["image_id"])

        return {
            "image":          image_tensor,
            "image_id":       sample["image_id"],
            "image_path":     sample["image_path"],
            "targets":        targets,
            "teacher_logits": teacher_logits,  # None if not yet run
            "task":           self.task,
        }

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """Custom collate: handles variable number of instances per image."""
        images     = torch.stack([b["image"] for b in batch])
        image_ids  = [b["image_id"] for b in batch]
        image_paths= [b["image_path"] for b in batch]
        targets    = [b["targets"] for b in batch]
        logits     = [b["teacher_logits"] for b in batch]
        tasks      = [b["task"] for b in batch]

        return {
            "images":         images,
            "image_ids":      image_ids,
            "image_paths":    image_paths,
            "targets":        targets,         # list of dicts (variable instances)
            "teacher_logits": logits,          # list, some may be None
            "task":           tasks[0],
        }