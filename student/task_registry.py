"""
Task Registry — Crack-Distill
==============================
This is the single place that maps task.type → concrete implementations.

To add a new task (e.g. panoptic_seg):
  1. Create StudentAdapter and LossAdapter subclasses below
  2. Register them in TASK_REGISTRY
  3. Change config.yaml task.type — nothing else.
"""

from dataclasses import dataclass
from typing import Type
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# Base interfaces  (all tasks must implement these)
# ──────────────────────────────────────────────

class BaseStudentAdapter:
    """Wraps any student model into a unified interface."""

    def __init__(self, cfg): ...

    def load_model(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, images):
        raise NotImplementedError

    def predict(self, images):
        """Post-processed inference output."""
        raise NotImplementedError


class BaseLossAdapter:
    """Unified loss interface for all task types."""

    def __init__(self, cfg): ...

    def task_loss(self, preds, targets) -> torch.Tensor:
        raise NotImplementedError

    def mask_kd_loss(self, student_logits, teacher_logits) -> torch.Tensor:
        raise NotImplementedError

    def feature_loss(self, student_feats, teacher_feats) -> torch.Tensor:
        raise NotImplementedError

    def boundary_loss(self, student_logits, teacher_logits) -> torch.Tensor:
        raise NotImplementedError

    def total_loss(self, preds, targets, student_logits,
                   teacher_logits, student_feats, teacher_feats) -> dict:
        cfg = self.cfg.distillation
        losses = {}

        losses["task"] = self.task_loss(preds, targets) * cfg.losses.task.weight

        if cfg.losses.mask_kd.enabled:
            losses["mask_kd"] = (
                self.mask_kd_loss(student_logits, teacher_logits)
                * cfg.losses.mask_kd.weight
            )

        if cfg.losses.feature.enabled:
            losses["feature"] = (
                self.feature_loss(student_feats, teacher_feats)
                * cfg.losses.feature.weight
            )

        if cfg.losses.boundary.enabled:
            losses["boundary"] = (
                self.boundary_loss(student_logits, teacher_logits)
                * cfg.losses.boundary.weight
            )

        losses["total"] = sum(losses.values())
        return losses


class BaseDataAdapter:
    """Unified dataset interface for all task types."""

    def __init__(self, cfg): ...

    def get_train_loader(self): raise NotImplementedError
    def get_val_loader(self):   raise NotImplementedError
    def get_test_loader(self):  raise NotImplementedError


# ──────────────────────────────────────────────
# Instance Segmentation Adapters
# ──────────────────────────────────────────────

class InstanceSegStudentAdapter(BaseStudentAdapter):
    """YOLO11-seg / RTMDet-Ins student wrapper."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.backbone = cfg.student.backbone
        self.device = cfg.student.device

    def load_model(self) -> nn.Module:
        from ultralytics import YOLO
        model = YOLO(f"{self.backbone}.pt")
        return model

    def forward(self, images):
        return self.model(images)

    def predict(self, images, conf=0.25, iou=0.45):
        return self.model.predict(images, conf=conf, iou=iou, task="segment")


class InstanceSegLossAdapter(BaseLossAdapter):
    """KD losses for instance segmentation."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.T = cfg.distillation.temperature

    def task_loss(self, preds, targets):
        # YOLO11 computes this internally; expose for custom override
        return preds["loss"] if isinstance(preds, dict) else preds[0]

    def mask_kd_loss(self, student_logits, teacher_logits):
        """KL divergence on soft mask probabilities."""
        import torch.nn.functional as F
        T = self.T
        s = F.log_softmax(student_logits / T, dim=1)
        t = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(s, t, reduction="batchmean") * (T ** 2)

    def feature_loss(self, student_feats, teacher_feats):
        """L2 feature alignment with 1×1 conv projection."""
        import torch.nn.functional as F
        loss = 0.0
        for sf, tf in zip(student_feats, teacher_feats):
            # Align spatial dims if needed
            if sf.shape != tf.shape:
                tf = F.interpolate(tf, size=sf.shape[2:])
            loss += F.mse_loss(sf, tf)
        return loss / max(len(student_feats), 1)

    def boundary_loss(self, student_logits, teacher_logits):
        """Up-weight loss at boundary pixels (where teacher is uncertain ~0.5)."""
        import torch
        import torch.nn.functional as F
        t_prob = torch.sigmoid(teacher_logits)
        # Boundary pixels: teacher probability close to 0.5
        boundary_weight = 1.0 - torch.abs(t_prob - 0.5) * 2  # peaks at 0.5
        boundary_weight = boundary_weight.detach()

        s_prob = torch.sigmoid(student_logits)
        bce = F.binary_cross_entropy(s_prob, t_prob.detach(), reduction="none")
        return (bce * boundary_weight).mean()


class InstanceSegDataAdapter(BaseDataAdapter):
    """Loads instance seg datasets, handles format conversion."""

    def __init__(self, cfg):
        self.cfg = cfg

    def get_train_loader(self):
        from data.dataset import CrackDistillDataset
        from torch.utils.data import DataLoader
        ds = CrackDistillDataset(self.cfg, split="train", task="instance_seg")
        return DataLoader(ds, batch_size=self.cfg.data.batch_size,
                          shuffle=True, num_workers=self.cfg.data.num_workers,
                          collate_fn=ds.collate_fn)

    def get_val_loader(self):
        from data.dataset import CrackDistillDataset
        from torch.utils.data import DataLoader
        ds = CrackDistillDataset(self.cfg, split="val", task="instance_seg")
        return DataLoader(ds, batch_size=self.cfg.data.batch_size,
                          shuffle=False, num_workers=self.cfg.data.num_workers,
                          collate_fn=ds.collate_fn)

    def get_test_loader(self):
        from data.dataset import CrackDistillDataset
        from torch.utils.data import DataLoader
        ds = CrackDistillDataset(self.cfg, split="test", task="instance_seg")
        return DataLoader(ds, batch_size=1, shuffle=False,
                          num_workers=self.cfg.data.num_workers,
                          collate_fn=ds.collate_fn)


# ──────────────────────────────────────────────
# Semantic Segmentation Adapters  (future swap)
# ──────────────────────────────────────────────

class SemanticSegStudentAdapter(BaseStudentAdapter):
    """Placeholder — swap backbone to DeepLabV3+, SegFormer, etc."""

    def __init__(self, cfg):
        self.cfg = cfg

    def load_model(self):
        # Example: from mmseg import build_segmentor
        raise NotImplementedError("Implement semantic backbone here")

    def predict(self, images):
        raise NotImplementedError


class SemanticSegLossAdapter(BaseLossAdapter):
    """KD losses for semantic seg (pixel-wise CE + KD)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.T = cfg.distillation.temperature

    def task_loss(self, preds, targets):
        import torch.nn.functional as F
        return F.cross_entropy(preds, targets)

    def mask_kd_loss(self, student_logits, teacher_logits):
        import torch.nn.functional as F
        T = self.T
        s = F.log_softmax(student_logits / T, dim=1)
        t = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(s, t, reduction="batchmean") * (T ** 2)

    def feature_loss(self, student_feats, teacher_feats):
        import torch.nn.functional as F
        loss = sum(F.mse_loss(sf, F.interpolate(tf, sf.shape[2:]))
                   for sf, tf in zip(student_feats, teacher_feats))
        return loss / max(len(student_feats), 1)

    def boundary_loss(self, student_logits, teacher_logits):
        # Reuse same boundary logic
        return InstanceSegLossAdapter.boundary_loss(
            self, student_logits, teacher_logits
        )


class SemanticSegDataAdapter(BaseDataAdapter):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_train_loader(self):
        from data.dataset import CrackDistillDataset
        from torch.utils.data import DataLoader
        ds = CrackDistillDataset(self.cfg, split="train", task="semantic_seg")
        return DataLoader(ds, batch_size=self.cfg.data.batch_size, shuffle=True)

    def get_val_loader(self):
        from data.dataset import CrackDistillDataset
        from torch.utils.data import DataLoader
        ds = CrackDistillDataset(self.cfg, split="val", task="semantic_seg")
        return DataLoader(ds, batch_size=self.cfg.data.batch_size, shuffle=False)

    def get_test_loader(self):
        from data.dataset import CrackDistillDataset
        from torch.utils.data import DataLoader
        ds = CrackDistillDataset(self.cfg, split="test", task="semantic_seg")
        return DataLoader(ds, batch_size=1, shuffle=False)


# ──────────────────────────────────────────────
# Detection Adapters  (future swap)
# ──────────────────────────────────────────────

class DetectionStudentAdapter(BaseStudentAdapter):
    def __init__(self, cfg):
        self.cfg = cfg

    def load_model(self):
        from ultralytics import YOLO
        return YOLO("yolo11n.pt")  # detection, no seg head

    def predict(self, images):
        return self.model.predict(images, task="detect")


class DetectionLossAdapter(BaseLossAdapter):
    def __init__(self, cfg):
        self.cfg = cfg
        self.T = cfg.distillation.temperature

    def task_loss(self, preds, targets):
        return preds["loss"] if isinstance(preds, dict) else preds[0]

    def mask_kd_loss(self, s, t):
        # For detection: distill on cls logits not masks
        import torch.nn.functional as F
        T = self.T
        return F.kl_div(F.log_softmax(s/T, -1),
                        F.softmax(t/T, -1), reduction="batchmean") * T**2

    def feature_loss(self, sf, tf):
        import torch.nn.functional as F
        return sum(F.mse_loss(s, F.interpolate(t, s.shape[2:]))
                   for s, t in zip(sf, tf)) / max(len(sf), 1)

    def boundary_loss(self, s, t):
        return torch.tensor(0.0)  # not applicable for detection


class DetectionDataAdapter(BaseDataAdapter):
    def __init__(self, cfg):
        self.cfg = cfg

    def get_train_loader(self):
        from data.dataset import CrackDistillDataset
        from torch.utils.data import DataLoader
        ds = CrackDistillDataset(self.cfg, split="train", task="detection")
        return DataLoader(ds, batch_size=self.cfg.data.batch_size, shuffle=True)

    def get_val_loader(self):
        from data.dataset import CrackDistillDataset
        from torch.utils.data import DataLoader
        ds = CrackDistillDataset(self.cfg, split="val", task="detection")
        return DataLoader(ds, batch_size=self.cfg.data.batch_size, shuffle=False)

    def get_test_loader(self):
        from data.dataset import CrackDistillDataset
        from torch.utils.data import DataLoader
        ds = CrackDistillDataset(self.cfg, split="test", task="detection")
        return DataLoader(ds, batch_size=1, shuffle=False)


# ──────────────────────────────────────────────
# TASK REGISTRY — single lookup table
# ──────────────────────────────────────────────

@dataclass
class TaskAdapters:
    student: Type[BaseStudentAdapter]
    loss:    Type[BaseLossAdapter]
    data:    Type[BaseDataAdapter]


TASK_REGISTRY = {
    "instance_seg": TaskAdapters(
        student=InstanceSegStudentAdapter,
        loss=InstanceSegLossAdapter,
        data=InstanceSegDataAdapter,
    ),
    "semantic_seg": TaskAdapters(
        student=SemanticSegStudentAdapter,
        loss=SemanticSegLossAdapter,
        data=SemanticSegDataAdapter,
    ),
    "detection": TaskAdapters(
        student=DetectionStudentAdapter,
        loss=DetectionLossAdapter,
        data=DetectionDataAdapter,
    ),
    # Add new tasks here — nothing else changes
}


def get_task_adapters(cfg) -> TaskAdapters:
    task_type = cfg.task.type
    if task_type not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_type}'. "
            f"Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_type]
