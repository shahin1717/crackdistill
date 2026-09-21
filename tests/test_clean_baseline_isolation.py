"""
Unit tests for P0-2: Clean Baseline Control Arm Isolation.
Verifies that setting distillation.enabled=False completely isolates
the baseline trainer from KD hooks, loss patching, dataset wrapping,
and teacher logit requirements.
"""

import os
import sys
import unittest
from pathlib import Path
import torch
import torch.nn as nn

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.config_loader import load_config, override_config, ConfigNode
from distillation.kd_trainer import KDSegmentationTrainer, KDYOLODataset


class TestCleanBaselineIsolation(unittest.TestCase):
    def setUp(self):
        self.master_cfg = load_config("configs/config.yaml")

    def test_baseline_initialization_without_logits(self):
        """
        Baseline arm must initialize cleanly even if logits_dir does not exist,
        without throwing FileNotFoundError or attempting symlink fallbacks.
        """
        overrides = {
            "distillation.enabled": False,
            "distillation.losses.mask_kd.enabled": False,
            "distillation.losses.feature.enabled": False,
            "distillation.losses.boundary.enabled": False,
            "teacher.logits_dir": "/tmp/nonexistent_teacher_logits_dir_12345",
            "train.epochs": 1,
            "train.amp": False,
        }
        cfg = override_config(self.master_cfg, overrides)
        trainer = KDSegmentationTrainer(cfg=cfg)

        self.assertFalse(trainer.is_kd_on)
        self.assertEqual(len(trainer._sam_targets), 0)
        self.assertIsInstance(trainer.proj_layers, nn.ModuleDict)
        self.assertEqual(len(trainer.proj_layers), 0)

    def test_baseline_setup_model_loss_and_hooks_unpatched(self):
        """
        setup_model() in baseline arm must NOT patch model.loss and must NOT register forward hooks.
        """
        overrides = {
            "distillation.enabled": False,
            "train.epochs": 1,
            "train.amp": False,
        }
        cfg = override_config(self.master_cfg, overrides)
        trainer = KDSegmentationTrainer(cfg=cfg)

        # Call setup_model
        trainer.setup_model()

        # 1. Verify model.loss is unpatched (no original_loss attribute)
        self.assertFalse(hasattr(trainer.model, "original_loss"), "model.loss was patched when KD was disabled!")

        # 2. Verify no active hooks are registered
        self.assertEqual(len(trainer._hook_handles), 0)
        for module in trainer.model.modules():
            self.assertEqual(len(module._forward_hooks), 0, f"Module {module} has forward hooks registered in clean baseline!")

    def test_baseline_build_dataset_unwrapped(self):
        """
        build_dataset() in baseline arm must return base YOLODataset, NOT KDYOLODataset.
        """
        overrides = {
            "distillation.enabled": False,
        }
        cfg = override_config(self.master_cfg, overrides)
        trainer = KDSegmentationTrainer(cfg=cfg)
        trainer.setup_model()

        data_path = "data/datasets/crack500_yolo/images/train"
        if Path(data_path).exists():
            dataset = trainer.build_dataset(img_path=data_path, mode="train", batch=4)
            self.assertNotIsInstance(dataset, KDYOLODataset, "Training dataset was wrapped in KDYOLODataset in clean baseline!")

    def test_baseline_kd_loss_returns_empty(self):
        """
        _kd_loss_from_preds() must unconditionally return {} when is_kd_on is False.
        """
        overrides = {
            "distillation.enabled": False,
        }
        cfg = override_config(self.master_cfg, overrides)
        trainer = KDSegmentationTrainer(cfg=cfg)
        trainer.setup_model()

        # Dummy inputs
        dummy_preds = torch.randn(1, 32, 160, 160)
        dummy_batch = {"img": torch.randn(1, 3, 512, 512)}

        kd_losses = trainer._kd_loss_from_preds(dummy_preds, dummy_batch, trainer.model)
        self.assertEqual(kd_losses, {})

    def test_kd_enabled_arm_activates_properly(self):
        """
        When distillation.enabled is True, trainer must set is_kd_on=True.
        """
        overrides = {
            "distillation.enabled": True,
            "teacher.logits_dir": "data/teacher_logits/",
        }
        cfg = override_config(self.master_cfg, overrides)
        trainer = KDSegmentationTrainer(cfg=cfg)
        self.assertTrue(trainer.is_kd_on)


if __name__ == "__main__":
    unittest.main()
