"""
Unit tests for P0-3: Augmentation Policy Parity.
Verifies that spatial data augmentation policies are strictly identical across
both clean baseline and knowledge distillation control arms, eliminating the confound.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.config_loader import load_config, override_config
from distillation.kd_trainer import KDSegmentationTrainer


class TestAugmentationParity(unittest.TestCase):
    def setUp(self):
        self.master_cfg = load_config("configs/config.yaml")

    def test_augmentation_parity_default_disabled(self):
        """
        By default (allow_spatial_aug=False), both baseline and KD arms must
        have all spatial augmentations (mosaic, fliplr, scale, translate, erasing) set to 0.0.
        """
        # 1. Baseline arm
        cfg_baseline = override_config(self.master_cfg, {
            "distillation.enabled": False,
            "train.allow_spatial_aug": False,
            "train.epochs": 1,
            "train.amp": False,
        })
        trainer_baseline = KDSegmentationTrainer(cfg=cfg_baseline)

        # 2. KD arm
        cfg_kd = override_config(self.master_cfg, {
            "distillation.enabled": True,
            "train.allow_spatial_aug": False,
            "teacher.logits_dir": "data/teacher_logits/",
            "train.epochs": 1,
            "train.amp": False,
        })
        trainer_kd = KDSegmentationTrainer(cfg=cfg_kd)

        # Check baseline overrides
        self.assertEqual(trainer_baseline.args.mosaic, 0.0)
        self.assertEqual(trainer_baseline.args.fliplr, 0.0)
        self.assertEqual(trainer_baseline.args.scale, 0.0)
        self.assertEqual(trainer_baseline.args.translate, 0.0)
        self.assertEqual(trainer_baseline.args.erasing, 0.0)

        # Check KD overrides
        self.assertEqual(trainer_kd.args.mosaic, 0.0)
        self.assertEqual(trainer_kd.args.fliplr, 0.0)
        self.assertEqual(trainer_kd.args.scale, 0.0)
        self.assertEqual(trainer_kd.args.translate, 0.0)
        self.assertEqual(trainer_kd.args.erasing, 0.0)

        # Check exact equality across spatial augmentation keys
        spatial_keys = ["mosaic", "close_mosaic", "degrees", "translate", "scale", "shear", "perspective", "fliplr", "flipud", "erasing"]
        for k in spatial_keys:
            baseline_val = getattr(trainer_baseline.args, k)
            kd_val = getattr(trainer_kd.args, k)
            self.assertEqual(baseline_val, kd_val, f"Augmentation mismatch for key '{k}': baseline={baseline_val} vs kd={kd_val}")

    def test_augmentation_parity_when_spatial_aug_allowed(self):
        """
        When allow_spatial_aug=True, spatial augmentations are not zeroed out in either arm.
        """
        cfg_baseline_aug = override_config(self.master_cfg, {
            "distillation.enabled": False,
            "train.allow_spatial_aug": True,
            "train.epochs": 1,
            "train.amp": False,
        })
        trainer_baseline_aug = KDSegmentationTrainer(cfg=cfg_baseline_aug)

        cfg_kd_aug = override_config(self.master_cfg, {
            "distillation.enabled": True,
            "train.allow_spatial_aug": True,
            "teacher.logits_dir": "data/teacher_logits/",
            "train.epochs": 1,
            "train.amp": False,
        })
        trainer_kd_aug = KDSegmentationTrainer(cfg=cfg_kd_aug)

        # In Ultralytics defaults, fliplr is 0.5
        self.assertGreater(trainer_baseline_aug.args.fliplr, 0.0)
        self.assertGreater(trainer_kd_aug.args.fliplr, 0.0)
        self.assertEqual(trainer_baseline_aug.args.fliplr, trainer_kd_aug.args.fliplr)


if __name__ == "__main__":
    unittest.main()
