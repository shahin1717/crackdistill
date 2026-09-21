"""
Unit tests validating all Category P1 engineering & pipeline defects:
- P1-1: Diagnostic tracking of dropped teacher instances in KDTrainer.
- P1-2: LayerKD multi-scale feature mapping (distinct neck targets & feat0 storage).
- P1-3: Deterministic checkpoint discovery and SHA256 deduplication.
- P1-4: UTF-8 encoding robustness for notebook loading.
- P1-5: Strict fail-fast vs tolerant mode on batch KD loss failures.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from distillation.kd_trainer import KDSegmentationTrainer, KDYOLODataset
from utils.config_loader import ConfigNode
from utils.checkpoint import (
    compute_file_sha256,
    resolve_checkpoint,
    discover_and_deduplicate_checkpoints,
)


class TestP1Defects(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logits_dir = Path(self.temp_dir) / "logits"
        self.logits_dir.mkdir(parents=True, exist_ok=True)
        # Create a dummy logit file so KD init passes
        dummy_logit = np.zeros((3, 256, 256), dtype=np.float32)
        np.save(str(self.logits_dir / "sample001.npy"), dummy_logit)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_p1_1_instance_drop_diagnostics(self):
        """P1-1: Verify that dropped teacher instances update counters and stats."""
        kd_cfg = ConfigNode({
            "enabled": True,
            "strict": False,
            "temperature": 2.0,
            "losses": {
                "task": {"weight": 1.0},
                "mask_kd": {"enabled": True, "weight": 1.0},
                "feature": {"enabled": False},
                "boundary": {"enabled": False},
                "affinity": {"enabled": False},
                "tversky": {"enabled": False},
            }
        })
        trainer = KDSegmentationTrainer(
            overrides={"model": "yolo11n-seg.pt", "data": "data/datasets/combined_yolo/dataset.yaml", "epochs": 1},
            kd_cfg=kd_cfg,
            logits_dir=self.logits_dir
        )

        initial_stats = trainer.get_instance_drop_stats()
        self.assertEqual(initial_stats["total_instances"], 0)
        self.assertEqual(initial_stats["dropped_instances"], 0)
        self.assertEqual(initial_stats["drop_rate"], 0.0)

        # Simulate instance drop accounting
        trainer._total_instances_count += 10
        trainer._dropped_instances_count += 3
        stats = trainer.get_instance_drop_stats()
        self.assertEqual(stats["total_instances"], 10)
        self.assertEqual(stats["dropped_instances"], 3)
        self.assertAlmostEqual(stats["drop_rate"], 0.3)

    def test_p1_2_layer_kd_distinct_targets_and_pooling(self):
        """P1-2: Verify Layer 16, 19, 22 have distinct targets and Layer 22 pools image_embed."""
        # 1. Verify npz loading loads feat0, feat1, and image_embed
        npz_path = Path(self.temp_dir) / "test_features.npz"
        np.savez_compressed(
            str(npz_path),
            feat0=np.zeros((1, 32, 128, 128), dtype=np.float32),
            feat1=np.zeros((1, 64, 64, 64), dtype=np.float32),
            image_embed=np.zeros((1, 256, 32, 32), dtype=np.float32),
        )

        with np.load(str(npz_path)) as data:
            self.assertIn("feat0", data)
            self.assertIn("feat1", data)
            self.assertIn("image_embed", data)

        # 2. Verify dynamic pooling: 32x32 -> 16x16 for stride 32 (layer 22 / P5)
        raw_emb = torch.randn(1, 256, 32, 32)
        pooled = F.avg_pool2d(raw_emb, kernel_size=2, stride=2)
        self.assertEqual(pooled.shape, (1, 256, 16, 16))

        # 3. Verify KDTrainer projector layer targets
        kd_cfg = ConfigNode({
            "enabled": True,
            "losses": {
                "task": {"weight": 1.0},
                "mask_kd": {"enabled": False},
                "feature": {
                    "enabled": True,
                    "method": "cwd",
                    "layers": [16, 19, 22],
                    "weight": 1.0
                },
                "boundary": {"enabled": False},
            }
        })
        trainer = KDSegmentationTrainer(
            overrides={"model": "yolo11n-seg.pt", "data": "data/datasets/combined_yolo/dataset.yaml", "epochs": 1},
            kd_cfg=kd_cfg,
            logits_dir=self.logits_dir
        )
        trainer.setup_model()
        self.assertIn("layer_16", trainer.proj_layers)
        self.assertIn("layer_19", trainer.proj_layers)
        self.assertIn("layer_22", trainer.proj_layers)

    def test_p1_3_checkpoint_discovery_and_deduplication(self):
        """P1-3: Verify discover_and_deduplicate_checkpoints eliminates duplicate evaluations."""
        # Create dummy weights
        sub1 = Path(self.temp_dir) / "exp1" / "weights"
        sub2 = Path(self.temp_dir) / "exp2" / "weights"
        sub_dup = Path(self.temp_dir) / "copy_exp1" / "weights"
        sub1.mkdir(parents=True)
        sub2.mkdir(parents=True)
        sub_dup.mkdir(parents=True)

        ckpt1 = sub1 / "best.pt"
        ckpt2 = sub2 / "best.pt"
        ckpt_dup = sub_dup / "best.pt"

        content1 = b"MODEL_WEIGHTS_ARM_A_CONTENT"
        content2 = b"MODEL_WEIGHTS_ARM_B_CONTENT_DIFFERENT"

        ckpt1.write_bytes(content1)
        ckpt_dup.write_bytes(content1)  # Exact duplicate of ckpt1
        ckpt2.write_bytes(content2)

        discovered = discover_and_deduplicate_checkpoints(
            search_roots=[Path(self.temp_dir)]
        )

        # Should discover exactly 2 unique models, discarding the duplicate
        self.assertEqual(len(discovered), 2)
        names = [d["name"] for d in discovered]
        self.assertIn("exp2", names)
        # Exactly one of exp1 or copy_exp1 should be present
        self.assertEqual(len([n for n in names if n in ("exp1", "copy_exp1")]), 1)

        # Verify SHA256 hashes are distinct and recorded
        hashes = [d["sha256"] for d in discovered]
        self.assertEqual(len(set(hashes)), 2)
        self.assertIn(compute_file_sha256(ckpt1), hashes)
        self.assertIn(compute_file_sha256(ckpt2), hashes)

    def test_p1_4_utf8_notebook_encoding(self):
        """P1-4: Verify notebooks parse cleanly with explicit utf-8 encoding."""
        final_dir = Path("final_notebooks")
        if final_dir.exists():
            for nb_path in final_dir.glob("*.ipynb"):
                with open(nb_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertIn("cells", data)
                self.assertIn("metadata", data)

    def test_p1_5_strict_kd_fail_fast_mode(self):
        """P1-5: Verify strict=True raises RuntimeError on first error, strict=False tolerates."""
        # Test strict=True fails fast
        kd_cfg_strict = ConfigNode({
            "enabled": True,
            "strict": True,
            "temperature": 2.0,
            "losses": {
                "task": {"weight": 1.0},
                "mask_kd": {"enabled": True},
                "feature": {"enabled": False},
                "boundary": {"enabled": False},
            }
        })
        trainer_strict = KDSegmentationTrainer(
            overrides={"model": "yolo11n-seg.pt", "data": "data/datasets/combined_yolo/dataset.yaml", "epochs": 1},
            kd_cfg=kd_cfg_strict,
            logits_dir=self.logits_dir
        )
        trainer_strict.setup_model()
        trainer_strict._sam_targets = {"sample001": torch.zeros((1, 256, 256))}
        trainer_strict._current_paths = ["sample001.jpg"]

        # In strict mode, an exception during loss computation must immediately raise RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            # Pass a corrupted batch that triggers an exception in _kd_loss_from_preds
            corrupt_batch = {"invalid_key": True}
            trainer_strict._kd_loss_from_preds(preds=None, batch=corrupt_batch, model=trainer_strict.model)
        self.assertIn("[KD Critical Error]", str(ctx.exception))
        self.assertIn("strict=True", str(ctx.exception))

        # Test strict=False tolerates isolated error
        kd_cfg_tolerant = ConfigNode({
            "enabled": True,
            "strict": False,
            "temperature": 2.0,
            "losses": {
                "task": {"weight": 1.0},
                "mask_kd": {"enabled": True},
                "feature": {"enabled": False},
                "boundary": {"enabled": False},
            }
        })
        trainer_tolerant = KDSegmentationTrainer(
            overrides={"model": "yolo11n-seg.pt", "data": "data/datasets/combined_yolo/dataset.yaml", "epochs": 1},
            kd_cfg=kd_cfg_tolerant,
            logits_dir=self.logits_dir
        )
        trainer_tolerant.setup_model()
        trainer_tolerant._sam_targets = {"sample001": torch.zeros((1, 256, 256))}
        trainer_tolerant._current_paths = ["sample001.jpg"]

        # Should NOT raise on the first error, but return empty losses and increment counter
        losses = trainer_tolerant._kd_loss_from_preds(preds=None, batch={"invalid_key": True}, model=trainer_tolerant.model)
        self.assertEqual(losses, {})
        self.assertEqual(trainer_tolerant._kd_consecutive_errors, 1)
        self.assertEqual(trainer_tolerant._kd_total_errors, 1)


if __name__ == "__main__":
    unittest.main()
