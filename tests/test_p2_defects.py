"""Unit test suite verifying fixes for Category P2 defects:
- P2-1: Zero Post-Remediation GPU Experiments (run_confirmatory_experiments harness & smoke test)
- P2-2: Missing GT-Soft Supervision Control Arm (generate_gt_soft_logits Gaussian softening & inverse sigmoid)
- P2-3: Untouched Canonical Test Set (evaluate_canonical_test_set split loading & metric calculation)
- P2-4: Single-Seed Evidence & Lack of Variance Bounds (stats_analyzer CI, paired t-test, Cohen's d)
- P2-5: Serial Tiling Latency Mismatch (benchmark_tiled_inference_pipeline two-level disambiguation)
"""

import unittest
import numpy as np
import torch
import tempfile
import json
from pathlib import Path
import cv2

# P2-4 Stats Analyzer
from utils.stats_analyzer import compute_stats, compare_two_arms, format_comparison_table

# P2-2 GT-Soft Pseudo-Logit Generator
from scripts.generate_gt_soft_logits import generate_soft_logit_from_mask

# P2-3 Canonical Test Evaluator
from scripts.evaluate_canonical_test_set import compute_dice_score, load_canonical_crops

# P2-5 Tiled Inference Pipeline Benchmark
from inference.tiled_inference import benchmark_tiled_inference_pipeline

# P2-1 Confirmatory Runner
from scripts.run_confirmatory_experiments import build_experiment_command, compute_sha256

# P2-4 Multi-seed Aggregator
from scripts.aggregate_multiseed_results import aggregate_results


class TestP24StatsAnalyzer(unittest.TestCase):
    """Test suite for P2-4 multi-seed statistical engine."""

    def test_compute_stats_known_values(self):
        vals = [0.50, 0.52, 0.51, 0.53, 0.54]
        res = compute_stats(vals)
        self.assertEqual(res["n"], 5)
        self.assertAlmostEqual(res["mean"], 0.52, places=4)
        self.assertAlmostEqual(res["std"], float(np.std(vals, ddof=1)), places=4)
        self.assertTrue(res["ci_lower"] < res["mean"] < res["ci_upper"])

    def test_compute_stats_single_value(self):
        res = compute_stats([0.55])
        self.assertEqual(res["n"], 1)
        self.assertEqual(res["mean"], 0.55)
        self.assertEqual(res["std"], 0.0)
        self.assertEqual(res["ci_lower"], 0.55)
        self.assertEqual(res["ci_upper"], 0.55)

    def test_compare_two_arms_paired(self):
        baseline = [0.50, 0.51, 0.49, 0.52, 0.50]
        kd_arm = [0.54, 0.56, 0.53, 0.55, 0.54]
        comp = compare_two_arms(baseline, kd_arm, paired=True)
        self.assertAlmostEqual(comp["diff_mean"], 0.04, places=3)
        self.assertLess(comp["t_pvalue"], 0.05)
        self.assertGreater(comp["cohens_d"], 2.0)
        self.assertIn("ci_lower", comp)
        self.assertIn("ci_upper", comp)

    def test_format_comparison_table(self):
        baseline = [0.50, 0.51, 0.49]
        kd_arm = [0.54, 0.55, 0.53]
        table = format_comparison_table(baseline, kd_arm, arm_a_name="Clean Baseline", arm_b_name="Mask-KD")
        self.assertIn("Clean Baseline", table)
        self.assertIn("Mask-KD", table)
        self.assertIn("Cohen's d", table)
        self.assertIn("95% CI", table)


class TestP22GTSoftPseudoLogits(unittest.TestCase):
    """Test suite for P2-2 GT-Soft pseudo-logit generator."""

    def test_generate_soft_logit_from_mask(self):
        # Create a synthetic 100x100 binary mask with a crack line down the center
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:, 48:52] = 255  # 4-pixel wide vertical crack

        logits = generate_soft_logit_from_mask(mask, sigma=2.0)
        self.assertEqual(logits.shape, (100, 100))
        self.assertEqual(logits.dtype, np.float32)

        # Foreground crack center should have positive logits
        fg_center_logit = logits[50, 50]
        # Background far from crack should have negative logits
        bg_logit = logits[50, 10]

        self.assertGreater(fg_center_logit, 0.0)
        self.assertLess(bg_logit, 0.0)
        self.assertGreater(fg_center_logit, bg_logit)

    def test_generate_soft_logit_empty_mask(self):
        # Test completely empty mask
        mask = np.zeros((64, 64), dtype=np.uint8)
        logits = generate_soft_logit_from_mask(mask, sigma=2.0)
        self.assertEqual(logits.shape, (64, 64))
        # All logits should be strongly negative (background)
        self.assertTrue(np.all(logits < -2.0))


class TestP23CanonicalTestSet(unittest.TestCase):
    """Test suite for P2-3 canonical test set evaluation."""

    def test_compute_dice_score(self):
        # Identical masks
        m1 = np.ones((50, 50), dtype=np.uint8)
        self.assertAlmostEqual(compute_dice_score(m1, m1), 1.0)

        # Disjoint masks
        m_a = np.zeros((50, 50), dtype=np.uint8)
        m_a[:25, :] = 1
        m_b = np.zeros((50, 50), dtype=np.uint8)
        m_b[25:, :] = 1
        self.assertAlmostEqual(compute_dice_score(m_a, m_b), 0.0)

        # Two empty masks
        empty = np.zeros((50, 50), dtype=np.uint8)
        self.assertAlmostEqual(compute_dice_score(empty, empty), 1.0)

    def test_load_canonical_crops(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / "images" / "test"
            labels_dir = tmp_path / "labels" / "test"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)

            # Create dummy files
            img_file = images_dir / "sample_001.png"
            lbl_file = labels_dir / "sample_001.txt"
            cv2.imwrite(str(img_file), np.zeros((100, 100, 3), dtype=np.uint8))
            lbl_file.write_text("0 0.5 0.5 0.2 0.2\n")

            crops = load_canonical_crops(tmp_path)
            self.assertEqual(len(crops), 1)
            self.assertEqual(crops[0]["stem"], "sample_001")


class TestP25TiledInferencePipeline(unittest.TestCase):
    """Test suite for P2-5 serial vs batched tiling benchmark and disambiguation."""

    class MockYOLO:
        """Mock YOLO model returning empty segmentation result."""
        def __init__(self):
            self.names = {0: "crack"}

        def __call__(self, x, verbose=False, **kwargs):
            return self.predict(x, verbose=verbose, **kwargs)

        def predict(self, x, imgsz=512, conf=0.25, verbose=False, **kwargs):
            class MockBoxResult:
                def __init__(self):
                    self.masks = None
                    self.boxes = None
            if isinstance(x, list):
                return [MockBoxResult() for _ in x]
            return [MockBoxResult()]

    def test_benchmark_tiled_inference_pipeline(self):
        mock_model = self.MockYOLO()
        res = benchmark_tiled_inference_pipeline(
            model=mock_model,
            image_shape=(600, 800, 3),
            tile_size=512,
            overlap=0.2,
            num_runs=3,
            device="cpu"
        )
        self.assertIn("level1_single_tile", res)
        self.assertIn("pipeline_serial", res)
        self.assertIn("pipeline_batched", res)
        self.assertIn("num_tiles", res)
        self.assertIn("critical_disambiguation_note", res)

        # Pipeline serial latency must be strictly greater than single-tile latency
        self.assertGreater(res["pipeline_serial"]["mean_ms"], res["level1_single_tile"]["mean_ms"])
        # Serial latency must be >= batched latency
        self.assertGreaterEqual(res["pipeline_serial"]["mean_ms"], res["pipeline_batched"]["mean_ms"])


class TestP21ConfirmatoryExperiments(unittest.TestCase):
    """Test suite for P2-1 confirmatory experiment runner."""

    def test_build_experiment_command_smoke_test(self):
        cmd = build_experiment_command(
            variant_name="baseline_clean",
            seed=42,
            is_smoke_test=True,
            epochs=150,
            batch_size=8,
            device="cpu"
        )
        cmd_str = " ".join(cmd)
        self.assertIn("train.py", cmd_str)
        self.assertIn("train.epochs=1", cmd_str)
        self.assertIn("data.batch_size=2", cmd_str)
        self.assertIn("device=cpu", cmd_str)

    def test_compute_sha256(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"crackdistill_test_hash")
            f.flush()
            h = compute_sha256(Path(f.name))
            self.assertEqual(len(h), 64)


class TestP24MultiSeedAggregator(unittest.TestCase):
    """Test suite for multi-seed result aggregator."""

    def test_aggregate_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create two seed runs for baseline and two for mask_kd
            r1 = {"metrics/mask_mAP50": 0.50, "metrics/mask_mAP50-95": 0.20, "metrics/box_mAP50": 0.55}
            r2 = {"metrics/mask_mAP50": 0.52, "metrics/mask_mAP50-95": 0.22, "metrics/box_mAP50": 0.57}
            (tmp_path / "baseline_clean_seed42.json").write_text(json.dumps(r1))
            (tmp_path / "baseline_clean_seed123.json").write_text(json.dumps(r2))

            kd1 = {"metrics/mask_mAP50": 0.54, "metrics/mask_mAP50-95": 0.24, "metrics/box_mAP50": 0.59}
            kd2 = {"metrics/mask_mAP50": 0.56, "metrics/mask_mAP50-95": 0.26, "metrics/box_mAP50": 0.61}
            (tmp_path / "mask_kd_seed42.json").write_text(json.dumps(kd1))
            (tmp_path / "mask_kd_seed123.json").write_text(json.dumps(kd2))

            summary = aggregate_results(tmp_path)
            self.assertIn("baseline_clean", summary)
            self.assertIn("mask_kd", summary)
            self.assertEqual(len(summary["baseline_clean"]), 2)
            self.assertAlmostEqual(float(np.mean(summary["baseline_clean"])), 0.51, places=4)


if __name__ == "__main__":
    unittest.main()
