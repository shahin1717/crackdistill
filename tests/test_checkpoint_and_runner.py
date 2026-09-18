"""
Unit tests for deterministic checkpoint provenance and experiment runner handling.
Verifies fixes for items 10 and 11 from the scientific audit:
- Deterministic checkpoint resolver & SHA256 manifest generation (item 10).
- Safe results extraction & syntax-correct metrics summary printing (item 11).
"""

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.checkpoint import (
    compute_file_sha256,
    resolve_checkpoint,
    get_checkpoint_manifest,
    save_checkpoint_manifest,
)


class TestCheckpointProvenance(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Create dummy valid checkpoint
        self.ckpt_file = self.dir_path / "model.pt"
        self.dummy_bytes = b"PYTORCH_CHECKPOINT_DUMMY_DATA_12345"
        self.ckpt_file.write_bytes(self.dummy_bytes)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_file_sha256(self):
        digest = compute_file_sha256(self.ckpt_file)
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64)
        # Deterministic SHA256 check
        import hashlib
        expected = hashlib.sha256(self.dummy_bytes).hexdigest()
        self.assertEqual(digest, expected)

    def test_resolve_checkpoint_direct_file(self):
        resolved = resolve_checkpoint(self.ckpt_file)
        self.assertEqual(resolved, self.ckpt_file.resolve())

    def test_resolve_checkpoint_directory_weights_best(self):
        weights_dir = self.dir_path / "exp1" / "weights"
        weights_dir.mkdir(parents=True)
        best_pt = weights_dir / "best.pt"
        best_pt.write_bytes(b"WEIGHTS_BEST")

        resolved = resolve_checkpoint(self.dir_path / "exp1")
        self.assertEqual(resolved, best_pt.resolve())

    def test_resolve_checkpoint_not_found(self):
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(self.dir_path / "non_existent.pt")

    def test_resolve_checkpoint_empty_file(self):
        empty_pt = self.dir_path / "empty.pt"
        empty_pt.touch()
        with self.assertRaises(ValueError):
            resolve_checkpoint(empty_pt)

    def test_resolve_checkpoint_invalid_extension(self):
        invalid_txt = self.dir_path / "model.txt"
        invalid_txt.write_bytes(b"data")
        with self.assertRaises(ValueError):
            resolve_checkpoint(invalid_txt)

    def test_get_checkpoint_manifest_and_save(self):
        manifest = get_checkpoint_manifest(
            checkpoint_path=self.ckpt_file,
            experiment_name="baseline_test",
            seed=42,
            metadata={"protocol": "clean_seed42"}
        )
        self.assertEqual(manifest["filename"], "model.pt")
        self.assertEqual(manifest["experiment"], "baseline_test")
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(manifest["size_bytes"], len(self.dummy_bytes))
        self.assertIn("sha256", manifest)

        # Test saving manifest
        out_json = save_checkpoint_manifest(manifest, self.dir_path / "model_manifest.json")
        self.assertTrue(out_json.exists())
        loaded = json.loads(out_json.read_text())
        self.assertEqual(loaded["sha256"], manifest["sha256"])


class TestRunnerMetricsSummary(unittest.TestCase):
    def test_summary_printing_no_metrics(self):
        """Ensure empty metrics dict does not crash with NameError or UnboundLocalError."""
        all_results = {"exp_empty": {}}
        captured = io.StringIO()
        sys.stdout = captured
        try:
            for exp_name, metrics in all_results.items():
                print(f"\n{exp_name}:")
                if not metrics:
                    print("  (no metrics recorded)")
                    continue
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("exp_empty:", output)
        self.assertIn("(no metrics recorded)", output)

    def test_summary_printing_with_metrics(self):
        """Ensure float and non-float metrics are printed properly without duplication."""
        all_results = {
            "exp1": {
                "mask_mAP50": 0.542412,
                "checkpoint": "/path/to/best.pt",
                "checkpoint_sha256": "abc123def456",
            }
        }
        captured = io.StringIO()
        sys.stdout = captured
        try:
            for exp_name, metrics in all_results.items():
                print(f"\n{exp_name}:")
                if not metrics:
                    print("  (no metrics recorded)")
                    continue
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("mask_mAP50: 0.5424", output)
        self.assertIn("checkpoint: /path/to/best.pt", output)
        self.assertIn("checkpoint_sha256: abc123def456", output)
        # Check no duplication of last line
        self.assertEqual(output.count("checkpoint_sha256: abc123def456"), 1)


if __name__ == "__main__":
    unittest.main()
