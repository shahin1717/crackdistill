#!/usr/bin/env python3
"""
Unit tests for final_notebooks/ suite generation and integrity.
Verifies that all notebooks are valid JSON, contain required audit fixes,
deterministic checkpoint handlers, and appropriate distillation configurations.
"""

import json
import unittest
from pathlib import Path


class TestNotebookSuite(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent
        self.final_dir = self.repo_root / "final_notebooks"

    def test_all_notebooks_valid_json(self):
        notebooks = list(self.final_dir.glob("*.ipynb"))
        self.assertGreaterEqual(len(notebooks), 10, "Expected at least 10 notebooks in final_notebooks")
        for nb_path in notebooks:
            with open(nb_path) as f:
                data = json.load(f)
            self.assertIn("cells", data, f"{nb_path.name} missing 'cells'")
            self.assertIn("metadata", data, f"{nb_path.name} missing 'metadata'")
            self.assertGreater(len(data["cells"]), 0, f"{nb_path.name} has no cells")

    def test_baseline_notebook(self):
        nb_path = self.final_dir / "00_run_baseline_clean_seed42.ipynb"
        self.assertTrue(nb_path.exists(), "00_run_baseline_clean_seed42.ipynb must exist")
        with open(nb_path) as f:
            data = json.load(f)
        full_text = "".join("".join(c.get("source", [])) for c in data["cells"])
        self.assertIn("'distillation.enabled': False", full_text)
        self.assertIn("resolve_checkpoint", full_text)
        self.assertIn("get_checkpoint_manifest", full_text)

    def test_full_kd_box_notebook(self):
        nb_path = self.final_dir / "01_run_full_kd_box_seed42.ipynb"
        self.assertTrue(nb_path.exists(), "01_run_full_kd_box_seed42.ipynb must exist")
        with open(nb_path) as f:
            data = json.load(f)
        full_text = "".join("".join(c.get("source", [])) for c in data["cells"])
        self.assertIn("'distillation.enabled': True", full_text)
        self.assertIn("'distillation.losses.mask_kd.enabled': True", full_text)
        self.assertIn("'distillation.losses.feature.enabled': True", full_text)
        self.assertIn("'distillation.losses.boundary.enabled': True", full_text)
        self.assertIn("[16, 19, 22]", full_text)
        self.assertIn("teacher_logits_box", full_text)
        self.assertIn("resolve_checkpoint", full_text)

    def test_full_kd_centroid_notebook(self):
        nb_path = self.final_dir / "01b_run_full_kd_centroid_seed42.ipynb"
        self.assertTrue(nb_path.exists(), "01b_run_full_kd_centroid_seed42.ipynb must exist")
        with open(nb_path) as f:
            data = json.load(f)
        full_text = "".join("".join(c.get("source", [])) for c in data["cells"])
        self.assertIn("'distillation.enabled': True", full_text)
        self.assertIn("teacher_logits_centroid", full_text)
        self.assertIn("[16, 19, 22]", full_text)


if __name__ == "__main__":
    unittest.main()
