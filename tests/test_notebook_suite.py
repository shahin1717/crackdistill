#!/usr/bin/env python3
"""
Unit tests for final_notebooks/ suite generation and integrity.
Verifies that all notebooks are valid JSON, pin the verified Ultralytics version, pair arms by seed,
and that the GT-soft control can never train on SAM 2 logits.
"""

import json
import unittest
from pathlib import Path

ARMS = ("1_baseline", "2_mask_kd", "3_gt_soft")


def notebook_text(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return "".join("".join(c.get("source", [])) for c in data["cells"])


class TestNotebookSuite(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent
        self.final_dir = self.repo_root / "final_notebooks"
        from scripts.build_final_notebooks import SEEDS
        self.seeds = SEEDS

    def test_all_notebooks_valid_json(self):
        notebooks = list(self.final_dir.glob("[1-9]_*.ipynb"))  # builder-generated suite only
        self.assertEqual(len(notebooks), sum(len(v) for v in self.seeds.values()) + 1)
        for nb_path in notebooks:
            with open(nb_path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("cells", data, f"{nb_path.name} missing 'cells'")
            self.assertGreater(len(data["cells"]), 0, f"{nb_path.name} has no cells")
            self.assertIn("ultralytics==8.4.60", notebook_text(nb_path), f"{nb_path.name} must pin Ultralytics")

    def test_every_arm_has_every_seed(self):
        for prefix in ARMS:
            arm = prefix.split("_", 1)[1]
            for seed in self.seeds[arm]:
                text = notebook_text(self.final_dir / f"{prefix}_seed{seed}.ipynb")
                self.assertIn(f'"arm": "{arm}"', text)
                self.assertIn(f'overrides["project.seed"] = {seed}', text)
                self.assertIn(f'"seed": {seed},', text)
                self.assertIn('"arm": ', text)
                self.assertIn("resolve_checkpoint", text)

    def test_baseline_notebook(self):
        text = notebook_text(self.final_dir / f"1_baseline_seed{self.seeds['baseline'][0]}.ipynb")
        self.assertIn("'distillation.enabled': False", text)
        self.assertNotIn('overrides["teacher.logits_dir"]', text)

    def test_mask_kd_and_gt_soft_share_the_loss_config(self):
        def overrides_block(text):
            start = text.index("overrides = {")
            return text[start:text.index("}", start)]

        kd = notebook_text(self.final_dir / f"2_mask_kd_seed{self.seeds['mask_kd'][0]}.ipynb")
        gt = notebook_text(self.final_dir / f"3_gt_soft_seed{self.seeds['gt_soft'][0]}.ipynb")
        self.assertEqual(overrides_block(kd), overrides_block(gt))
        self.assertIn("'distillation.losses.feature.enabled': False", kd)
        self.assertIn("'distillation.progressive.enabled': False", kd)
        self.assertIn('overrides["teacher.logits_dir"] = "data/teacher_logits_box/"', kd)

    def test_gt_soft_never_uses_sam_logits(self):
        text = notebook_text(self.final_dir / f"3_gt_soft_seed{self.seeds['gt_soft'][0]}.ipynb")
        self.assertIn('overrides["teacher.logits_dir"] = "data/teacher_logits_gt_soft/"', text)
        self.assertIn("%%writefile scripts/generate_gt_soft_logits.py", text)
        self.assertIn("scripts/generate_gt_soft_logits.py --data data/datasets/crack500_yolo --out data/teacher_logits_gt_soft", text)
        self.assertNotIn("os.symlink(src_f", text)                 # no teacher-logit linking at all
        self.assertNotIn("generate_teacher_logits.py --prompt-type", text)  # never falls back to SAM 2


if __name__ == "__main__":
    unittest.main()
