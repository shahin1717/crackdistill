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

    def _run_step1(self, notebook, fake_input: Path, cwd: Path):
        """Execute the notebook's Step 1 cell against a fake /kaggle/input tree."""
        import json as _json
        import os
        cells = _json.load(open(self.final_dir / notebook, encoding="utf-8"))["cells"]
        src = next("".join(c["source"]) for c in cells if "".join(c["source"]).startswith("# ── Step 1"))
        src = src.replace('Path("/kaggle/input/distill_datasetforme")', f'Path("{fake_input}/missing")')
        src = src.replace('Path("/kaggle/input")', f'Path("{fake_input}")')
        old = os.getcwd()
        os.chdir(cwd)
        try:
            exec(compile(src, notebook, "exec"), {})
        finally:
            os.chdir(old)

    def test_step1_links_raw_crack500_not_other_datasets(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input" / "datasets" / "owner" / "distill-datasetforme"
            for decoy in ("combined_yolo", "crack500_yolo_augmented", "deepcrack_yolo"):
                (root / "datasets" / decoy / "images").mkdir(parents=True)
                (root / "datasets" / decoy / "dataset.yaml").write_text("path: .\n")
            (root / "datasets" / "crack500" / "traincrop").mkdir(parents=True)
            (root / "teacher_logits").mkdir(parents=True)
            work = Path(tmp) / "work"
            work.mkdir()
            self._run_step1(f"2_mask_kd_seed{self.seeds['mask_kd'][0]}.ipynb", Path(tmp) / "input", work)
            self.assertEqual((work / "data/datasets/crack500").resolve(), (root / "datasets" / "crack500").resolve())
            self.assertFalse((work / "data/datasets/crack500_yolo").exists())
            self.assertEqual((work / "data/teacher_logits_box").resolve(), (root / "teacher_logits").resolve())

    def test_step1_converted_fallback_only_accepts_crack500_yolo(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input" / "ds"
            for name in ("combined_yolo", "crack500_yolo"):
                (root / name / "images").mkdir(parents=True)
                (root / name / "dataset.yaml").write_text("path: .\n")
            work = Path(tmp) / "work"
            work.mkdir()
            self._run_step1(f"1_baseline_seed{self.seeds['baseline'][0]}.ipynb", Path(tmp) / "input", work)
            self.assertEqual((work / "data/datasets/crack500_yolo").resolve(), (root / "crack500_yolo").resolve())

    def test_gt_soft_never_uses_sam_logits(self):
        text = notebook_text(self.final_dir / f"3_gt_soft_seed{self.seeds['gt_soft'][0]}.ipynb")
        self.assertIn('overrides["teacher.logits_dir"] = "data/teacher_logits_gt_soft/"', text)
        self.assertIn("%%writefile scripts/generate_gt_soft_logits.py", text)
        self.assertIn("scripts/generate_gt_soft_logits.py --data data/datasets/crack500_yolo --out data/teacher_logits_gt_soft", text)
        self.assertNotIn("os.symlink(src_f", text)                 # no teacher-logit linking at all
        self.assertNotIn("generate_teacher_logits.py --prompt-type", text)  # never falls back to SAM 2


if __name__ == "__main__":
    unittest.main()
