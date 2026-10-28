#!/usr/bin/env python3
"""
Unit tests for final_notebooks/ suite generation and integrity.
Verifies that all notebooks are valid JSON, pin the verified Ultralytics version, pair arms by seed,
and that the GT-soft control can never train on SAM 2 logits.
"""

import json
import unittest
from pathlib import Path

ARMS = ("A1_baseline_d4", "A2_mask_kd_d4", "A3_gt_soft_d4", "A4_baseline_ultra_aug", "A5_mask_kd_gated_d4")


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
        notebooks = list(self.final_dir.glob("[ABEI][0-9]_*.ipynb"))  # builder-generated suite only
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
        text = notebook_text(self.final_dir / f"A1_baseline_d4_seed{self.seeds['baseline_d4'][0]}.ipynb")
        self.assertIn("'distillation.enabled': False", text)
        self.assertIn("'train.d4_aug': True", text)
        self.assertNotIn('overrides["teacher.logits_dir"]', text)

    def test_gated_kd_differs_from_a2_only_in_the_gate(self):
        a2 = notebook_text(self.final_dir / "A2_mask_kd_d4_seed0.ipynb")
        a5 = notebook_text(self.final_dir / "A5_mask_kd_gated_d4_seed0.ipynb")
        self.assertIn("'distillation.losses.mask_kd.box_crop': True", a5)
        self.assertIn("'distillation.losses.mask_kd.gt_gate': True", a5)
        self.assertNotIn("'distillation.losses.mask_kd.box_crop'", a2)
        self.assertIn("'train.d4_aug': True", a5)

    def test_reference_baseline_uses_ultralytics_default_aug(self):
        text = notebook_text(self.final_dir / f"A4_baseline_ultra_aug_seed{self.seeds['baseline_ultra_aug'][0]}.ipynb")
        self.assertIn("'distillation.enabled': False", text)
        self.assertIn("'train.allow_spatial_aug': True", text)
        self.assertNotIn("'train.d4_aug': True", text)

    def test_mask_kd_and_gt_soft_share_the_loss_config(self):
        def overrides_block(text):
            start = text.index("overrides = {")
            return text[start:text.index("}", start)]

        kd = notebook_text(self.final_dir / f"A2_mask_kd_d4_seed{self.seeds['mask_kd_d4'][0]}.ipynb")
        gt = notebook_text(self.final_dir / f"A3_gt_soft_d4_seed{self.seeds['gt_soft_d4'][0]}.ipynb")
        self.assertEqual(overrides_block(kd), overrides_block(gt))
        self.assertIn("'distillation.losses.feature.enabled': False", kd)
        self.assertIn("'distillation.progressive.enabled': False", kd)
        self.assertIn('overrides["teacher.logits_dir"] = "data/teacher_logits_box/"', kd)
        self.assertIn("'train.d4_aug': True", kd)

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
            self._run_step1(f"A2_mask_kd_d4_seed{self.seeds['mask_kd_d4'][0]}.ipynb", Path(tmp) / "input", work)
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
            self._run_step1(f"A1_baseline_d4_seed{self.seeds['baseline_d4'][0]}.ipynb", Path(tmp) / "input", work)
            self.assertEqual((work / "data/datasets/crack500_yolo").resolve(), (root / "crack500_yolo").resolve())

    def test_efficiency_pairs_share_one_subset_and_their_full_data_recipe(self):
        def overrides_block(text):
            start = text.index("overrides = {")
            return text[start:text.index("}", start)]

        for pct, base, kd in (("25", "E1_baseline_d4_p25", "E2_mask_kd_gated_d4_p25"),
                              ("50", "E3_baseline_d4_p50", "E4_mask_kd_gated_d4_p50")):
            b = notebook_text(self.final_dir / f"{base}_seed0.ipynb")
            k = notebook_text(self.final_dir / f"{kd}_seed0.ipynb")
            for text in (b, k):
                self.assertIn(f"dataset_photos{pct}.yaml", text)
                self.assertIn("%%writefile scripts/make_train_subset.py", text)
            subset_line = next(l for l in b.splitlines() if l.startswith("assert (subset["))
            self.assertIn(subset_line, k)
            self.assertEqual(overrides_block(b), overrides_block(notebook_text(self.final_dir / "A1_baseline_d4_seed0.ipynb")))
            self.assertEqual(overrides_block(k), overrides_block(notebook_text(self.final_dir / "A5_mask_kd_gated_d4_seed0.ipynb")))

    def test_full_data_notebooks_have_no_subset(self):
        for prefix in ARMS:
            self.assertNotIn("dataset_photos", notebook_text(self.final_dir / f"{prefix}_seed0.ipynb"))

    def test_gt_soft_never_uses_sam_logits(self):
        text = notebook_text(self.final_dir / f"A3_gt_soft_d4_seed{self.seeds['gt_soft_d4'][0]}.ipynb")
        self.assertIn('overrides["teacher.logits_dir"] = "data/teacher_logits_gt_soft/"', text)
        self.assertIn("%%writefile scripts/generate_gt_soft_logits.py", text)
        self.assertIn("scripts/generate_gt_soft_logits.py --data data/datasets/crack500_yolo --out data/teacher_logits_gt_soft", text)
        self.assertNotIn("os.symlink(src_f", text)                 # no teacher-logit linking at all
        self.assertNotIn("generate_teacher_logits.py --prompt-type", text)  # never falls back to SAM 2


if __name__ == "__main__":
    unittest.main()
