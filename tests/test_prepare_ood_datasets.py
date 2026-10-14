"""Converters for the zero-shot crack datasets (layouts and mask rules from docs/DATASETS.md)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.prepare_ood_datasets import prepare


def write_img(path, size=(48, 32)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), 90, np.uint8)).save(path)


def crack_mask(size=(48, 32)):
    m = np.zeros((size[1], size[0]), bool)
    m[10:13, 5:40] = True
    return m


class TestPrepareOOD(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "raw"
        self.out = Path(self.tmp.name) / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, n, expected_mask):
        masks = sorted((self.out / "masks").glob("*.png"))
        imgs = sorted((self.out / "images").glob("*.png"))
        self.assertEqual(len(masks), n)
        self.assertEqual([p.name for p in imgs], [p.name for p in masks])
        m = np.array(Image.open(masks[0]))
        self.assertEqual(set(np.unique(m)) - {0, 255}, set())
        self.assertTrue(np.array_equal(m == 255, expected_mask))
        meta = json.loads((self.out / "meta.json").read_text())
        self.assertEqual(meta["n_pairs"], n)

    def test_cfd_mat_uses_label_2_only(self):
        import scipy.io as sio
        seg = np.ones((32, 48), np.uint8)
        seg[crack_mask()] = 2
        seg[20:30, 0:10] = 3                         # enclosed pavement block, not crack
        write_img(self.src / "image" / "001.jpg")
        write_img(self.src / "image" / "201.jpg")    # unannotated extra -> skipped
        (self.src / "groundTruth").mkdir(parents=True)
        sio.savemat(self.src / "groundTruth" / "001.mat", {"groundTruth": {"Segmentation": seg, "Boundaries": seg * 0}})
        prepare("cfd", self.src, self.out)
        self.check(1, crack_mask())

    def test_gaps384_mode1_png(self):
        write_img(self.src / "croppedimg" / "train_0001_1_1.jpg")
        (self.src / "croppedgt").mkdir(parents=True)
        Image.fromarray(crack_mask()).convert("1").save(self.src / "croppedgt" / "train_0001_1_1.png")
        prepare("gaps384", self.src, self.out)
        self.check(1, crack_mask())

    def test_ael_inverted_and_renamed(self):
        write_img(self.src / "img" / "AIGLE_RN" / "Im_GT_AIGLE_RN_C18aor.jpg")
        gt = self.src / "gt" / "AIGLE_RN" / "GT_AIGLE_RN_C18a.png"
        gt.parent.mkdir(parents=True)
        Image.fromarray(np.where(crack_mask(), 0, 255).astype(np.uint8)).save(gt)   # crack = black
        prepare("ael", self.src, self.out)
        self.check(1, crack_mask())

    def test_deepcrack_test_split(self):
        write_img(self.src / "test_img" / "a.jpg")
        write_img(self.src / "train_img" / "b.jpg")  # train split is not used for zero-shot testing
        (self.src / "test_lab").mkdir(parents=True)
        Image.fromarray(np.where(crack_mask(), 255, 0).astype(np.uint8)).save(self.src / "test_lab" / "a.png")
        prepare("deepcrack", self.src, self.out)
        self.check(1, crack_mask())

    def test_thin_label_set_auto_polarity(self):
        write_img(self.src / "image" / "x.jpg")
        (self.src / "gt").mkdir(parents=True)
        Image.fromarray(np.where(crack_mask(), 0, 255).astype(np.uint8)).save(self.src / "gt" / "x.bmp")
        prepare("crkwh100", self.src, self.out)
        self.check(1, crack_mask())
        self.assertTrue(json.loads((self.out / "meta.json").read_text())["thin_labels"])

    def test_size_mismatch_is_an_error(self):
        write_img(self.src / "test_img" / "a.jpg", size=(48, 32))
        (self.src / "test_lab").mkdir(parents=True)
        Image.fromarray(np.zeros((10, 10), np.uint8)).save(self.src / "test_lab" / "a.png")
        with self.assertRaises(ValueError):
            prepare("deepcrack", self.src, self.out)


if __name__ == "__main__":
    unittest.main()
