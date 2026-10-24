"""Photo-level training subsets for the data-efficiency curve (whole parent photos, nested, fixed seed)."""

import sys
import tempfile
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.make_train_subset import parent_photo, select_photos, write_subset


def fake_dataset(root, photos=8, crops_per_photo=3):
    for p in range(photos):
        for c in range(crops_per_photo):
            stem = f"2016022{p}_0809{p:02d}_{c * 640 + 1}_361"
            (root / "images" / "train").mkdir(parents=True, exist_ok=True)
            (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
            (root / "images" / "train" / f"{stem}.jpg").write_bytes(b"")
            (root / "labels" / "train" / f"{stem}.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n")


class TestTrainSubset(unittest.TestCase):
    def test_parent_photo(self):
        self.assertEqual(parent_photo("20160222_080933_361_1281"), "20160222_080933")

    def test_nested_and_deterministic(self):
        stems = [f"p{p}_x_{c}_1" for p in range(20) for c in range(3)]
        q, h = select_photos(stems, 0.25, seed=0), select_photos(stems, 0.5, seed=0)
        self.assertEqual(len(q), 5)
        self.assertEqual(len(h), 10)
        self.assertTrue(q <= h)
        self.assertEqual(q, select_photos(stems, 0.25, seed=0))

    def test_write_subset_keeps_whole_photos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_dataset(root)
            info = write_subset(root, 0.5, seed=0)
            self.assertEqual((info["photos"], info["crops"]), (4, 12))
            listed = Path(root / "train_photos50.txt").read_text().split()
            self.assertEqual(len(listed), 12)
            self.assertTrue(all(Path(p).is_absolute() and Path(p).exists() for p in listed))
            self.assertEqual(len({parent_photo(Path(p).stem) for p in listed}), 4)
            yaml_text = Path(info["yaml"]).read_text()
            self.assertIn("train: train_photos50.txt", yaml_text)
            self.assertIn("val: images/val", yaml_text)


if __name__ == "__main__":
    unittest.main()
