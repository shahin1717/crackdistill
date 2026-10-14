"""
Augmentation-consistent KD: flips + 90-degree rotations (the dihedral group D4) applied identically to the
letterboxed image, GT masks/boxes and the warped teacher map. The canvas is square, so every element is exact.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from distillation.kd_trainer import d4_apply, d4_boxes, D4AugDataset


def box_mask(xywh, size):
    """Rasterize one normalized xywh box into a size x size mask."""
    x, y, w, h = xywh
    m = torch.zeros(size, size)
    m[round((y - h / 2) * size):round((y + h / 2) * size), round((x - w / 2) * size):round((x + w / 2) * size)] = 1
    return m


def mask_box(m):
    """Normalized xywh of the nonzero area of a square mask."""
    ys, xs = torch.nonzero(m, as_tuple=True)
    s = m.shape[-1]
    x0, x1, y0, y1 = xs.min().item(), xs.max().item() + 1, ys.min().item(), ys.max().item() + 1
    return torch.tensor([(x0 + x1) / 2 / s, (y0 + y1) / 2 / s, (x1 - x0) / s, (y1 - y0) / s])


class TestD4Ops(unittest.TestCase):
    def test_group_has_eight_distinct_elements(self):
        t = torch.arange(16.0).reshape(4, 4)
        outs = {tuple(d4_apply(t, k).flatten().tolist()) for k in range(8)}
        self.assertEqual(len(outs), 8)
        self.assertTrue(torch.equal(d4_apply(t, 0), t))

    def test_boxes_follow_the_pixels(self):
        box = torch.tensor([0.30, 0.20, 0.25, 0.10])  # off-centre and non-square, so every element moves it differently
        m = box_mask(box.tolist(), 160)
        for k in range(8):
            with self.subTest(k=k):
                expected = mask_box(d4_apply(m, k))
                got = d4_boxes(box[None], k)[0]
                self.assertTrue(torch.allclose(got, expected, atol=1e-6), f"k={k}: {got} vs {expected}")

    def test_leading_dims_untouched(self):
        t = torch.randn(3, 5, 8, 8)
        for k in range(8):
            out = d4_apply(t, k)
            self.assertEqual(out.shape, t.shape)
            self.assertTrue(torch.equal(out[1, 2], d4_apply(t[1, 2], k)))


class _FakeBase(torch.utils.data.Dataset):
    """Minimal Ultralytics-style training item: square uint8 image, per-instance masks at 1/4 res, boxes."""

    def __init__(self):
        img = torch.zeros(3, 64, 64, dtype=torch.uint8)
        img[:, 8:16, 16:40] = 255                       # object at x 16-40, y 8-16
        masks = torch.zeros(1, 16, 16)
        masks[0, 2:4, 4:10] = 1
        self.item = {"img": img, "masks": masks, "sem_masks": masks[0].clone().long(),
                     "bboxes": torch.tensor([[28 / 64, 12 / 64, 24 / 64, 8 / 64]]), "cls": torch.zeros(1, 1),
                     "im_file": "a.jpg", "ratio_pad": (0.5, 0.5), "ori_shape": (128, 128)}

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.item.items()}

    @staticmethod
    def collate_fn(batch):
        return batch


class TestD4Dataset(unittest.TestCase):
    def test_image_masks_boxes_stay_consistent(self):
        ds = D4AugDataset(_FakeBase())
        seen = set()
        for _ in range(200):
            item = ds[0]
            k = item["d4"]
            seen.add(k)
            obj = (item["img"][0] > 0).float()
            self.assertTrue(torch.allclose(mask_box(obj), item["bboxes"][0], atol=1e-6))
            self.assertTrue(torch.allclose(mask_box(item["masks"][0]), item["bboxes"][0], atol=1e-6))
            self.assertTrue(torch.equal(item["sem_masks"], item["masks"][0].long()))
        self.assertEqual(seen, set(range(8)))

    def test_passthrough_attributes(self):
        ds = D4AugDataset(_FakeBase())
        self.assertEqual(len(ds), 1)
        self.assertIs(ds.collate_fn, _FakeBase.collate_fn)


class TestD4ThroughTrainingTransforms(unittest.TestCase):
    """Teacher warped onto the letterboxed canvas and then D4-transformed must sit on the D4-transformed GT."""

    def test_teacher_follows_every_d4_element(self):
        import tempfile
        import cv2
        import torch.nn.functional as F
        from ultralytics.cfg import get_cfg
        from ultralytics.data.dataset import YOLODataset
        from distillation.kd_trainer import SPATIAL_AUG_OFF, letterbox_content_box, warp_to_letterbox

        h0, w0 = 360, 640
        poly = [(w0 * 0.10, h0 * 0.20), (w0 * 0.45, h0 * 0.20), (w0 * 0.45, h0 * 0.35), (w0 * 0.10, h0 * 0.35)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            cv2.imwrite(str(root / "images" / "train" / "a.jpg"), np.full((h0, w0, 3), 128, np.uint8))
            (root / "labels" / "train" / "a.txt").write_text(
                "0 " + " ".join(f"{x / w0:.6f} {y / h0:.6f}" for x, y in poly) + "\n")
            hyp = get_cfg(overrides=dict(SPATIAL_AUG_OFF, imgsz=512, overlap_mask=False, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0))
            base = YOLODataset(img_path=str(root / "images" / "train"), data={"names": {0: "crack"}, "nc": 1, "channels": 3},
                               imgsz=512, augment=True, hyp=hyp, rect=False, batch_size=1, task="segment")
            ds = D4AugDataset(base)

            src = np.zeros((h0, w0), np.uint8)
            cv2.fillPoly(src, [np.array(poly, dtype=np.int32)], 1)
            seen = set()
            for _ in range(60):
                s = ds[0]
                k = s["d4"]
                seen.add(k)
                target = (256, 256)
                gt = F.interpolate(s["masks"].float()[None], size=target, mode="nearest")[0, 0] > 0.5
                box = letterbox_content_box(s["ori_shape"], s["ratio_pad"], s["img"].shape[-2:], target)
                teacher = warp_to_letterbox(torch.from_numpy(src).float()[None, None], box, target, 0.0)[0, 0]
                teacher = d4_apply(teacher, k) > 0.5
                iou = (gt & teacher).sum().item() / max(1, (gt | teacher).sum().item())
                self.assertGreater(iou, 0.9, f"k={k}")
            self.assertEqual(seen, set(range(8)))


class TestTrainerD4Wiring(unittest.TestCase):
    def _trainer(self, kd, **extra):
        from distillation.kd_trainer import KDSegmentationTrainer
        from utils.config_loader import load_config, override_config
        over = {"distillation.enabled": kd, "train.epochs": 1, "train.d4_aug": True,
                "distillation.losses.feature.enabled": False, "distillation.losses.boundary.enabled": False,
                "teacher.logits_dir": "data/teacher_logits_box/"}
        over.update(extra)
        return KDSegmentationTrainer(override_config(load_config("configs/config.yaml"), over))

    def test_both_arms_get_d4(self):
        from distillation.kd_trainer import KDYOLODataset
        data_path = "data/datasets/crack500_yolo/images/train"
        if not Path(data_path).exists():
            self.skipTest("local Crack500 not available")
        base = self._trainer(False)
        base.setup_model()
        self.assertIsInstance(base.build_dataset(data_path, mode="train", batch=4), D4AugDataset)
        self.assertNotIsInstance(base.build_dataset(data_path.replace("train", "val"), mode="val", batch=4), D4AugDataset)
        kd = self._trainer(True)
        kd.setup_model()
        ds = kd.build_dataset(data_path, mode="train", batch=4)
        self.assertIsInstance(ds, KDYOLODataset)
        self.assertIsInstance(ds.base_dataset, D4AugDataset)

    def test_d4_refuses_untransformed_feature_kd(self):
        with self.assertRaises(ValueError):
            self._trainer(True, **{"distillation.losses.feature.enabled": True})


if __name__ == "__main__":
    unittest.main()
