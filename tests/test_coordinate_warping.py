"""
Unit tests for P0-1 Geometric Coordinate Warping and Alignment.
Verifies that teacher mask logits and feature maps are projected accurately
from unpadded raw image space into letterbox student canvas space.

All tests exercise the production helpers imported from distillation.kd_trainer
(single source of truth), and the pipeline test checks them against Ultralytics'
own resize + LetterBox output rather than a re-implementation of the formula.
"""

import math
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from distillation.kd_trainer import letterbox_content_box, warp_to_letterbox


def _ultralytics_letterbox(img: np.ndarray, imgsz: int):
    """Replicate the training-time geometry: BaseDataset.load_image (long side -> imgsz) + LetterBox."""
    from ultralytics.data.augment import LetterBox

    h0, w0 = img.shape[:2]
    r = imgsz / max(h0, w0)
    if r != 1:
        w, h = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    # BaseDataset.get_image_and_label stores (h_ratio, w_ratio)
    labels = {"img": img, "ratio_pad": (h / h0, w / w0)}
    labels = LetterBox(new_shape=(imgsz, imgsz), scaleup=False)(labels=labels)
    return labels["img"], labels["ratio_pad"]


class TestCoordinateWarping(unittest.TestCase):
    def test_letterbox_warping_landscape(self):
        """360x640 into a 640x640 canvas: 140 px top/bottom padding -> 35 px on a 160x160 canvas."""
        # Ultralytics ratio_pad: ((h_ratio, w_ratio), (left, top))
        ratio_pad = ((1.0, 1.0), (0.0, 140.0))
        box = letterbox_content_box((360, 640), ratio_pad, (640, 640), (160, 160))
        self.assertEqual(box, (35, 0, 90, 160))

        raw_t = torch.full((1, 1, 360, 640), -20.0)
        raw_t[:, :, 160:200, 300:340] = 10.0
        padded = warp_to_letterbox(raw_t, box, (160, 160), pad_value=-20.0)

        self.assertEqual(padded.shape, (1, 1, 160, 160))
        self.assertTrue(torch.all(padded[:, :, :35, :] == -20.0))
        self.assertTrue(torch.all(padded[:, :, 125:, :] == -20.0))
        # Active patch lands at the centre of the content region (35 + 45, 80)
        self.assertGreater(padded[0, 0, 80, 80].item(), 0.0)

    def test_extreme_aspect_ratio_portrait(self):
        """640x360 portrait: horizontal padding instead of vertical."""
        ratio_pad = ((1.0, 1.0), (140.0, 0.0))
        box = letterbox_content_box((640, 360), ratio_pad, (640, 640), (160, 160))
        self.assertEqual(box, (0, 35, 160, 90))

    def test_non_uniform_ratio_not_transposed(self):
        """ratio_pad[0] is (h_ratio, w_ratio); swapping them must change the result (audit F2)."""
        # 400x800 source, resized to 200x512 content inside a 512x512 canvas
        ratio_pad = ((0.5, 0.64), (0.0, 156.0))
        top, left, unpad_h, unpad_w = letterbox_content_box((400, 800), ratio_pad, (512, 512), (256, 256))
        self.assertEqual((unpad_h, unpad_w), (100, 256))
        self.assertEqual((top, left), (78, 0))

    def test_valid_content_mask(self):
        """Mask must be 1.0 inside content and 0.0 in letterbox padding."""
        top, left, unpad_h, unpad_w = letterbox_content_box((360, 640), ((1.0, 1.0), (0.0, 140.0)), (640, 640), (160, 160))
        valid = torch.zeros((160, 160), dtype=torch.bool)
        valid[top : top + unpad_h, left : left + unpad_w] = True
        self.assertEqual(int(valid.sum()), 90 * 160)
        self.assertFalse(valid[:top].any())
        self.assertFalse(valid[top + unpad_h :].any())

    def test_ratio_pad_absent_reconstructs_from_ori_shape(self):
        """Without ratio_pad the box is rebuilt from ori_shape exactly like LetterBox would."""
        with_rp = letterbox_content_box((360, 640), ((0.8, 0.8), (0.0, 112.0)), (512, 512), (256, 256))
        without_rp = letterbox_content_box((360, 640), None, (512, 512), (256, 256))
        self.assertEqual(with_rp, without_rp)

    def test_missing_ori_shape_returns_none(self):
        """No original shape means the geometry is unknown; callers must not silently plain-resize."""
        self.assertIsNone(letterbox_content_box(None, ((0.8, 0.8), (0.0, 112.0)), (512, 512), (256, 256)))

    def test_matches_ultralytics_letterbox_pipeline(self):
        """A marker in source space must land where Ultralytics' own resize + LetterBox puts it."""
        target = (256, 256)
        for (h0, w0), imgsz in [((360, 640), 512), ((640, 360), 512), ((648, 848), 512),
                                ((720, 1920), 640), ((512, 512), 512), ((360, 640), 768)]:
            with self.subTest(shape=(h0, w0), imgsz=imgsz):
                src = np.zeros((h0, w0, 3), dtype=np.uint8)
                y0, y1, x0, x1 = h0 // 4, h0 // 4 + h0 // 5, w0 // 3, w0 // 3 + w0 // 6
                src[y0:y1, x0:x1] = 255

                canvas, ratio_pad = _ultralytics_letterbox(src, imgsz)
                self.assertEqual(canvas.shape[:2], (imgsz, imgsz))
                student = cv2.resize(canvas[..., 0], target[::-1], interpolation=cv2.INTER_LINEAR) > 127

                box = letterbox_content_box((h0, w0), ratio_pad, (imgsz, imgsz), target)
                teacher_src = torch.from_numpy((src[..., 0] > 127).astype(np.float32))[None, None]
                teacher = warp_to_letterbox(teacher_src, box, target, pad_value=0.0)[0, 0].numpy() > 0.5

                inter = np.logical_and(student, teacher).sum()
                union = np.logical_or(student, teacher).sum()
                self.assertGreater(inter / union, 0.95)

                ys, xs = np.nonzero(student)
                yt, xt = np.nonzero(teacher)
                self.assertLess(abs(ys.mean() - yt.mean()), 1.0)
                self.assertLess(abs(xs.mean() - xt.mean()), 1.0)


class TestTrainingPipelineGeometry(unittest.TestCase):
    """
    The KD loss only runs on training batches. Ultralytics' training transforms emit
    ratio_pad as (h_ratio, w_ratio) with no padding entry, while the canvas is still
    letterboxed. Build a tiny dataset and push it through the real training transforms.
    """

    ZERO_AUG = dict(mosaic=0.0, close_mosaic=0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
                    perspective=0.0, fliplr=0.0, flipud=0.0, erasing=0.0, mixup=0.0, copy_paste=0.0,
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)

    def setUp(self):
        import tempfile
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _dataset(self, h0, w0, polys, imgsz=512, overlap_mask=True):
        from ultralytics.cfg import get_cfg
        from ultralytics.data.dataset import YOLODataset

        img_dir = self.temp_dir / "images" / "train"
        lbl_dir = self.temp_dir / "labels" / "train"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        cv2.imwrite(str(img_dir / "a.jpg"), rng.integers(60, 200, (h0, w0, 3), dtype=np.uint8))
        with open(lbl_dir / "a.txt", "w") as f:
            for poly in polys:
                f.write("0 " + " ".join(f"{x / w0:.6f} {y / h0:.6f}" for x, y in poly) + "\n")
        hyp = get_cfg(overrides=dict(self.ZERO_AUG, imgsz=imgsz, overlap_mask=overlap_mask))
        return YOLODataset(img_path=str(img_dir), data={"names": {0: "crack"}, "nc": 1, "channels": 3},
                           imgsz=imgsz, augment=True, hyp=hyp, task="segment", rect=False, batch_size=1)

    def test_training_batch_teacher_aligns_with_gt(self):
        for (h0, w0), imgsz in [((360, 640), 512), ((640, 360), 512), ((720, 1920), 640)]:
            with self.subTest(shape=(h0, w0), imgsz=imgsz):
                poly = [(w0 * 0.30, h0 * 0.40), (w0 * 0.60, h0 * 0.40), (w0 * 0.60, h0 * 0.55), (w0 * 0.30, h0 * 0.55)]
                s = self._dataset(h0, w0, [poly], imgsz=imgsz)[0]
                self.assertEqual(tuple(s["img"].shape[-2:]), (imgsz, imgsz))

                target = (256, 256)
                gt = F.interpolate((s["masks"][0] > 0).float()[None, None], size=target, mode="nearest")[0, 0] > 0.5

                src = np.zeros((h0, w0), dtype=np.uint8)
                cv2.fillPoly(src, [np.array(poly, dtype=np.int32)], 1)
                box = letterbox_content_box(s["ori_shape"], s["ratio_pad"], s["img"].shape[-2:], target)
                teacher = warp_to_letterbox(torch.from_numpy(src).float()[None, None], box, target, 0.0)[0, 0] > 0.5

                iou = (gt & teacher).sum().item() / max(1, (gt | teacher).sum().item())
                self.assertGreater(iou, 0.9, f"ratio_pad={s['ratio_pad']} box={box}")

    def test_instance_order_matches_label_order_only_without_overlap_masks(self):
        """
        Teacher logits are stored in label-file order and the KD loss uses teacher[target_gt_idx].
        overlap_mask=True sorts batch instances by area, breaking that pairing.
        """
        h0, w0 = 360, 640
        small = [(50, 50), (100, 50), (100, 80), (50, 80)]
        big = [(300, 150), (600, 150), (600, 300), (300, 300)]
        for overlap_mask, expected_first in [(True, "big"), (False, "small")]:
            with self.subTest(overlap_mask=overlap_mask):
                s = self._dataset(h0, w0, [small, big], overlap_mask=overlap_mask)[0]
                first_box_w = float(s["bboxes"][0][2])  # normalized xywh
                self.assertEqual("big" if first_box_w > 0.3 else "small", expected_first)

    def test_trainer_disables_overlap_masks_for_both_arms(self):
        from distillation.kd_trainer import KDSegmentationTrainer
        from utils.config_loader import load_config, override_config

        master = load_config("configs/config.yaml")
        for enabled in (False, True):
            with self.subTest(kd_enabled=enabled):
                cfg = override_config(master, {"distillation.enabled": enabled, "train.epochs": 1})
                trainer = KDSegmentationTrainer(cfg=cfg)
                self.assertFalse(trainer.args.overlap_mask)


class TestMissingGeometryFallback(unittest.TestCase):
    """Audit F5: a batch without ori_shape must not silently fall back to the pre-P0-1 plain resize."""

    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.logits_dir = Path(self.temp_dir) / "logits"
        self.logits_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(self.logits_dir / "sample001.npy"), np.zeros((1, 256, 256), dtype=np.float32))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _trainer(self, strict):
        from distillation.kd_trainer import KDSegmentationTrainer
        from utils.config_loader import ConfigNode

        kd_cfg = ConfigNode({
            "enabled": True,
            "strict": strict,
            "temperature": 2.0,
            "losses": {
                "task": {"weight": 1.0},
                "mask_kd": {"enabled": True},
                "feature": {"enabled": False},
                "boundary": {"enabled": False},
            },
        })
        return KDSegmentationTrainer(
            overrides={"model": "yolo11n-seg.pt", "data": "data/datasets/combined_yolo/dataset.yaml", "epochs": 1},
            kd_cfg=kd_cfg,
            logits_dir=self.logits_dir,
        )

    def test_strict_mode_raises(self):
        trainer = self._trainer(strict=True)
        with self.assertRaises(RuntimeError) as ctx:
            trainer._letterbox_box(None, None, (512, 512), (256, 256), "sample001")
        self.assertIn("ori_shape", str(ctx.exception))

    def test_permissive_mode_warns_once(self):
        import contextlib
        import io

        trainer = self._trainer(strict=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertIsNone(trainer._letterbox_box(None, None, (512, 512), (256, 256), "a"))
            self.assertIsNone(trainer._letterbox_box(None, None, (512, 512), (256, 256), "b"))
        self.assertEqual(buf.getvalue().count("Missing letterbox metadata"), 1)

    def test_present_metadata_returns_box(self):
        trainer = self._trainer(strict=True)
        box = trainer._letterbox_box((360, 640), ((0.8, 0.8), (0.0, 112.0)), (512, 512), (256, 256), "a")
        self.assertEqual(box, (56, 0, 144, 256))


if __name__ == "__main__":
    unittest.main()
