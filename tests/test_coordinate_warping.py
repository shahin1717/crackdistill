"""
Unit tests for P0-1 Geometric Coordinate Warping and Alignment.
Verifies that teacher mask logits and feature maps are projected accurately
from unpadded raw image space into letterbox student canvas space.
"""

import math
import sys
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestCoordinateWarping(unittest.TestCase):
    def test_letterbox_warping_landscape(self):
        """
        Test letterbox projection for landscape image (e.g. 360x640 -> 640x640 canvas).
        Scale ratio = 640 / 640 = 1.0 (or 640/max(360, 640)).
        With 160x160 student mask canvas, verify unpadded size and top/bottom padding.
        """
        target_h, target_w = 160, 160
        ori_h, ori_w = 360, 640
        # Ultralytics ratio_pad: ((scale_w, scale_h), (pad_w, pad_h))
        # For 360x640 into 640x640:
        # r = min(640/360, 640/640) = 1.0
        # unpad_h = round(360 * 1.0) = 360; pad_h = (640 - 360) / 2 = 140
        # ratio_pad = ((1.0, 1.0), (0.0, 140.0))
        ratio_pad = ((1.0, 1.0), (0.0, 140.0))
        (r_w, r_h), (pad_w, pad_h) = ratio_pad

        scale_x = target_w / (ori_w * r_w + pad_w * 2)  # 160 / 640 = 0.25
        scale_y = target_h / (ori_h * r_h + pad_h * 2)  # 160 / 640 = 0.25

        t_unpad_w = max(1, int(round(ori_w * r_w * scale_x)))  # 640 * 0.25 = 160
        t_unpad_h = max(1, int(round(ori_h * r_h * scale_y)))  # 360 * 0.25 = 90

        t_pad_x = int(round(pad_w * scale_x))  # 0
        t_pad_y = int(round(pad_h * scale_y))  # 140 * 0.25 = 35

        t_left = t_pad_x
        t_right = target_w - t_unpad_w - t_left
        t_top = t_pad_y
        t_bottom = target_h - t_unpad_h - t_top

        self.assertEqual(t_unpad_w, 160)
        self.assertEqual(t_unpad_h, 90)
        self.assertEqual(t_top, 35)
        self.assertEqual(t_bottom, 35)
        self.assertEqual(t_left, 0)
        self.assertEqual(t_right, 0)

        # Create synthetic teacher logit in raw space: (1, 1, 360, 640)
        # Put a high-confidence patch in center (180, 320)
        raw_t = torch.full((1, 1, ori_h, ori_w), -20.0)
        raw_t[:, :, 160:200, 300:340] = 10.0

        # Warp:
        scaled = F.interpolate(raw_t, size=(t_unpad_h, t_unpad_w), mode="bilinear", align_corners=False)
        padded = F.pad(scaled, (t_left, t_right, t_top, t_bottom), value=-20.0)

        self.assertEqual(padded.shape, (1, 1, 160, 160))
        # Top padding region must be strictly background (-20.0)
        self.assertTrue(torch.all(padded[:, :, :t_top, :] == -20.0))
        # Bottom padding region must be strictly background (-20.0)
        self.assertTrue(torch.all(padded[:, :, target_h - t_bottom:, :] == -20.0))
        # Active patch should be centered in the content region
        # Expected vertical center: 35 + 45 = 80
        # Expected horizontal center: 80
        center_val = padded[0, 0, 80, 80].item()
        self.assertGreater(center_val, 0.0)

    def test_valid_content_mask(self):
        """
        Verify valid_content_mask generation for student mask loss normalization.
        Mask must be 1.0 inside content and 0.0 in letterbox padding.
        """
        target_h, target_w = 160, 160
        t_top, t_bottom, t_left, t_right = 35, 35, 0, 0

        valid_mask = torch.zeros((1, 1, target_h, target_w), dtype=torch.float32)
        valid_mask[:, :, t_top : target_h - t_bottom, t_left : target_w - t_right] = 1.0

        # Total content area = 90 * 160 = 14400 pixels
        self.assertEqual(valid_mask.sum().item(), 90 * 160)
        # Verify padding areas have zero weight
        self.assertEqual(valid_mask[:, :, :t_top, :].sum().item(), 0.0)
        self.assertEqual(valid_mask[:, :, target_h - t_bottom:, :].sum().item(), 0.0)

    def test_extreme_aspect_ratio_portrait(self):
        """
        Test letterbox projection for tall portrait image (e.g. 640x360).
        Verifies horizontal padding calculation and stability.
        """
        target_h, target_w = 160, 160
        ori_h, ori_w = 640, 360
        ratio_pad = ((1.0, 1.0), (140.0, 0.0))
        (r_w, r_h), (pad_w, pad_h) = ratio_pad

        scale_x = target_w / (ori_w * r_w + pad_w * 2)
        scale_y = target_h / (ori_h * r_h + pad_h * 2)

        t_unpad_w = max(1, int(round(ori_w * r_w * scale_x)))
        t_unpad_h = max(1, int(round(ori_h * r_h * scale_y)))

        t_pad_x = int(round(pad_w * scale_x))
        t_pad_y = int(round(pad_h * scale_y))

        t_left = t_pad_x
        t_right = target_w - t_unpad_w - t_left
        t_top = t_pad_y
        t_bottom = target_h - t_unpad_h - t_top

        self.assertEqual(t_unpad_h, 160)
        self.assertEqual(t_unpad_w, 90)
        self.assertEqual(t_left, 35)
        self.assertEqual(t_right, 35)
        self.assertEqual(t_top, 0)
        self.assertEqual(t_bottom, 0)


if __name__ == "__main__":
    unittest.main()
