"""
Teacher Module — SAM 2
======================
Runs SAM 2 on training images using GT box prompts,
saves raw pre-sigmoid logits as .npy files.

CRITICAL: We save logits (floats), not binary masks.
Boundary pixels hover near 0.0 (sigmoid ~0.5) — that
uncertainty IS the dark knowledge that makes KD work.
"""

import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import cv2


class SAM2Teacher:
    """
    SAM 2 teacher inference wrapper.

    Usage:
        teacher = SAM2Teacher(cfg)
        teacher.load()
        teacher.generate_logits(dataset)  # saves .npy files
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.teacher.device
        self.prompt_type = cfg.teacher.prompt_type
        self.logits_dir = Path(cfg.teacher.logits_dir)
        self.logits_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.predictor = None

    def load(self):
        """Load SAM 2 model and predictor."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self.model = build_sam2(
                config_file=self.cfg.teacher.config,
                ckpt_path=self.cfg.teacher.checkpoint,
                device=self.device,
            )
            self.predictor = SAM2ImagePredictor(self.model)
            print(f"[Teacher] SAM 2 loaded on {self.device}")

        except ImportError:
            print("[Teacher] sam2 package not found.")
            print("  Install: pip install git+https://github.com/facebookresearch/sam2.git")
            raise

    def _get_box_prompt_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert binary mask to bounding box [x1, y1, x2, y2]."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return np.array([cmin, rmin, cmax, rmax], dtype=np.float32)

    @torch.no_grad()
    def generate_logits_for_image(
        self,
        image_path: str,
        gt_masks: list[np.ndarray],   # list of binary instance masks
        image_id: str,
    ) -> dict:
        """
        Run SAM 2 on one image, return dict of logit arrays.

        Returns:
            {
              "image_id": str,
              "logits":   np.ndarray  shape (N, H, W) float32  ← pre-sigmoid
              "masks":    np.ndarray  shape (N, H, W) bool
            }
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image)

        all_logits = []
        all_masks = []

        for mask in gt_masks:
            if self.prompt_type == "box":
                box = self._get_box_prompt_from_mask(mask)
                if box is None:
                    continue
                masks_out, scores, logits = self.predictor.predict(
                    box=box,
                    multimask_output=False,   # single best mask
                )
            elif self.prompt_type == "point":
                # Use centroid as point prompt
                M = cv2.moments(mask.astype(np.uint8))
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                masks_out, scores, logits = self.predictor.predict(
                    point_coords=np.array([[cx, cy]]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )
            else:
                raise ValueError(f"Unknown prompt_type: {self.prompt_type}")

            # logits shape: (1, 256, 256) — SAM internal resolution
            # We store the raw float logits (pre-sigmoid)
            all_logits.append(logits[0].astype(np.float32))   # (256, 256)
            all_masks.append(masks_out[0])                    # (H, W) bool

        if not all_logits:
            return None

        # Extract features from predictor
        features = self.predictor._features
        image_embed = features["image_embed"]  # shape (1, 256, 64, 64)
        high_res_feats = features["high_res_feats"]  # [(1, 32, 256, 256), (1, 64, 128, 128)]

        return {
            "image_id": image_id,
            "logits": np.stack(all_logits, axis=0),   # (N, 256, 256)
            "masks":  np.stack(all_masks,  axis=0),   # (N, H, W)
            "image_embed": image_embed.cpu().half().numpy(),
            "feat0": high_res_feats[0].cpu().half().numpy(),
            "feat1": high_res_feats[1].cpu().half().numpy(),
        }

    def save_logits(self, result: dict):
        """Save logits as .npy, binary masks as .npz, and features as .npz."""
        if result is None:
            return
        image_id = result["image_id"]
        np.save(self.logits_dir / f"{image_id}_logits.npy", result["logits"])
        np.savez_compressed(
            self.logits_dir / f"{image_id}_masks.npz",
            masks=result["masks"]
        )
        if "image_embed" in result:
            np.savez_compressed(
                self.logits_dir / f"{image_id}_features.npz",
                image_embed=result["image_embed"],
                feat0=result["feat0"],
                feat1=result["feat1"]
            )

    def load_logits(self, image_id: str) -> dict | None:
        """Load pre-saved logits for an image."""
        logit_path = self.logits_dir / f"{image_id}_logits.npy"
        mask_path  = self.logits_dir / f"{image_id}_masks.npz"

        if not logit_path.exists():
            return None

        logits = np.load(logit_path)
        masks  = np.load(mask_path)["masks"] if mask_path.exists() else None

        return {"image_id": image_id, "logits": logits, "masks": masks}

    def generate_all(self, image_paths: list, gt_masks_list: list, image_ids: list):
        """
        Batch generate and save logits for all training images.

        Args:
            image_paths:    list of str paths to images
            gt_masks_list:  list of list[np.ndarray] (per-image instance masks)
            image_ids:      list of str unique IDs
        """
        print(f"[Teacher] Generating logits for {len(image_paths)} images...")
        skipped = 0
        for img_path, gt_masks, img_id in tqdm(
            zip(image_paths, gt_masks_list, image_ids),
            total=len(image_paths)
        ):
            # Skip if already generated
            if (self.logits_dir / f"{img_id}_logits.npy").exists():
                skipped += 1
                continue

            result = self.generate_logits_for_image(img_path, gt_masks, img_id)
            self.save_logits(result)

        print(f"[Teacher] Done. Skipped {skipped} already-generated images.")

    @torch.no_grad()
    def generate_pseudo_labels(
        self, image_path: str, image_id: str, auto_everything: bool = True
    ) -> dict:
        """
        For RDD2022 (bbox-only): generate pseudo instance masks via SAM 2.
        Use this to expand training data without GT masks.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)

        if auto_everything:
            # SAM 2 automatic mask generator — no prompts needed
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            generator = SAM2AutomaticMaskGenerator(
                self.model,
                points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                min_mask_region_area=100,
            )
            auto_masks = generator.generate(image)
            return {
                "image_id": image_id,
                "pseudo_masks": [m["segmentation"] for m in auto_masks],
                "scores":       [m["predicted_iou"] for m in auto_masks],
            }


# ──────────────────────────────────────────────
# Soft logit utilities used during training
# ──────────────────────────────────────────────

def logits_to_soft_probs(logits: np.ndarray, temperature: float = 4.0) -> torch.Tensor:
    """
    Convert saved SAM logits to soft probability targets.

    Args:
        logits:      np.ndarray (N, H, W) raw SAM logits
        temperature: KD temperature (higher = softer distribution)

    Returns:
        torch.Tensor (N, H, W) in [0, 1]
    """
    t = torch.from_numpy(logits).float()
    return torch.sigmoid(t / temperature)


def resize_logits_to_student(
    logits: np.ndarray,
    target_size: tuple[int, int]
) -> np.ndarray:
    """
    Resize SAM logits (256×256 internal) to student output size.
    Uses bilinear interpolation — safe for logits (not binary).
    """
    n = logits.shape[0]
    resized = []
    for i in range(n):
        r = cv2.resize(
            logits[i],
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR   # bilinear for logits
        )
        resized.append(r)
    return np.stack(resized, axis=0)
