"""
CrackDistill Inference Package
==============================
Provides high-resolution tiled inference and sliding-window segmentation utilities.
"""

from inference.tiled_inference import (
    create_gaussian_weight_map,
    tiled_predict_image_gaussian,
    tiled_predict_prob_map,
)

__all__ = [
    "create_gaussian_weight_map",
    "tiled_predict_image_gaussian",
    "tiled_predict_prob_map",
]
