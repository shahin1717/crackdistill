"""
CrackDistill — Gaussian-Weighted Tiled Sliding-Window Inference Engine
=====================================================================
Performs overlapping sliding-window inference on high-resolution images
(e.g., 2000x1500 full-scene asphalt pavement photographs) using 2D Gaussian
apodization blending to eliminate tiling boundary discontinuities.
"""

from pathlib import Path
import cv2
import numpy as np
import torch


def create_gaussian_weight_map(tile_size: int = 512, sigma: float = 0.35) -> np.ndarray:
    """
    Generate a 2D Gaussian window to smoothly blend overlapping inference tiles.

    Args:
        tile_size (int): Dimension of square tile (default 512).
        sigma (float): Gaussian spread parameter relative to [-1, 1] range (default 0.35).

    Returns:
        np.ndarray: Normalized 2D float32 weight array of shape (tile_size, tile_size).
    """
    ax = np.linspace(-1, 1, tile_size)
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    gauss_2d = np.outer(gauss_1d, gauss_1d).astype(np.float32)
    gauss_2d = np.maximum(gauss_2d, 0.05)  # Minimum floor to prevent edge zero division
    return gauss_2d / gauss_2d.max()


def tiled_predict_prob_map(
    model,
    img_bgr: np.ndarray,
    tile_size: int = 512,
    stride: int = 384,
    conf: float = 0.25,
    sigma: float = 0.35,
    device: str | None = None,
) -> np.ndarray:
    """
    Run overlapping sliding-window inference on full-resolution image with
    2D Gaussian apodization blending, returning continuous probability map.

    Args:
        model: Ultralytics YOLO segmentation model or callable.
        img_bgr (np.ndarray): Full-resolution input image in BGR format (H, W, 3).
        tile_size (int): Size of sliding-window tile (default 512).
        stride (int): Sliding step in pixels; overlap is tile_size - stride (default 384 = 25% overlap).
        conf (float): Detection confidence threshold for model.predict (default 0.25).
        sigma (float): 2D Gaussian apodization parameter (default 0.35).
        device (str | None): Computing device ('cuda', 'cpu', or auto-detect if None).

    Returns:
        np.ndarray: Continuous crack probability map of shape (H, W) in range [0.0, 1.0].
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    h, w = img_bgr.shape[:2]
    full_prob_map = np.zeros((h, w), dtype=np.float32)
    weight_accum_map = np.zeros((h, w), dtype=np.float32)
    weight_window = create_gaussian_weight_map(tile_size, sigma)

    # Compute step coordinates covering the entire canvas
    y_steps = list(range(0, max(1, h - tile_size + 1), stride))
    if not y_steps or y_steps[-1] + tile_size < h:
        y_steps.append(max(0, h - tile_size))

    x_steps = list(range(0, max(1, w - tile_size + 1), stride))
    if not x_steps or x_steps[-1] + tile_size < w:
        x_steps.append(max(0, w - tile_size))

    for y0 in y_steps:
        for x0 in x_steps:
            tile = img_bgr[y0 : y0 + tile_size, x0 : x0 + tile_size]
            
            # Pad tile if canvas is smaller than tile_size
            pad_h = max(0, tile_size - tile.shape[0])
            pad_w = max(0, tile_size - tile.shape[1])
            if pad_h > 0 or pad_w > 0:
                tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

            results = model.predict(tile, imgsz=tile_size, conf=conf, verbose=False, device=device)
            r = results[0]

            tile_prob = np.zeros((tile_size, tile_size), dtype=np.float32)
            if r.masks is not None and len(r.masks) > 0:
                for m in r.masks.data.cpu().numpy():
                    m_resized = cv2.resize(m, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                    tile_prob = np.maximum(tile_prob, m_resized)

            # Unpad if tile was padded
            actual_h = min(tile_size, h - y0)
            actual_w = min(tile_size, w - x0)

            full_prob_map[y0 : y0 + actual_h, x0 : x0 + actual_w] += (
                tile_prob[:actual_h, :actual_w] * weight_window[:actual_h, :actual_w]
            )
            weight_accum_map[y0 : y0 + actual_h, x0 : x0 + actual_w] += (
                weight_window[:actual_h, :actual_w]
            )

    weight_accum_map = np.maximum(weight_accum_map, 1e-5)
    return full_prob_map / weight_accum_map


def tiled_predict_image_gaussian(
    model,
    img_bgr: np.ndarray,
    tile_size: int = 512,
    stride: int = 384,
    conf: float = 0.25,
    threshold: float = 0.35,
    sigma: float = 0.35,
    device: str | None = None,
) -> np.ndarray:
    """
    Run overlapping sliding-window inference on full-resolution image with
    2D Gaussian apodization blending, returning binary mask (0 or 1).

    Args:
        model: Ultralytics YOLO segmentation model or callable.
        img_bgr (np.ndarray): Full-resolution input image in BGR format (H, W, 3).
        tile_size (int): Size of sliding-window tile (default 512).
        stride (int): Sliding step in pixels (default 384).
        conf (float): Detection confidence threshold (default 0.25).
        threshold (float): Final binarization probability threshold (default 0.35).
        sigma (float): 2D Gaussian apodization parameter (default 0.35).
        device (str | None): Computing device ('cuda', 'cpu', or auto-detect).

    Returns:
        np.ndarray: Binary crack segmentation mask of shape (H, W) with values {0, 1}, uint8.
    """
    prob_map = tiled_predict_prob_map(
        model=model,
        img_bgr=img_bgr,
        tile_size=tile_size,
        stride=stride,
        conf=conf,
        sigma=sigma,
        device=device,
    )
    return (prob_map > threshold).astype(np.uint8)
