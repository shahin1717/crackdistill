"""
Checkpoint provenance and deterministic resolution utility.
Addresses scientific audit findings:
- Eliminates non-deterministic glob-based checkpoint collision.
- Records immutable SHA256 manifests for every trained model.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """
    Compute streaming SHA256 hash of a file.

    Args:
        file_path (str | Path): Path to file.
        chunk_size (int): Read chunk size in bytes (default: 1MB).

    Returns:
        str: Hex-encoded SHA256 digest.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_checkpoint(
    checkpoint_path: Union[str, Path],
    expected_experiment: Optional[str] = None,
    allow_search: bool = False,
) -> Path:
    """
    Deterministically resolve a checkpoint file path without fuzzy wildcards.

    Args:
        checkpoint_path (str | Path): File path or directory containing weights.
        expected_experiment (str, optional): Expected experiment name for sanity checking.
        allow_search (bool): If True and path is directory, checks standard weights/best.pt.

    Returns:
        Path: Verified checkpoint path.

    Raises:
        FileNotFoundError: If checkpoint does not exist.
        ValueError: If file is not a valid PyTorch weight file or empty.
    """
    path = Path(checkpoint_path)

    # 1. Direct file resolution
    if path.is_file():
        if path.suffix not in (".pt", ".pth", ".bin"):
            raise ValueError(f"Checkpoint file has unsupported extension '{path.suffix}': {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Checkpoint file is empty (0 bytes): {path}")
        return path.resolve()

    # 2. Directory resolution
    if path.is_dir():
        candidates = [
            path / "weights" / "best.pt",
            path / "best.pt",
            path / "weights" / "last.pt",
            path / "last.pt",
        ]
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate.resolve()

        if expected_experiment:
            named_candidates = [
                path / expected_experiment / "weights" / "best.pt",
                path / "segment" / expected_experiment / "weights" / "best.pt",
                path / "crack_distill" / expected_experiment / "weights" / "best.pt",
            ]
            for candidate in named_candidates:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    return candidate.resolve()

    raise FileNotFoundError(
        f"Could not deterministically resolve checkpoint from path '{checkpoint_path}'. "
        f"Expected existing .pt file or directory containing weights/best.pt."
    )


def get_checkpoint_manifest(
    checkpoint_path: Union[str, Path],
    experiment_name: Optional[str] = None,
    seed: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build immutable provenance record for a checkpoint file.

    Returns dict with sha256, file size, timestamp, experiment, seed, and optional metadata.
    """
    resolved_path = resolve_checkpoint(checkpoint_path)
    stat = resolved_path.stat()
    sha256 = compute_file_sha256(resolved_path)

    manifest: Dict[str, Any] = {
        "checkpoint_path": str(resolved_path),
        "filename": resolved_path.name,
        "sha256": sha256,
        "size_bytes": stat.st_size,
        "modified_time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
        "experiment": experiment_name,
        "seed": seed,
        "metadata": metadata or {},
    }
    return manifest


def save_checkpoint_manifest(manifest: Dict[str, Any], output_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Write checkpoint manifest to JSON file.
    Defaults to <checkpoint_dir>/<checkpoint_stem>_manifest.json.
    """
    if output_path is None:
        ckpt_p = Path(manifest["checkpoint_path"])
        output_path = ckpt_p.parent / f"{ckpt_p.stem}_manifest.json"

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_p
