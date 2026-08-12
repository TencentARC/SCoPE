"""Camera-pose canonicalization shared by every SCoPE dataset.

All SCoPE datasets, regardless of their on-disk layout, converge on one camera
convention before the batch reaches the model:

1. Poses are OpenCV camera-to-world (c2w) matrices, shape ``[T, 3, 4]``.
2. The first camera is mapped to the identity, so every clip is expressed
   relative to its own first frame (``inv(c2w[0]) @ c2w``). This is the
   RealEstate10K convention; all datasets use it so the model never sees a
   dataset-specific world frame.
3. Translation is preprocessed by a per-clip near-distance depth: it is
   multiplied by ``trajectory_scale / near_depth``. ``near_depth`` comes from an
   offline estimate (``scripts/estimate_near_depth.py``) and is only a
   near-depth normalization that brings the translation magnitude into a
   comparable range across heterogeneous datasets. Scale itself is handled
   inside the model by a dedicated scale gate, not by this preprocessing.

The functions here are intentionally pure NumPy so the convention can be unit
tested without importing torch or decoding any video.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def to_c2w_44(poses: np.ndarray) -> np.ndarray:
    """Return homogeneous ``[T, 4, 4]`` c2w from ``[T, 3, 4]`` or ``[T, 4, 4]``."""
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected camera poses [T,3,4] or [T,4,4], got {poses.shape}")
    if poses.shape[-2:] == (4, 4):
        return poses
    bottom = np.broadcast_to(np.asarray([0, 0, 0, 1], dtype=poses.dtype), (poses.shape[0], 1, 4))
    return np.concatenate([poses, bottom], axis=1)


def first_camera_relative(poses: np.ndarray) -> np.ndarray:
    """Express all cameras relative to the first (first camera -> identity).

    Args:
        poses: OpenCV c2w, ``[T, 3, 4]`` or ``[T, 4, 4]``.

    Returns:
        ``[T, 3, 4]`` c2w with ``result[0]`` equal to identity.
    """
    c2w = to_c2w_44(poses)
    relative = np.linalg.inv(c2w[0])[None] @ c2w
    return relative[:, :3].astype(np.float32)


def scale_translation(
    poses34: np.ndarray, near_depth: float | None, trajectory_scale: float = 1.0
) -> np.ndarray:
    """Scale the translation column by ``trajectory_scale / near_depth``.

    When ``near_depth`` is ``None`` only ``trajectory_scale`` is applied. The
    rotation block is never touched.
    """
    poses34 = np.array(poses34, dtype=np.float32, copy=True)
    scale = float(trajectory_scale)
    if near_depth is not None:
        if not np.isfinite(near_depth) or near_depth <= 0:
            raise ValueError(f"near_depth must be finite and positive, got {near_depth}")
        scale = scale / float(near_depth)
    if scale != 1.0:
        poses34[:, :, 3] *= scale
    return poses34


def load_near_depth_map(near_depth_json: str | Path | None) -> dict[str, float] | None:
    """Load a ``clip_id -> near_depth`` map produced by estimate_near_depth.py.

    File layout::

        { "<clip_id>": {"near_depth": <float|null>, ...}, ... }

    Entries with null / non-finite / non-positive values are dropped. Returns
    ``None`` when no path is given or the file is missing.
    """
    if near_depth_json is None:
        return None
    path = Path(near_depth_json)
    if not path.exists():
        print(f"[near_depth] file not found: {path} - trajectory_scale used for all clips")
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for key, value in raw.items():
        near_depth = value.get("near_depth") if isinstance(value, dict) else value
        if near_depth is None:
            continue
        if not np.isfinite(near_depth) or near_depth <= 0:
            continue
        out[key] = float(near_depth)
    print(f"[near_depth] loaded {len(out)} clips from {path}")
    return out
