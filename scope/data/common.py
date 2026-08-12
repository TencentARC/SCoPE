"""Shared video/pose plumbing and the base dataset for SCoPE training.

Each dataset only has to implement how it finds clips on disk and how it reads
one clip's frames, raw c2w poses, intrinsics, and caption. The base class then
applies the single shared convention (first-camera-relative poses + a per-clip
near-depth translation preprocessing; scale itself is handled by the model's
scale gate) and emits the common batch contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from scope.data._pose import first_camera_relative, scale_translation

CONTRACT_KEYS = ("video_id", "video", "pose", "x_fov", "xi", "caption")


def frames_to_video_tensor(frames: list[Image.Image], height: int, width: int) -> torch.Tensor:
    """Resize PIL frames and stack into ``[C, T, H, W]`` in ``[-1, 1]``."""
    tensors = []
    for frame in frames:
        if frame.size != (width, height):
            frame = frame.resize((width, height), Image.LANCZOS)
        tensor = TF.to_tensor(frame) * 2.0 - 1.0
        tensors.append(tensor)
    return torch.stack(tensors, dim=1).contiguous()


def linspace_indices(total: int, num_frames: int) -> list[int]:
    """Uniformly sample ``num_frames`` indices from ``[0, total)`` (tail-padded)."""
    if total <= 0:
        raise ValueError("Cannot sample from an empty clip")
    if total >= num_frames:
        return np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    return np.pad(np.arange(total), (0, num_frames - total), mode="edge").tolist()


class BaseCameraVideoDataset(Dataset):
    """Base class enforcing the shared SCoPE camera/video convention.

    Subclasses implement :meth:`_load_raw` returning a dict with keys
    ``video_id``, ``frames`` (list of PIL images), ``poses`` (OpenCV c2w
    ``[T, 3, 4]`` or ``[T, 4, 4]``), ``x_fov`` (radians), ``xi``, and
    ``caption``. The base class canonicalizes poses and builds the batch.
    """

    def __init__(
        self,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        near_depth_map: dict[str, float] | None = None,
        trajectory_scale: float = 1.0,
        return_first_frame: bool = True,
        max_retries: int = 10,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.near_depth_map = near_depth_map
        self.trajectory_scale = trajectory_scale
        self.return_first_frame = return_first_frame
        self.max_retries = max_retries

    def _load_raw(self, index: int) -> dict[str, Any]:
        raise NotImplementedError

    def filter_ids_by_near_depth(self, ids: list[str]) -> list[str]:
        """Drop clips lacking a valid near-depth when a map is configured."""
        if self.near_depth_map is None:
            return ids
        allowed = set(self.near_depth_map)
        kept = [i for i in ids if i in allowed]
        dropped = len(ids) - len(kept)
        if dropped:
            print(f"[{type(self).__name__}] near_depth dropped {dropped}/{len(ids)} clips")
        return kept

    def _finalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        video = frames_to_video_tensor(raw["frames"], self.height, self.width)
        poses = first_camera_relative(raw["poses"])
        near_depth = None
        if self.near_depth_map is not None:
            near_depth = self.near_depth_map.get(raw["video_id"])
        poses = scale_translation(poses, near_depth, self.trajectory_scale)

        result: dict[str, Any] = {
            "video_id": raw["video_id"],
            "video": video,
            "pose": torch.from_numpy(poses),
            "x_fov": float(raw["x_fov"]),
            "xi": float(raw.get("xi", 0.0)),
            "caption": str(raw["caption"]).strip(),
        }
        if self.return_first_frame:
            first = video[:, 0, :, :]
            result["first_frame_image"] = first
            result["first_frame_pil"] = TF.to_pil_image(torch.clamp((first + 1.0) / 2.0, 0, 1))
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(self.max_retries):
            try:
                return self._finalize(self._load_raw(index))
            except Exception as error:  # noqa: BLE001 - skip corrupt clip, try next
                last_error = error
                print(f"[{type(self).__name__}] skipping index {index}: {error}")
                index = (index + 1) % len(self)
        raise RuntimeError(
            f"{type(self).__name__}: failed after {self.max_retries} retries: {last_error}"
        )
