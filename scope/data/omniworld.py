"""OmniWorld loader.

On-disk layout (per scene, under ``data_root``)::

    <scene>/
        color/<frame>.png                  RGB frames
        camera/split_<n>.json              focals / cx / cy
        droidclib/split_<n>.json           DROID-SLAM extrinsics (T, 4, 4)
        text/<start>_<end>.json            caption windows

An offline index (``scripts/build_omniworld_index.py``) lists one 81-frame
training window per entry. DROID-SLAM emits camera-from-world extrinsics, so we
invert them to camera-to-world. Intrinsics are given for the original
resolution and rescaled to the training width when computing FOV.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scope.data.common import BaseCameraVideoDataset


class OmniWorldDataset(BaseCameraVideoDataset):
    def __init__(
        self,
        data_root: str,
        index_path: str,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        pose_is_w2c: bool = True,
        caption_fields: tuple[str, ...] = (
            "Video_Caption",
            "Short_Caption",
            "Background_Caption",
        ),
        max_videos: int | None = None,
        near_depth_map: dict[str, float] | None = None,
        trajectory_scale: float = 1.0,
        return_first_frame: bool = True,
    ) -> None:
        super().__init__(
            num_frames=num_frames,
            height=height,
            width=width,
            near_depth_map=near_depth_map,
            trajectory_scale=trajectory_scale,
            return_first_frame=return_first_frame,
        )
        self.data_root = Path(data_root)
        self.pose_is_w2c = pose_is_w2c
        self.caption_fields = caption_fields

        index_file = Path(index_path)
        if not index_file.is_file():
            raise FileNotFoundError(
                f"OmniWorld index not found: {index_file}. Build it with "
                f"scripts/build_omniworld_index.py."
            )
        entries = json.loads(index_file.read_text(encoding="utf-8"))
        keep = set(self.filter_ids_by_near_depth([self._entry_id(e) for e in entries]))
        entries = [e for e in entries if self._entry_id(e) in keep]
        if max_videos is not None:
            entries = entries[:max_videos]
        self.entries = entries
        if not self.entries:
            raise ValueError(f"No OmniWorld windows found for {index_file}")
        print(f"[OmniWorld] {len(self.entries)} windows")

    @staticmethod
    def _entry_id(entry: dict[str, Any]) -> str:
        return f"{entry['scene']}_split{int(entry['split_idx'])}_{int(entry['frame_start']):06d}"

    def __len__(self) -> int:
        return len(self.entries)

    def _select_caption(self, captions: dict[str, Any]) -> str:
        for field in self.caption_fields:
            value = captions.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        parts = [v.strip() for v in captions.values() if isinstance(v, str) and v.strip()]
        return " ".join(parts)

    def _compute_x_fov(self, scene: str, split_idx: int) -> float:
        droid = json.loads(
            (self.data_root / scene / "droidclib" / f"split_{split_idx}.json").read_text()
        )
        intr = droid.get("orig_intrinsic") or droid.get("crop_intrinsic")
        if intr is not None:
            fx_orig, w_orig = float(intr["fx"]), float(intr["cx"]) * 2.0
        else:
            camera = json.loads(
                (self.data_root / scene / "camera" / f"split_{split_idx}.json").read_text()
            )
            focals = camera.get("focals", [])
            if not focals:
                return math.radians(60.0)
            fx_orig, w_orig = float(np.mean(focals)), float(camera.get("cx", 640.0)) * 2.0
        fx_train = fx_orig * (self.width / max(w_orig, 1.0))
        return 2.0 * math.atan(self.width / (2.0 * fx_train))

    def _load_raw(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        scene = entry["scene"]
        split_idx = int(entry["split_idx"])
        frame_start = int(entry["frame_start"])
        local_start = int(entry["split_local_start"])
        color_dir = self.data_root / scene / "color"
        pil_frames = [
            Image.open(color_dir / f"{i:06d}.png").convert("RGB")
            for i in range(frame_start, frame_start + self.num_frames)
        ]

        droid = json.loads(
            (self.data_root / scene / "droidclib" / f"split_{split_idx}.json").read_text()
        )
        extrinsics = np.asarray(droid["extrinsics"], dtype=np.float32)
        window = extrinsics[local_start : local_start + self.num_frames]
        poses = np.linalg.inv(window) if self.pose_is_w2c else window

        caption_obj = json.loads(
            (self.data_root / scene / entry["caption_file"]).read_text(encoding="utf-8")
        )
        caption = self._select_caption(caption_obj.get("captions", caption_obj))
        return {
            "video_id": self._entry_id(entry),
            "frames": pil_frames,
            "poses": poses[:, :3],
            "x_fov": self._compute_x_fov(scene, split_idx),
            "xi": 0.0,
            "caption": caption,
        }
