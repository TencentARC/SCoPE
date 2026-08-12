"""PanShot loader.

On-disk layout::

    <data_root>/
        captioned-<split>.jsonl        {"video": "<name>-fov<F>-xi<X>", "caption": ...}
        videos-<split>/<name>-fov<F>-xi<X>.mp4    81-frame RGB clip
        pose-<split>/<name>.npy                    (81, 3, 4) OpenCV c2w

The horizontal FOV (degrees) and unified-camera ``xi`` are encoded in the video
name suffix. PanShot clips are pre-cut to the target length, so frames are
linspace-sampled.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image

from scope.data.common import BaseCameraVideoDataset, linspace_indices

_FOV_XI_RE = re.compile(r"-fov([\d.]+)-xi([\d.]+)$")


def _parse_fov_xi(name: str) -> tuple[float, float]:
    match = _FOV_XI_RE.search(name)
    if match is None:
        raise ValueError(f"Cannot parse fov/xi from PanShot name: {name}")
    return float(np.radians(float(match.group(1)))), float(match.group(2))


class PanShotDataset(BaseCameraVideoDataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        pinhole_only: bool = True,
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
        video_dir = self.data_root / f"videos-{split}"
        pose_dir = self.data_root / f"pose-{split}"
        entries: list[tuple[str, str, Path, Path]] = []
        with (self.data_root / f"captioned-{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                name = record["video"]
                if pinhole_only and _parse_fov_xi(name)[1] > 0:
                    continue
                pose_key = _FOV_XI_RE.sub("", name)
                mp4 = video_dir / f"{name}.mp4"
                pose = pose_dir / f"{pose_key}.npy"
                if mp4.exists() and pose.exists():
                    entries.append((name, record["caption"], mp4, pose))

        keep = set(self.filter_ids_by_near_depth([e[0] for e in entries]))
        entries = [e for e in entries if e[0] in keep]
        if max_videos is not None:
            entries = entries[:max_videos]
        self.entries = entries
        if not self.entries:
            raise ValueError(f"No PanShot clips found under {self.data_root}")
        print(f"[PanShot/{split}] {len(self.entries)} clips")

    def __len__(self) -> int:
        return len(self.entries)

    def _load_raw(self, index: int) -> dict[str, Any]:
        name, caption, mp4_path, pose_path = self.entries[index]
        raw_frames = list(iio.imiter(mp4_path))
        indices = linspace_indices(len(raw_frames), self.num_frames)
        pil_frames = [Image.fromarray(raw_frames[i]) for i in indices]

        poses = np.load(pose_path).astype(np.float32)
        if poses.shape[0] != self.num_frames:
            pose_indices = linspace_indices(poses.shape[0], self.num_frames)
            poses = poses[pose_indices]
        x_fov, xi = _parse_fov_xi(name)
        return {
            "video_id": name,
            "frames": pil_frames,
            "poses": poses,
            "x_fov": x_fov,
            "xi": xi,
            "caption": caption,
        }
