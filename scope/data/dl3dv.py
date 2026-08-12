"""DL3DV loader.

On-disk layout::

    <data_root>/<resolution>/<scene>/
        transforms.json      frames[] (OpenGL c2w) + fl_x/w/h
        wan2_caption.json     {"caption": ...}
        images_4/<name>.png   downsampled RGB frames

DL3DV's ``transforms.json`` uses the Nerfstudio/OpenGL convention (camera looks
along -Z). We convert each c2w to OpenCV (``c2w @ diag(1, -1, -1, 1)``) at the
loader boundary so downstream Plucker rays point the correct way. DL3DV clips
are pre-cut, so a fixed stride of 1 is used.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scope.data.common import BaseCameraVideoDataset, linspace_indices

_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


class DL3DVDataset(BaseCameraVideoDataset):
    def __init__(
        self,
        data_root: str,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        sample_stride: int = 1,
        random_sample: bool = True,
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
        self.sample_stride = sample_stride
        self.random_sample = random_sample

        scenes = self._scan_scenes()
        kept = self.filter_ids_by_near_depth([s.name for s in scenes])
        keep = set(kept)
        self.scenes = [s for s in scenes if s.name in keep]
        if max_videos is not None:
            self.scenes = self.scenes[:max_videos]
        if not self.scenes:
            raise ValueError(f"No DL3DV scenes found under {self.data_root}")
        print(f"[DL3DV] {len(self.scenes)} scenes")

    def _scan_scenes(self) -> list[Path]:
        index_path = self.data_root / "valid_video_dirs.json"
        if index_path.exists():
            return [Path(p) for p in json.loads(index_path.read_text(encoding="utf-8"))]
        scenes: list[Path] = []
        for res_dir in sorted(self.data_root.iterdir()):
            if not res_dir.is_dir():
                continue
            for scene in sorted(res_dir.iterdir()):
                if (
                    (scene / "transforms.json").exists()
                    and (scene / "wan2_caption.json").exists()
                    and (scene / "images_4").exists()
                ):
                    scenes.append(scene)
        try:
            index_path.write_text(json.dumps([str(p) for p in scenes]), encoding="utf-8")
        except OSError:
            pass
        return scenes

    def __len__(self) -> int:
        return len(self.scenes)

    def _sample_indices(self, total: int) -> list[int]:
        stride = self.sample_stride
        if not self.random_sample or total < self.num_frames * stride:
            return linspace_indices(total, self.num_frames)
        max_start = max(0, total - (self.num_frames - 1) * stride - 1)
        start = random.randint(0, max_start) if max_start > 0 else 0
        return [start + i * stride for i in range(self.num_frames)]

    def _load_raw(self, index: int) -> dict[str, Any]:
        scene = self.scenes[index]
        meta = json.loads((scene / "transforms.json").read_text(encoding="utf-8"))
        frames = meta["frames"]
        indices = self._sample_indices(len(frames))

        pil_frames = [
            Image.open(scene / "images_4" / Path(frames[i]["file_path"]).name).convert("RGB")
            for i in indices
        ]
        poses = np.stack(
            [
                (np.asarray(frames[i]["transform_matrix"], dtype=np.float32) @ _GL2CV)[:3]
                for i in indices
            ]
        )
        caption = json.loads((scene / "wan2_caption.json").read_text(encoding="utf-8")).get(
            "caption", ""
        )
        x_fov = 2.0 * np.arctan(meta["w"] / (2.0 * meta["fl_x"]))
        return {
            "video_id": scene.name,
            "frames": pil_frames,
            "poses": poses,
            "x_fov": float(x_fov),
            "xi": 0.0,
            "caption": caption,
        }
