"""RealEstate10K loader.

On-disk layout::

    <data_root>/
        <split>_caption.json          {video_id: caption}
        <split>.txt                    one video_id per line
        process/<split>/<batch>/<video_id>/
            transforms.json            frames[] + fl_x/fl_y/cx/cy/w/h
            <frame>.png                RGB frames referenced by file_path

``transform_matrix`` is read as OpenCV camera-to-world. RealEstate10K is
high-frame-rate, so training samples a random stride in ``[1, sample_stride]``
to expose a range of motion magnitudes.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scope.data.common import BaseCameraVideoDataset, linspace_indices


class RealEstate10KDataset(BaseCameraVideoDataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        sample_stride: int = 4,
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
        self.split = split
        self.sample_stride = sample_stride
        self.random_sample = random_sample

        self.captions: dict[str, str] = json.loads(
            (self.data_root / f"{split}_caption.json").read_text(encoding="utf-8")
        )
        listed = [
            line.strip().replace(".txt", "")
            for line in (self.data_root / f"{split}.txt").read_text().splitlines()
            if line.strip()
        ]
        self.clip_dirs = self._resolve_clip_dirs([vid for vid in listed if vid in self.captions])
        ids = self.filter_ids_by_near_depth(list(self.clip_dirs))
        if max_videos is not None:
            ids = ids[:max_videos]
        self.video_ids = ids
        if not self.video_ids:
            raise ValueError(f"No RealEstate10K clips found under {self.data_root}")
        print(f"[RealEstate10K/{split}] {len(self.video_ids)} clips")

    def _resolve_clip_dirs(self, ids: list[str]) -> dict[str, Path]:
        process_dir = self.data_root / "process" / self.split
        index_path = process_dir / f"{self.split}_index.json"
        if index_path.exists():
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            return {k: Path(v) for k, v in raw.items() if k in set(ids)}
        target = set(ids)
        mapping: dict[str, Path] = {}
        for batch_dir in sorted(process_dir.glob("*")):
            if not batch_dir.is_dir():
                continue
            for entry in batch_dir.iterdir():
                if entry.name in target and (entry / "transforms.json").exists():
                    mapping[entry.name] = entry
        try:
            index_path.write_text(
                json.dumps({k: str(v) for k, v in mapping.items()}), encoding="utf-8"
            )
        except OSError:
            pass
        return mapping

    def __len__(self) -> int:
        return len(self.video_ids)

    def _sample_indices(self, total: int) -> list[int]:
        if not self.random_sample or total < self.num_frames:
            return linspace_indices(total, self.num_frames)
        max_stride = max(1, total // self.num_frames)
        stride = random.randint(1, min(self.sample_stride, max_stride))
        max_start = max(0, total - (self.num_frames - 1) * stride - 1)
        start = random.randint(0, max_start) if max_start > 0 else 0
        return [start + i * stride for i in range(self.num_frames)]

    def _load_raw(self, index: int) -> dict[str, Any]:
        video_id = self.video_ids[index]
        clip_dir = self.clip_dirs[video_id]
        meta = json.loads((clip_dir / "transforms.json").read_text(encoding="utf-8"))
        frames = meta["frames"]
        indices = self._sample_indices(len(frames))

        pil_frames = [Image.open(clip_dir / frames[i]["file_path"]).convert("RGB") for i in indices]
        poses = np.stack(
            [np.asarray(frames[i]["transform_matrix"], dtype=np.float32)[:3] for i in indices]
        )
        x_fov = 2.0 * np.arctan(meta["w"] / (2.0 * meta["fl_x"]))
        return {
            "video_id": video_id,
            "frames": pil_frames,
            "poses": poses,
            "x_fov": float(x_fov),
            "xi": 0.0,
            "caption": self.captions[video_id],
        }
