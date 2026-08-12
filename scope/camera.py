"""Pipeline unit that converts camera trajectories into SCoPE coordinates."""

from __future__ import annotations

from typing import Any

import torch
from einops import repeat

from diffsynth.utils import PipelineUnit
from scope.geometry import compute_plucker_coordinates


class SCoPECameraUnit(PipelineUnit):
    """Build token-aligned sightline coordinates from a pinhole camera path."""

    def __init__(self) -> None:
        super().__init__(input_params=("height", "width", "camera_control_panshot"))

    def process(
        self,
        pipe: Any,
        height: int,
        width: int,
        camera_control_panshot: dict[str, Any] | None,
    ) -> dict[str, dict[str, torch.Tensor | int]]:
        if camera_control_panshot is None:
            return {}
        if getattr(pipe.dit, "camera_condition", "none") != "scope":
            return {}

        pose = camera_control_panshot["pose"][:, ::4]  # [B, T_latent, 3, 4]
        x_fov = camera_control_panshot["x_fov"]
        if not torch.is_tensor(x_fov):
            x_fov = torch.tensor([float(x_fov)], device=pose.device, dtype=pose.dtype)
        x_fov = x_fov.to(device=pose.device, dtype=pose.dtype).reshape(-1)

        c2w = torch.eye(4, device=pose.device, dtype=pose.dtype)
        c2w = repeat(c2w, "i j -> b t i j", b=pose.shape[0], t=pose.shape[1]).clone()
        c2w[..., :3, :4] = pose

        patch_factor = pipe.vae.upsampling_factor * 2
        patches_x = width // patch_factor
        patches_y = height // patch_factor
        focal = (width * 0.5) / torch.tan(x_fov * 0.5)
        intrinsics = torch.zeros(
            (pose.shape[0], pose.shape[1], 3, 3), device=pose.device, dtype=pose.dtype
        )
        intrinsics[..., 0, 0] = focal[:, None]
        intrinsics[..., 1, 1] = focal[:, None]
        intrinsics[..., 0, 2] = width * 0.5
        intrinsics[..., 1, 2] = height * 0.5
        intrinsics[..., 2, 2] = 1.0

        coordinates = compute_plucker_coordinates(
            c2w,
            intrinsics,
            patches_y,
            patches_x,
            height,
            width,
        )
        return {
            "control_camera_dit_input": {
                "plucker_6d": coordinates,
                "num_frames": pose.shape[1],
            }
        }
