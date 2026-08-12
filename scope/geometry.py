"""Camera-ray and Plücker-coordinate utilities used by SCoPE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_camera_rays(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    patches_y: int,
    patches_x: int,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-space ray origins and directions on the token grid.

    Args:
        c2w: Camera-to-world matrices, shaped ``[B, T, 4, 4]``.
        intrinsics: Pinhole intrinsics, shaped ``[B, T, 3, 3]``.
    """
    batch, frames = c2w.shape[:2]
    dtype = c2w.dtype
    device = c2w.device
    patch_width = image_width / patches_x
    patch_height = image_height / patches_y
    u = torch.linspace(
        0.5 * patch_width,
        image_width - 0.5 * patch_width,
        patches_x,
        device=device,
        dtype=dtype,
    )
    v = torch.linspace(
        0.5 * patch_height,
        image_height - 0.5 * patch_height,
        patches_y,
        device=device,
        dtype=dtype,
    )
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
    grid_u = grid_u[None, None].expand(batch, frames, -1, -1)
    grid_v = grid_v[None, None].expand(batch, frames, -1, -1)

    fx = intrinsics[..., 0, 0, None, None]
    fy = intrinsics[..., 1, 1, None, None]
    cx = intrinsics[..., 0, 2, None, None]
    cy = intrinsics[..., 1, 2, None, None]
    directions_camera = torch.stack(
        ((grid_u - cx) / fx, (grid_v - cy) / fy, torch.ones_like(grid_u)), dim=-1
    )  # [B, T, H, W, 3]
    directions_camera = F.normalize(directions_camera, dim=-1, eps=1e-8)

    rotation = c2w[..., :3, :3]
    translation = c2w[..., :3, 3]
    directions_world = torch.einsum("btij,bthwj->bthwi", rotation, directions_camera)
    directions_world = F.normalize(directions_world, dim=-1, eps=1e-8)
    origins_world = translation[..., None, None, :].expand_as(directions_world)
    return origins_world, directions_world


def compute_plucker_coordinates(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    patches_y: int,
    patches_x: int,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Return ``(direction, moment)`` coordinates shaped ``[B, S, 6]``."""
    origins, directions = compute_camera_rays(
        c2w,
        intrinsics,
        patches_y,
        patches_x,
        image_height,
        image_width,
    )
    moments = torch.cross(origins, directions, dim=-1)
    batch = c2w.shape[0]
    return torch.cat((directions.reshape(batch, -1, 3), moments.reshape(batch, -1, 3)), dim=-1)
