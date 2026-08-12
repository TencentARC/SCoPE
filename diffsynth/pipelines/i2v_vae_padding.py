"""
I2V VAE 未知帧像素填充策略（仅推理/实验 opt-in）

Wan 官方 image2video.py 对未知帧使用 ``torch.zeros``，张量值在 **[-1, 1]**
归一化空间里为 **0**，对应 RGB 约 **128 的中灰**，不是 uint8 意义的黑 (0→-1)。

本模块供 ``WanVideoUnit_ImageEmbedderVAE`` 在 ``pipe.i2v_vae_unknown_frame_fill``
被设置时选用；未设置时行为与上游完全一致。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

# ImageNet 均值 (RGB, [0,1])，映射到 preprocess_image 的 [-1, 1]
_IMAGENET_MEAN_01: Tuple[float, float, float] = (0.485, 0.456, 0.406)

FILL_MODE_ALIASES = {
    None: "official_zero",
    "official": "official_zero",
    "zero": "official_zero",
    "official_zero": "official_zero",
    "black": "black",
    "gray": "official_zero",
    "mid_gray": "official_zero",
    "first_frame_mean": "first_frame_mean",
    "imagenet_mean": "imagenet_mean",
}

FILL_MODE_DESCRIPTIONS = {
    "official_zero": "官方 Wan: torch.zeros → 归一化空间 0 ≈ RGB128 中灰",
    "black": "归一化空间 -1 ≈ RGB0 真黑（常被误称为 zero pixel）",
    "first_frame_mean": "首帧逐通道均值铺满未知帧",
    "imagenet_mean": "ImageNet RGB 均值映射到 [-1,1]",
}


def normalize_fill_mode(mode: Optional[str]) -> str:
    if mode not in FILL_MODE_ALIASES:
        known = sorted({k for k in FILL_MODE_ALIASES if k is not None})
        raise ValueError(f"Unknown i2v_vae_unknown_frame_fill={mode!r}. Known: {known}")
    return FILL_MODE_ALIASES[mode]


def resolve_padding_mode(pipe) -> Optional[str]:
    return getattr(pipe, "i2v_vae_unknown_frame_fill", None)


def _rgb01_to_normalized(v: float) -> float:
    return v * 2.0 - 1.0


def make_vae_unknown_frames(
    num_pad_frames: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    mode: Optional[str] = None,
    first_frame_chw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    构造 VAE encode 用的未知帧像素块 ``[3, T_pad, H, W]``（与首帧同 dtype/device）。

    Args:
        num_pad_frames: 待填充帧数 (通常 num_frames-1 或 num_frames-2)
        first_frame_chw: 已 preprocess 的首帧 ``[3, H, W]``，``first_frame_mean`` 需要
    """
    if num_pad_frames <= 0:
        return torch.empty(3, 0, height, width, device=device, dtype=dtype)

    key = normalize_fill_mode(mode)
    shape = (3, num_pad_frames, height, width)

    if key == "official_zero":
        return torch.zeros(shape, device=device, dtype=dtype)

    if key == "black":
        return torch.full(shape, -1.0, device=device, dtype=dtype)

    if key == "first_frame_mean":
        if first_frame_chw is None:
            raise ValueError("first_frame_mean requires first_frame_chw")
        fm = first_frame_chw
        if fm.dim() != 3 or fm.shape[0] != 3:
            raise ValueError(f"first_frame_chw must be [3,H,W], got {tuple(fm.shape)}")
        mean = fm.reshape(3, -1).mean(dim=1).view(3, 1, 1, 1)
        return mean.expand(shape).contiguous()

    if key == "imagenet_mean":
        vals = torch.tensor(
            [_rgb01_to_normalized(v) for v in _IMAGENET_MEAN_01],
            device=device,
            dtype=dtype,
        )
        return vals.view(3, 1, 1, 1).expand(shape).contiguous()

    raise RuntimeError(f"Unhandled fill mode: {key}")
