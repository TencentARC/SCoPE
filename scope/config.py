"""Public SCoPE inference defaults matching the released 32k checkpoint."""

from __future__ import annotations

from dataclasses import dataclass

SCOPE_MODEL_ID = "TencentARC/SCoPE"


@dataclass(frozen=True)
class InferenceConfig:
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16
    num_inference_steps: int = 40
    sigma_shift: float = 5.0
    cfg_scale: float = 3.5
    switch_dit_boundary: float = 0.9
    seed: int = 42


@dataclass(frozen=True)
class ArchitectureConfig:
    plucker_init: str = "zero"
    plucker_init_scale: float = 0.01
    plucker_mlp_hidden: int = 64
    plucker_scale: float = 1.0
    gate_init_bias: float = -2.0
    scale_gate_hidden: int = 0
    use_camera_residual: bool = False
    normalize_moment: bool = False
