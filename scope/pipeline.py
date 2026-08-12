"""Wan2.2-A14B inference pipeline with SCoPE camera conditioning."""

from __future__ import annotations

import torch

from diffsynth.pipelines.wan_video_panshot import (
    WanVideoPipeline as BaseWanVideoPipeline,
)
from diffsynth.pipelines.wan_video_panshot import (
    WanVideoUnit_ImageEmbedderVAE,
)
from scope.camera import SCoPECameraUnit


class SCoPEPipeline(BaseWanVideoPipeline):
    """Wan2.2 dual-expert pipeline with token-aligned camera coordinates."""

    _CAMERA_KEYS = ("control_camera_dit_input", "control_camera_latents_input")

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        self.units[-1] = SCoPECameraUnit()

    def build_i2v_conditioning(
        self,
        input_image,
        num_frames: int,
        height: int,
        width: int,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> torch.Tensor | None:
        """Build the Wan2.2 I2V mask and first-frame VAE conditioning tensor."""
        if not getattr(self.dit, "require_vae_embedding", False):
            return None
        unit = WanVideoUnit_ImageEmbedderVAE()
        outputs = unit.process(
            pipe=self,
            input_image=input_image,
            end_image=None,
            num_frames=num_frames,
            height=height,
            width=width,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return outputs.get("y")

    def __call__(self, *args, camera_cfg_scale=1.0, **kwargs):
        """Run inference, optionally applying classifier-free guidance to camera inputs."""
        if camera_cfg_scale is None or camera_cfg_scale == 1.0:
            return super().__call__(*args, **kwargs)

        original_model_fn = self.model_fn

        def model_fn_with_camera_cfg(**fn_kwargs):
            camera_inputs = {
                key: fn_kwargs.pop(key) for key in self._CAMERA_KEYS if key in fn_kwargs
            }
            if not camera_inputs:
                return original_model_fn(**fn_kwargs)
            noise_without_camera = original_model_fn(**fn_kwargs)
            fn_kwargs.update(camera_inputs)
            noise_with_camera = original_model_fn(**fn_kwargs)
            return noise_without_camera + camera_cfg_scale * (
                noise_with_camera - noise_without_camera
            )

        self.model_fn = model_fn_with_camera_cfg
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self.model_fn = original_model_fn
