"""Wan self-attention with SCoPE Normalize-Gate-Inject encoding."""

import torch

from diffsynth.models.wan_video_dit import (
    SelfAttention,
    modulate,
    rope_apply,
)
from scope.encoding import SightlineCoordinatePE


class SelfAttentionWithSCoPE(SelfAttention):
    """
    Self-Attention with Normalize-Gate-Inject Plücker PE.

    All layers get scale-gated Q/K PE. Layers with enable_cam_residual=True
    also add a gated frame-uniform camera residual.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        plucker_init: str = "zero",
        plucker_init_scale: float = 0.01,
        plucker_mlp_hidden: int = 0,
        plucker_scale: float = 0.0,
        gate_init_bias: float = -2.0,
        disable_spatial_rope: bool = False,
        enable_cam_residual: bool = True,
        scale_gate_hidden: int = 0,
        log_scale_aug_prob: float = 0.0,
        log_scale_aug_range: tuple = (-1.2, 1.6),
    ):
        super().__init__(dim, num_heads, eps)
        self.disable_spatial_rope = disable_spatial_rope
        self.plucker_pe = SightlineCoordinatePE(
            dim=dim,
            plucker_init=plucker_init,
            plucker_init_scale=plucker_init_scale,
            plucker_mlp_hidden=plucker_mlp_hidden,
            plucker_scale=plucker_scale,
            gate_init_bias=gate_init_bias,
            enable_cam_residual=enable_cam_residual,
            scale_gate_hidden=scale_gate_hidden,
            log_scale_aug_prob=log_scale_aug_prob,
            log_scale_aug_range=log_scale_aug_range,
        )

    def _mask_spatial_rope(self, freqs: torch.Tensor) -> torch.Tensor:
        head_dim = self.dim // self.num_heads
        half_head = head_dim // 2
        f_dim = half_head - 2 * (half_head // 3)
        masked = freqs.clone()
        masked[..., f_dim:] = 1.0
        return masked

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor | None = None,
        control_camera_dit_input: dict | None = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        if freqs is not None:
            if self.disable_spatial_rope:
                freqs = self._mask_spatial_rope(freqs)
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)

        cam_residual = None
        if control_camera_dit_input is not None and "plucker_6d" in control_camera_dit_input:
            q, k, cam_residual = self.plucker_pe.apply_to_qk_and_output(
                q,
                k,
                control_camera_dit_input["plucker_6d"],
                num_frames=control_camera_dit_input.get("num_frames", 1),
            )

        if cam_residual is not None:
            v = v + cam_residual
        out = self.attn(q, k, v)
        return self.o(out)


def create_scope_block_forward():
    """Return a DiT block forward method that passes SCoPE coordinates."""

    def forward_scope(self, x, context, t_mod, freqs, control_camera_dit_input=None):
        has_seq = t_mod.dim() == 4
        chunk_dim = 2 if has_seq else 1

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)

        if has_seq:
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            shift_mlp = shift_mlp.squeeze(2)
            scale_mlp = scale_mlp.squeeze(2)
            gate_mlp = gate_mlp.squeeze(2)

        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        residual = self.self_attn(
            input_x,
            freqs=freqs,
            control_camera_dit_input=control_camera_dit_input,
        )

        x = self.gate(x, gate_msa, residual)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))

        return x

    return forward_scope
