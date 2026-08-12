"""
SCoPE: Sightline-Coordinate Positional Encoding — Normalize-Gate-Inject

用于解决跨数据集 scale 不一致导致的数值不稳定问题。

核心机制：
  1. 将 6D Plücker 坐标 (d, m) 解耦为归一化几何 (d, m̂) + 对数尺度 log‖m‖
     → 投影输入变为 7D，几何方向 scale-invariant
  2. 增加 scale_gate: log_scale → sigmoid(MLP) ∈ (0,1)
     → 动态调节 PE 注入强度：近景强、远景弱
  3. log_scale 同时参与 E_q / E_k 投影，不丢失绝对距离信息
  4. PE 输出经 RMSNorm 归一化，与 content path 的 QKNorm 对称
     → α 直接控制 geometry/content 的相对比例

数学形式：
    d_i, m̂_i, s_i = decompose(r_i)           # d 不变, m̂=m/‖m‖, s=log‖m‖
    pe_q_i = gate(s_i) · α_q · RMSNorm(E_q(d_i, m̂_i, s_i))
    pe_k_j = gate(s_j) · α_k · RMSNorm(E_k(m̂_j, d_j, s_j))   ← flip (d,m̂)
    q_i = QKNorm(W_Q x_i) + pe_q_i
    k_j = QKNorm(W_K x_j) + pe_k_j

α=1.0 时 geometry 与 content 等权参与 attention。

Usage:
    pe = SightlineCoordinatePE(dim=1536, plucker_init="zero", plucker_scale=1.0)
    q, k = pe.apply_to_qk(q, k, plucker_6d)
    # or with cam_residual:
    q, k, cam_res = pe.apply_to_qk_and_output(q, k, plucker_6d, num_frames=21)
"""

import torch
from torch import nn


class _RMSNorm(nn.Module):
    """Per-token RMSNorm with learnable scale (matches WAN's QKNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SightlineCoordinatePE(nn.Module):
    """
    Sightline-coordinate positional encoding with Normalize-Gate-Inject.

    Args:
        dim: attention feature dimension (e.g. 3072 for 5B).
        plucker_init: "zero" or "small" for E_q/E_k initialization.
        plucker_init_scale: std for "small" init.
        plucker_mlp_hidden: if > 0, use 7→hidden→dim MLP; if 0, use 7→dim Linear.
        plucker_scale: if > 0, add learnable α_q/α_k initialized to this value.
            With PE RMSNorm, α=1.0 means geometry and content contribute equally.
        gate_init_bias: initial value for cam_residual gate logit (only used
            when enable_cam_residual=True).
        enable_cam_residual: whether to add frame-uniform gated camera residual.
        scale_gate_hidden: hidden dim of the scale gate MLP. Defaults to dim // 4.
        log_scale_aug_prob: probability of applying a uniform per-sample shift
            to the log_scale that feeds the scale_gate MLP during training.
            0.0 = disabled (backward compatible).  Only `scale_gate` input is
            perturbed; feat_q/feat_k (E_q/E_k inputs) keep the true log_scale.
            Only active when self.training is True.
        log_scale_aug_range: (lo, hi) tuple of the uniform shift range in
            natural-log units.  Default (-1.2, 1.6) spans roughly ÷3.3 … ×5.
    """

    def __init__(
        self,
        dim: int,
        plucker_init: str = "zero",
        plucker_init_scale: float = 0.01,
        plucker_mlp_hidden: int = 0,
        plucker_scale: float = 0.0,
        gate_init_bias: float = -2.0,
        enable_cam_residual: bool = True,
        scale_gate_hidden: int = 0,
        log_scale_aug_prob: float = 0.0,
        log_scale_aug_range: tuple = (-1.2, 1.6),
    ):
        super().__init__()
        self.dim = dim
        self.use_mlp = plucker_mlp_hidden > 0
        self.use_scale = plucker_scale > 0
        self.enable_cam_residual = enable_cam_residual
        self.log_scale_aug_prob = float(log_scale_aug_prob)
        self.log_scale_aug_range = (float(log_scale_aug_range[0]), float(log_scale_aug_range[1]))

        in_dim = 7  # (d(3), m̂(3), log_s(1))

        # ── Q/K geometric projections ────────────────────────────────────
        if self.use_mlp:
            self.eq = nn.Sequential(
                nn.Linear(in_dim, plucker_mlp_hidden, bias=False),
                nn.GELU(),
                nn.Linear(plucker_mlp_hidden, dim, bias=False),
            )
            self.ek = nn.Sequential(
                nn.Linear(in_dim, plucker_mlp_hidden, bias=False),
                nn.GELU(),
                nn.Linear(plucker_mlp_hidden, dim, bias=False),
            )
        else:
            self.eq = nn.Linear(in_dim, dim, bias=False)
            self.ek = nn.Linear(in_dim, dim, bias=False)

        # ── PE RMSNorm: align PE magnitude with content QKNorm ──────────
        self.norm_pe_q = _RMSNorm(dim)
        self.norm_pe_k = _RMSNorm(dim)

        # ── Scale gate: log_scale → (0, 1) per-dim ──────────────────────
        sg_hidden = scale_gate_hidden if scale_gate_hidden > 0 else max(dim // 4, 1)
        self.scale_gate = nn.Sequential(
            nn.Linear(1, sg_hidden),
            nn.SiLU(),
            nn.Linear(sg_hidden, dim),
            nn.Sigmoid(),
        )
        # init gate bias so output ≈ 0.5 at start (log_scale=0 → neutral)
        nn.init.zeros_(self.scale_gate[0].bias)
        nn.init.zeros_(self.scale_gate[2].bias)

        # ── Learnable per-layer scale α ──────────────────────────────────
        # Shape (1,) instead of () because FSDP refuses to shard 0-dim
        # parameters. Broadcasting `alpha * pe_q` is identical for both shapes.
        if self.use_scale:
            self.alpha_q = nn.Parameter(torch.tensor([plucker_scale]))
            self.alpha_k = nn.Parameter(torch.tensor([plucker_scale]))

        # Optional camera residual.────────────────────────────────────────────────────────────
        if self.enable_cam_residual:
            if self.use_mlp:
                self.ev = nn.Sequential(
                    nn.Linear(in_dim, plucker_mlp_hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(plucker_mlp_hidden, dim, bias=False),
                )
                self.gate_proj = nn.Sequential(
                    nn.Linear(in_dim, plucker_mlp_hidden, bias=True),
                    nn.GELU(),
                    nn.Linear(plucker_mlp_hidden, dim, bias=False),
                )
            else:
                self.ev = nn.Linear(in_dim, dim, bias=False)
                self.gate_proj = nn.Linear(in_dim, dim, bias=False)
            self.gate_logit = nn.Parameter(torch.full((dim,), gate_init_bias))

        self._init_weights(plucker_init, plucker_init_scale)

    # ─────────────────────────────────────────────────────────────────────
    # Backward-compat ckpt loading
    # ─────────────────────────────────────────────────────────────────────
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # alpha_q / alpha_k were 0-dim scalars in earlier checkpoints; FSDP
        # requires shape (1,). Promote legacy entries while loading.
        for name in ("alpha_q", "alpha_k"):
            key = prefix + name
            if key in state_dict and state_dict[key].dim() == 0:
                state_dict[key] = state_dict[key].view(1)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Weight init
    # ─────────────────────────────────────────────────────────────────────

    def _init_weights(self, mode: str, scale: float):
        if self.use_mlp:
            # For 2-layer MLP with zero init: only zero the OUTPUT layer.
            # Zeroing both layers creates dead gradients (h=GELU(0)=0 → ∂L/∂W=0).
            # The first layer keeps default kaiming init so hidden activations ≠ 0.
            qk_output_layers = [self.eq[2], self.ek[2]]
            qk_input_layers = [self.eq[0], self.ek[0]]
        else:
            qk_output_layers = [self.eq, self.ek]
            qk_input_layers = []

        for m in qk_output_layers:
            if mode == "zero":
                nn.init.zeros_(m.weight)
            else:
                nn.init.normal_(m.weight, 0.0, scale)

        for m in qk_input_layers:
            if mode == "zero":
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
            else:
                nn.init.normal_(m.weight, 0.0, scale)

        if self.enable_cam_residual:
            v_modules = [self.ev[0], self.ev[2]] if self.use_mlp else [self.ev]
            for m in v_modules:
                nn.init.normal_(m.weight, 0.0, scale)
            if self.use_mlp:
                nn.init.xavier_uniform_(self.gate_proj[0].weight)
                nn.init.zeros_(self.gate_proj[0].bias)
                nn.init.zeros_(self.gate_proj[2].weight)
            else:
                nn.init.zeros_(self.gate_proj.weight)

    # ─────────────────────────────────────────────────────────────────────
    # Plücker decomposition
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def decompose_plucker(plucker_6d: torch.Tensor):
        """Decompose (d, m) → (d, m̂, log‖m‖).

        Returns:
            feat_q: (B, S, 7) = (d, m̂, log_s) for Q projection.
            feat_k: (B, S, 7) = (m̂, d, log_s) for K projection (flip).
            log_scale: (B, S, 1) for scale gate.
        """
        d = plucker_6d[..., :3]
        m = plucker_6d[..., 3:]

        m_norm = m.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        m_hat = m / m_norm
        log_scale = torch.log(m_norm)

        feat_q = torch.cat([d, m_hat, log_scale], dim=-1)
        feat_k = torch.cat([m_hat, d, log_scale], dim=-1)  # flip d ↔ m̂
        return feat_q, feat_k, log_scale

    # ─────────────────────────────────────────────────────────────────────
    # Training-time scale augmentation
    # ─────────────────────────────────────────────────────────────────────

    def _maybe_perturb_log_scale(self, log_scale: torch.Tensor) -> torch.Tensor:
        """Apply a per-sample uniform shift to log_scale during training.

        The shift is shared across all tokens of a sample (same offset for
        all frames / patches), mimicking the effect of globally rescaling
        the camera translation (e.g. `poses[:, :, 3] *= k` → log_scale += log k).

        Only the copy fed into `scale_gate` is perturbed; E_q / E_k still see
        the true log_scale so absolute-distance information is preserved.

        No-op when:
          * not training, or
          * `log_scale_aug_prob <= 0`, or
          * the Bernoulli draw rejects this forward.
        """
        if not self.training or self.log_scale_aug_prob <= 0.0:
            return log_scale
        # Bernoulli(prob) gate — batch-wide single draw to minimise overhead.
        if torch.rand((), device=log_scale.device).item() > self.log_scale_aug_prob:
            return log_scale
        lo, hi = self.log_scale_aug_range
        B = log_scale.shape[0]
        # (B, 1, 1) broadcast over (S, 1) → per-sample scalar shift.
        shift = torch.empty(B, 1, 1, device=log_scale.device, dtype=log_scale.dtype).uniform_(
            lo, hi
        )
        return log_scale + shift

    # ─────────────────────────────────────────────────────────────────────
    # Core forward: Q/K only
    # ─────────────────────────────────────────────────────────────────────

    def apply_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        plucker_6d: torch.Tensor,
    ):
        """Add scale-gated Plücker PE to Q and K.

        Args:
            q: (B, S, D) query after RoPE.
            k: (B, S, D) key after RoPE.
            plucker_6d: (B, S, 6) raw Plücker coordinates (d, m).

        Returns:
            q, k with Normalize-Gate-Inject PE applied.
        """
        feat_q, feat_k, log_scale = self.decompose_plucker(plucker_6d)

        pe_q = self.norm_pe_q(self.eq(feat_q.to(q.dtype)))
        pe_k = self.norm_pe_k(self.ek(feat_k.to(k.dtype)))

        # Perturb the gate's log_scale input only — feat_q/feat_k keep the
        # true log_scale so E_q / E_k absolute-distance information stays intact.
        log_scale_for_gate = self._maybe_perturb_log_scale(log_scale)
        gate = self.scale_gate(log_scale_for_gate.to(q.dtype))  # (B, S, D)
        pe_q = gate * pe_q
        pe_k = gate * pe_k

        if self.use_scale:
            pe_q = self.alpha_q * pe_q
            pe_k = self.alpha_k * pe_k

        return q + pe_q, k + pe_k

    # ─────────────────────────────────────────────────────────────────────
    # Extended forward: Q/K + cam_residual
    # ─────────────────────────────────────────────────────────────────────

    def apply_to_qk_and_output(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        plucker_6d: torch.Tensor,
        num_frames: int = 1,
    ):
        """Apply Plücker PE to Q/K, optionally compute frame-uniform cam_residual.

        Args:
            q, k: (B, S, dim) where S = num_frames * H * W.
            plucker_6d: (B, S, 6) raw Plücker coordinates.
            num_frames: latent frame count T, for frame-level averaging.

        Returns:
            q, k: with Normalize-Gate-Inject PE applied.
            cam_residual: (B, S, dim) or None.
        """
        orig_dtype = q.dtype
        feat_q, feat_k, log_scale = self.decompose_plucker(plucker_6d)
        feat_q = feat_q.to(orig_dtype)
        feat_k = feat_k.to(orig_dtype)

        pe_q = self.norm_pe_q(self.eq(feat_q))
        pe_k = self.norm_pe_k(self.ek(feat_k))

        # Perturb the gate's log_scale input only — feat_q/feat_k keep the
        # true log_scale so E_q / E_k absolute-distance information stays intact.
        log_scale_for_gate = self._maybe_perturb_log_scale(log_scale)
        gate = self.scale_gate(log_scale_for_gate.to(orig_dtype))
        pe_q = gate * pe_q
        pe_k = gate * pe_k

        if self.use_scale:
            pe_q = self.alpha_q * pe_q
            pe_k = self.alpha_k * pe_k

        q = q + pe_q
        k = k + pe_k

        cam_residual = None
        if self.enable_cam_residual:
            B, S, C = feat_q.shape
            spatial = S // num_frames
            # frame-level average of normalized features
            feat_frame = feat_q.reshape(B, num_frames, spatial, C).mean(dim=2, keepdim=True)
            feat_frame = feat_frame.expand(B, num_frames, spatial, C).reshape(B, S, C)

            cam_gate = torch.sigmoid(self.gate_logit + self.gate_proj(feat_frame))
            cam_residual = cam_gate.to(orig_dtype) * self.ev(feat_frame)

        return q.to(orig_dtype), k.to(orig_dtype), cam_residual
