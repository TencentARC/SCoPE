"""Install SCoPE attention in both Wan2.2-A14B experts."""

import torch

from scope.modeling import SelfAttentionWithSCoPE, create_scope_block_forward


def _patch_expert(
    dit,
    method: str,
    height: int,
    width: int,
    copy_self_attn_weights: bool,
    plucker_init: str,
    plucker_init_scale: float,
    plucker_mlp_hidden: int,
    plucker_scale: float,
    gate_init_bias: float,
    disable_spatial_rope: bool,
    cam_residual_layers: list[int] | None,
    scale_gate_hidden: int,
    log_scale_aug_prob: float,
    log_scale_aug_range: tuple,
    log_prefix: str = "[SCoPE]",
):
    """Replace self-attention in one DiT expert."""
    dit.camera_condition = method
    num_blocks = len(dit.blocks)

    if cam_residual_layers is None:
        cam_residual_set = set(range(num_blocks))
        layer_desc = "all"
    else:
        cam_residual_set = set(cam_residual_layers)
        layer_desc = str(sorted(cam_residual_set))

    print(
        f"{log_prefix} method={method}, plucker_init={plucker_init}, "
        f"plucker_init_scale={plucker_init_scale}, "
        f"plucker_mlp_hidden={plucker_mlp_hidden}, plucker_scale={plucker_scale}, "
        f"gate_init_bias={gate_init_bias}, scale_gate_hidden={scale_gate_hidden}, "
        f"log_scale_aug_prob={log_scale_aug_prob}, "
        f"log_scale_aug_range={log_scale_aug_range}, "
        f"disable_spatial_rope={disable_spatial_rope}, "
        f"cam_residual_layers={layer_desc} ({len(cam_residual_set)}/{num_blocks} blocks)"
    )

    for i, block in enumerate(dit.blocks):
        original_attn = block.self_attn
        enable_cam_residual = i in cam_residual_set

        new_attn = SelfAttentionWithSCoPE(
            dim=dit.dim,
            num_heads=block.num_heads,
            eps=1e-6,
            plucker_init=plucker_init,
            plucker_init_scale=plucker_init_scale,
            plucker_mlp_hidden=plucker_mlp_hidden,
            plucker_scale=plucker_scale,
            gate_init_bias=gate_init_bias,
            disable_spatial_rope=disable_spatial_rope,
            enable_cam_residual=enable_cam_residual,
            scale_gate_hidden=scale_gate_hidden,
            log_scale_aug_prob=log_scale_aug_prob,
            log_scale_aug_range=log_scale_aug_range,
        )

        if copy_self_attn_weights:
            new_attn.q.weight.data = original_attn.q.weight.data.clone()
            if original_attn.q.bias is not None and new_attn.q.bias is not None:
                new_attn.q.bias.data = original_attn.q.bias.data.clone()
            new_attn.k.weight.data = original_attn.k.weight.data.clone()
            if original_attn.k.bias is not None and new_attn.k.bias is not None:
                new_attn.k.bias.data = original_attn.k.bias.data.clone()
            new_attn.v.weight.data = original_attn.v.weight.data.clone()
            if original_attn.v.bias is not None and new_attn.v.bias is not None:
                new_attn.v.bias.data = original_attn.v.bias.data.clone()
            new_attn.o.weight.data = original_attn.o.weight.data.clone()
            if original_attn.o.bias is not None and new_attn.o.bias is not None:
                new_attn.o.bias.data = original_attn.o.bias.data.clone()
            new_attn.norm_q.weight.data = original_attn.norm_q.weight.data.clone()
            new_attn.norm_k.weight.data = original_attn.norm_k.weight.data.clone()

        block.self_attn = new_attn

    forward_fn = create_scope_block_forward()
    for block in dit.blocks:
        block.forward = forward_fn.__get__(block, block.__class__)


def patch_scope(
    pipe,
    method,
    height,
    width,
    copy_self_attn_weights: bool = True,
    plucker_init: str = "zero",
    plucker_init_scale: float = 0.01,
    plucker_mlp_hidden: int = 0,
    plucker_scale: float = 0.0,
    gate_init_bias: float = -2.0,
    disable_spatial_rope: bool = False,
    cam_residual_layers: list[int] | None = None,
    scale_gate_hidden: int = 0,
    log_scale_aug_prob: float = 0.0,
    log_scale_aug_range: tuple = (-1.2, 1.6),
    **kwargs,
):
    """Patch the high- and low-noise experts and return trainable key patterns."""
    if getattr(pipe, "dit2", None) is None:
        raise RuntimeError("SCoPE requires both Wan2.2-A14B experts; low-noise expert is missing.")

    common_kwargs = {
        "method": method,
        "height": height,
        "width": width,
        "copy_self_attn_weights": copy_self_attn_weights,
        "plucker_init": plucker_init,
        "plucker_init_scale": plucker_init_scale,
        "plucker_mlp_hidden": plucker_mlp_hidden,
        "plucker_scale": plucker_scale,
        "gate_init_bias": gate_init_bias,
        "disable_spatial_rope": disable_spatial_rope,
        "cam_residual_layers": cam_residual_layers,
        "scale_gate_hidden": scale_gate_hidden,
        "log_scale_aug_prob": log_scale_aug_prob,
        "log_scale_aug_range": log_scale_aug_range,
    }
    _patch_expert(pipe.dit, log_prefix="[SCoPE/dit]", **common_kwargs)
    _patch_expert(pipe.dit2, log_prefix="[SCoPE/dit2]", **common_kwargs)

    keywords = ["plucker_pe", "self_attn", "norm3", "ffn"]
    return keywords


def validate_official_low_expert(dit) -> None:
    """Verify that the patched low-noise expert remains an exact zero-delta model."""
    for index, block in enumerate(dit.blocks):
        positional_encoding = block.self_attn.plucker_pe
        if positional_encoding.enable_cam_residual:
            raise RuntimeError(f"Low expert block {index} has camera residual enabled")
        q_output = (
            positional_encoding.eq[2] if positional_encoding.use_mlp else positional_encoding.eq
        )
        k_output = (
            positional_encoding.ek[2] if positional_encoding.use_mlp else positional_encoding.ek
        )
        if torch.count_nonzero(q_output.weight).item() != 0:
            raise RuntimeError(f"Low expert block {index} has non-zero SCoPE query weights")
        if torch.count_nonzero(k_output.weight).item() != 0:
            raise RuntimeError(f"Low expert block {index} has non-zero SCoPE key weights")


def enable_scope_grad(pipe, keywords, expert: str = "high_noise_model"):
    """Enable gradients for the selected expert; the released run trained high noise only."""
    pipe.eval()
    pipe.requires_grad_(False)

    if getattr(pipe, "dit2", None) is None:
        raise RuntimeError("SCoPE training requires both Wan2.2-A14B experts.")

    if expert == "high_noise_model":
        targets = [(pipe.dit, "")]
    elif expert == "low_noise_model":
        targets = [(pipe.dit2, "[dit2] ")]
    elif expert == "both":
        targets = [(pipe.dit, ""), (pipe.dit2, "[dit2] ")]
    else:
        raise ValueError(f"Unknown expert selection: {expert}")

    if keywords == "*":
        for dit, _ in targets:
            dit.train()
            dit.requires_grad_(True)
    else:
        for dit, prefix in targets:
            for name, module in dit.named_modules():
                if any(keyword in name for keyword in keywords):
                    print(f"Trainable: {prefix}{name}")
                    module.train()
                    module.requires_grad_(True)
            # DiTBlock.modulation / Head.modulation 是直接挂在 block/head 上的
            # nn.Parameter (非子 Module), named_modules 匹配不到, 需按 param 名补开.
            for name, param in dit.named_parameters():
                if not param.requires_grad and any(keyword in name for keyword in keywords):
                    print(f"Trainable param: {prefix}{name}")
                    param.requires_grad = True

    trainable_params = 0
    seen_params = set()
    for dit, _ in targets:
        for _, module in dit.named_modules():
            for param in module.parameters():
                if param.requires_grad and id(param) not in seen_params:
                    trainable_params += param.numel()
                    seen_params.add(id(param))
    print(f"Total number of trainable parameters (dit+dit2): {trainable_params:,}")
