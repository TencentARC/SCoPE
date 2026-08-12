"""Load the complete SCoPE inference model from sharded weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from diffsynth.models import ModelManager
from diffsynth.models.utils import init_weights_on_device
from diffsynth.models.wan_video_dit import WanModel
from scope.config import SCOPE_MODEL_ID, ArchitectureConfig, InferenceConfig
from scope.patch import patch_scope, validate_official_low_expert
from scope.pipeline import SCoPEPipeline

_DIT_CONFIG: dict[str, Any] = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 36,
    "dim": 5120,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 40,
    "num_layers": 40,
    "eps": 1e-6,
    "require_clip_embedding": False,
}


def resolve_model_dir(source: str = SCOPE_MODEL_ID, cache_dir: Path | None = None) -> Path:
    """Resolve a local complete model directory or download it from Hugging Face."""
    local = Path(source).expanduser()
    if local.is_dir():
        return local.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=source,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            allow_patterns=[
                "high_noise_model/*",
                "low_noise_model/*",
                "models_t5_umt5-xxl-enc-bf16.pth",
                "Wan2.1_VAE.pth",
                "google/umt5-xxl/*",
                "model_index.json",
            ],
        )
    )


def _component_shards(component_dir: Path) -> list[Path]:
    index_path = component_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.is_file():
        single = component_dir / "diffusion_pytorch_model.safetensors"
        if single.is_file():
            return [single]
        raise FileNotFoundError(f"Missing SCoPE weights in {component_dir}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid safetensors index: {index_path}")
    return [component_dir / name for name in dict.fromkeys(weight_map.values())]


def _load_complete_component(model: torch.nn.Module, component_dir: Path) -> None:
    expected = set(model.state_dict())
    loaded: set[str] = set()
    for shard_path in _component_shards(component_dir):
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
        shard = load_file(str(shard_path), device="cpu")
        duplicate = loaded.intersection(shard)
        if duplicate:
            raise ValueError(f"Duplicate keys in {shard_path.name}: {sorted(duplicate)[:5]}")
        unexpected = set(shard).difference(expected)
        if unexpected:
            raise ValueError(f"Unexpected keys in {shard_path.name}: {sorted(unexpected)[:5]}")
        model.load_state_dict(shard, strict=False, assign=True)
        loaded.update(shard)

    missing = expected.difference(loaded)
    if missing:
        raise ValueError(f"Incomplete component {component_dir.name}: {sorted(missing)[:5]}")
    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(f"Unmaterialized parameters: {meta_parameters[:5]}")


def _install_scope_architecture(pipe: SCoPEPipeline, config: InferenceConfig) -> None:
    arch = ArchitectureConfig()
    patch_scope(
        pipe,
        method="scope",
        height=config.height,
        width=config.width,
        plucker_init=arch.plucker_init,
        plucker_init_scale=arch.plucker_init_scale,
        plucker_mlp_hidden=arch.plucker_mlp_hidden,
        plucker_scale=arch.plucker_scale,
        gate_init_bias=arch.gate_init_bias,
        cam_residual_layers=[] if not arch.use_camera_residual else None,
        scale_gate_hidden=arch.scale_gate_hidden,
    )
    pipe.dit.plucker_normalize_moment = arch.normalize_moment
    pipe.dit2.plucker_normalize_moment = arch.normalize_moment


def load_pipeline(model_dir: Path, config: InferenceConfig) -> SCoPEPipeline:
    """Load every inference component without consulting the Wan base repository."""
    pipe = SCoPEPipeline(device="cpu", torch_dtype=torch.bfloat16)
    with init_weights_on_device():
        pipe.dit = WanModel(**_DIT_CONFIG)
        pipe.dit2 = WanModel(**_DIT_CONFIG)
        _install_scope_architecture(pipe, config)

    _load_complete_component(pipe.dit, model_dir / "high_noise_model")
    _load_complete_component(pipe.dit2, model_dir / "low_noise_model")
    validate_official_low_expert(pipe.dit2)

    manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    manager.load_model(str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth"))
    manager.load_model(str(model_dir / "Wan2.1_VAE.pth"))
    pipe.text_encoder = manager.fetch_model("wan_video_text_encoder")
    pipe.vae = manager.fetch_model("wan_video_vae")
    if pipe.text_encoder is None or pipe.vae is None:
        raise RuntimeError("The complete SCoPE package must contain both T5 and VAE weights")

    tokenizer_dir = model_dir / "google" / "umt5-xxl"
    pipe.prompter.fetch_models(pipe.text_encoder)
    pipe.prompter.fetch_tokenizer(str(tokenizer_dir))
    pipe.height_division_factor = pipe.vae.upsampling_factor * 2
    pipe.width_division_factor = pipe.vae.upsampling_factor * 2
    pipe.switch_DiT_boundary = config.switch_dit_boundary
    return pipe
