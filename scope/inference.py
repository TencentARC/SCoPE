"""Command-line inference for SCoPE on Wan2.2-I2V-A14B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from diffsynth import save_video
from scope.config import SCOPE_MODEL_ID, InferenceConfig
from scope.pipeline import SCoPEPipeline
from scope.weights import load_pipeline, resolve_model_dir


def _prepare_device(pipe: SCoPEPipeline, vram_limit_gb: float | None) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.eval()
    pipe.device = str(device)
    if device.type == "cuda":
        pipe.enable_vram_management(vram_limit=vram_limit_gb)
    else:
        pipe.to(device)
    for expert in (pipe.dit, pipe.dit2):
        for module in expert.modules():
            positional_encoding = getattr(module, "plucker_pe", None)
            if positional_encoding is None:
                continue
            for name in ("norm_pe_q", "norm_pe_k"):
                getattr(positional_encoding, name).to(device=device, dtype=pipe.torch_dtype)
            for name in ("alpha_q", "alpha_k", "gate_logit"):
                parameter = getattr(positional_encoding, name, None)
                if isinstance(parameter, torch.nn.Parameter):
                    parameter.data = parameter.data.to(device=device, dtype=pipe.torch_dtype)
    return device


def _select_example(
    manifest_path: Path,
    case_id: str | None,
    trajectory_id: str | None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    case = next(
        (item for item in cases if item["id"] == case_id),
        cases[0] if case_id is None else None,
    )
    if case is None:
        raise KeyError(f"Unknown case: {case_id}")
    trajectories = case["trajectories"]
    trajectory = next(
        (item for item in trajectories if item["id"] == trajectory_id),
        trajectories[0] if trajectory_id is None else None,
    )
    if trajectory is None:
        raise KeyError(f"Unknown trajectory {trajectory_id!r} for case {case['id']!r}")
    root = manifest_path.parent
    return {
        **case,
        "first_frame": root / case["first_frame"],
        "pose": root / trajectory["pose"],
        "trajectory_id": trajectory["id"],
    }


def _custom_example(
    input_image: Path | None,
    prompt: str | None,
    camera_path: Path | None,
    x_fov: float | None,
    xi: float,
) -> dict[str, Any] | None:
    values = {
        "input_image": input_image,
        "prompt": prompt,
        "camera_path": camera_path,
        "x_fov": x_fov,
    }
    if not any(value is not None for value in values.values()):
        return None
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(f"Custom inference requires: {', '.join(missing)}")
    return {
        "id": "custom",
        "first_frame": input_image,
        "caption": prompt,
        "pose": camera_path,
        "x_fov": x_fov,
        "xi": xi,
        "trajectory_id": camera_path.stem,
    }


@torch.inference_mode()
def generate(
    pipe: SCoPEPipeline,
    example: dict[str, Any],
    output: Path,
    config: InferenceConfig,
    negative_prompt: str,
    device: torch.device,
) -> None:
    image = Image.open(example["first_frame"]).convert("RGB")
    image = image.resize((config.width, config.height), Image.Resampling.LANCZOS)
    pose = np.load(example["pose"], allow_pickle=False).astype(np.float32)
    if (
        pose.ndim != 3
        or pose.shape[0] != config.num_frames
        or pose.shape[1:] not in ((3, 4), (4, 4))
    ):
        raise ValueError(f"Expected pose [81,3,4] or [81,4,4], got {pose.shape}")
    pose = pose[:, :3, :4]
    camera = {
        "pose": torch.from_numpy(pose)[None].to(device=device, dtype=pipe.torch_dtype),
        "x_fov": torch.tensor([example["x_fov"]], device=device, dtype=pipe.torch_dtype),
        "xi": torch.tensor([example.get("xi", 0.0)], device=device, dtype=pipe.torch_dtype),
    }
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        video = pipe(
            prompt=example["caption"],
            input_image=image,
            camera_control_panshot=camera,
            negative_prompt=negative_prompt,
            num_inference_steps=config.num_inference_steps,
            sigma_shift=config.sigma_shift,
            cfg_scale=config.cfg_scale,
            tiled=False,
            seed=config.seed,
            height=config.height,
            width=config.width,
            num_frames=config.num_frames,
            switch_DiT_boundary=config.switch_dit_boundary,
            lock_first_frame=False,
            camera_cfg_scale=1.0,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output), fps=config.fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an I2V sample with SCoPE")
    parser.add_argument("--manifest", type=Path, default=Path("examples/manifest.json"))
    parser.add_argument("--case", default=None)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--input_image", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--camera_path", type=Path, default=None)
    parser.add_argument("--x_fov", type=float, default=None)
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument(
        "--output_path",
        "--output",
        dest="output_path",
        type=Path,
        default=Path("outputs/sample.mp4"),
    )
    parser.add_argument("--model_path", "--scope-model", dest="model_path", default=SCOPE_MODEL_ID)
    parser.add_argument("--cache_dir", "--cache-dir", dest="cache_dir", type=Path, default=None)
    parser.add_argument(
        "--negative_prompt",
        "--negative-prompt",
        dest="negative_prompt",
        type=Path,
        default=Path("configs/negative_prompt.txt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vram_limit_gb", "--vram-limit-gb", dest="vram_limit_gb", type=float, default=None
    )
    args = parser.parse_args()

    config = InferenceConfig(seed=args.seed)
    example = _custom_example(
        args.input_image,
        args.prompt,
        args.camera_path,
        args.x_fov,
        args.xi,
    )
    if example is None:
        example = _select_example(args.manifest, args.case, args.trajectory)
    elif args.case is not None or args.trajectory is not None:
        parser.error("Use either a manifest case or custom inputs, not both")
    model_dir = resolve_model_dir(args.model_path, args.cache_dir)
    pipe = load_pipeline(model_dir, config)
    print("Loaded SCoPE.")
    device = _prepare_device(pipe, args.vram_limit_gb)
    negative_prompt = args.negative_prompt.read_text(encoding="utf-8").strip()
    generate(pipe, example, args.output_path, config, negative_prompt, device)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
