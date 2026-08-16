"""Command-line inference for SCoPE on Wan2.2-I2V-A14B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from diffsynth import save_video
from scope.config import SCOPE_MODEL_ID, InferenceConfig
from scope.example_selection import (
    choose_case_trajectory,
    load_pose,
    resolve_example_inputs,
)
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


@torch.inference_mode()
def generate(
    pipe: SCoPEPipeline,
    example: dict[str, Any],
    output: Path,
    config: InferenceConfig,
    negative_prompt: str,
    device: torch.device,
    pose: np.ndarray | None = None,
) -> None:
    image = Image.open(example["first_frame"]).convert("RGB")
    image = image.resize((config.width, config.height), Image.Resampling.LANCZOS)
    if pose is None:
        pose = load_pose(example["pose"], config.num_frames)
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
    parser.add_argument("--xi", type=float, default=None)
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
    interactive_case = (
        args.case is not None
        and args.trajectory is None
        and args.camera_path is None
        and args.input_image is None
        and args.prompt is None
        and args.x_fov is None
        and args.xi is None
    )
    if interactive_case:
        if not sys.stdin.isatty():
            parser.error(
                "--case without --trajectory requires an interactive terminal; "
                "pass --trajectory when running non-interactively"
            )
        try:
            args.trajectory = choose_case_trajectory(args.manifest, args.case)
        except (EOFError, KeyboardInterrupt):
            parser.error("Trajectory selection cancelled")
        except (KeyError, ValueError) as error:
            parser.error(str(error))

    try:
        example = resolve_example_inputs(
            args.manifest,
            args.case,
            args.trajectory,
            args.input_image,
            args.prompt,
            args.camera_path,
            args.x_fov,
            args.xi,
        )
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    if args.case is not None and args.camera_path is not None:
        print(f"Using case {args.case!r} with external pose {args.camera_path}")
    try:
        pose = load_pose(example["pose"], config.num_frames)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Validated pose {example['pose']} with shape {pose.shape}.")
    model_dir = resolve_model_dir(args.model_path, args.cache_dir)
    pipe = load_pipeline(model_dir, config)
    print("Loaded SCoPE.")
    device = _prepare_device(pipe, args.vram_limit_gb)
    negative_prompt = args.negative_prompt.read_text(encoding="utf-8").strip()
    generate(pipe, example, args.output_path, config, negative_prompt, device, pose=pose)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
