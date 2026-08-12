"""Generate every trajectory for one release example case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scope.config import SCOPE_MODEL_ID, InferenceConfig
from scope.inference import _prepare_device, generate
from scope.weights import load_pipeline, resolve_model_dir


def _select_case_examples(manifest_path: Path, case_id: str) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = next((item for item in manifest["cases"] if item["id"] == case_id), None)
    if case is None:
        raise KeyError(f"Unknown case: {case_id}")

    root = manifest_path.parent
    return [
        {
            **case,
            "first_frame": root / case["first_frame"],
            "pose": root / trajectory["pose"],
            "trajectory_id": trajectory["id"],
        }
        for trajectory in case["trajectories"]
    ]


def _output_path(output_dir: Path, example: dict[str, Any]) -> Path:
    return output_dir / f"{example['id']}__{example['trajectory_id']}.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate every trajectory for one case with a single model load"
    )
    parser.add_argument("--manifest", type=Path, default=Path("examples/manifest.json"))
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--output_dir", "--output-dir", dest="output_dir", type=Path, default=Path("outputs")
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    examples = _select_case_examples(args.manifest, args.case)
    pending = [
        example
        for example in examples
        if args.overwrite or not _output_path(args.output_dir, example).is_file()
    ]
    print(f"Selected {len(examples)} trajectories for {args.case}; {len(pending)} pending.")
    if not pending:
        return

    config = InferenceConfig(seed=args.seed)
    model_dir = resolve_model_dir(args.model_path, args.cache_dir)
    pipe = load_pipeline(model_dir, config)
    print("Loaded the complete SCoPE model once for this case.")
    device = _prepare_device(pipe, args.vram_limit_gb)
    negative_prompt = args.negative_prompt.read_text(encoding="utf-8").strip()

    for index, example in enumerate(pending, start=1):
        output = _output_path(args.output_dir, example)
        temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
        print(f"[{index}/{len(pending)}] Generating {example['trajectory_id']}")
        generate(pipe, example, temporary, config, negative_prompt, device)
        temporary.replace(output)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
