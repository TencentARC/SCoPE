"""Resolve manifest, hybrid, and fully custom inference inputs without loading the model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _find_case(manifest_path: Path, case_id: str | None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    case = next(
        (item for item in cases if item["id"] == case_id),
        cases[0] if case_id is None else None,
    )
    if case is None:
        raise KeyError(f"Unknown case: {case_id}")
    return case


def select_example(
    manifest_path: Path,
    case_id: str | None,
    trajectory_id: str | None,
) -> dict[str, Any]:
    """Select one trajectory registered under a manifest case."""
    case = _find_case(manifest_path, case_id)
    trajectories = case["trajectories"]
    trajectory = next(
        (item for item in trajectories if item["id"] == trajectory_id),
        trajectories[0] if trajectory_id is None else None,
    )
    if trajectory is None:
        available = ", ".join(item["id"] for item in trajectories)
        raise KeyError(
            f"Unknown trajectory {trajectory_id!r} for case {case['id']!r}. "
            f"Available: {available}"
        )
    root = manifest_path.parent
    return {
        **case,
        "first_frame": root / case["first_frame"],
        "pose": root / trajectory["pose"],
        "trajectory_id": trajectory["id"],
    }


def choose_case_trajectory(
    manifest_path: Path,
    case_id: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> str:
    """Ask a terminal user to choose one registered trajectory by number or id."""
    case = _find_case(manifest_path, case_id)
    trajectories = case["trajectories"]
    if not trajectories:
        raise ValueError(f"Case {case_id!r} has no trajectories")

    output_fn(f"Available trajectories for {case_id!r}:")
    for index, trajectory in enumerate(trajectories, start=1):
        output_fn(f"  [{index}] {trajectory['id']}  ({trajectory['pose']})")

    ids = [trajectory["id"] for trajectory in trajectories]
    while True:
        choice = input_fn(f"Select a trajectory [1-{len(ids)}] or enter its id: ").strip()
        if choice in ids:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(ids):
            return ids[int(choice) - 1]
        output_fn(f"Invalid selection {choice!r}; enter a listed number or trajectory id.")


def select_case_with_external_pose(
    manifest_path: Path,
    case_id: str,
    camera_path: Path,
    x_fov: float | None = None,
    xi: float | None = None,
) -> dict[str, Any]:
    """Use a manifest case's image/prompt/intrinsics with an external pose."""
    case = _find_case(manifest_path, case_id)
    root = manifest_path.parent
    example = {
        **case,
        "first_frame": root / case["first_frame"],
        "pose": camera_path,
        "trajectory_id": camera_path.stem,
    }
    if x_fov is not None:
        example["x_fov"] = x_fov
    if xi is not None:
        example["xi"] = xi
    return example


def custom_example(
    input_image: Path | None,
    prompt: str | None,
    camera_path: Path | None,
    x_fov: float | None,
    xi: float | None,
) -> dict[str, Any] | None:
    """Resolve a fully custom image, prompt, pose, and camera configuration."""
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
        "xi": 0.0 if xi is None else xi,
        "trajectory_id": camera_path.stem,
    }


def resolve_example_inputs(
    manifest_path: Path,
    case_id: str | None,
    trajectory_id: str | None,
    input_image: Path | None,
    prompt: str | None,
    camera_path: Path | None,
    x_fov: float | None,
    xi: float | None,
) -> dict[str, Any]:
    """Resolve manifest, case-plus-external-pose, or fully custom inputs."""
    if case_id is not None and camera_path is not None:
        if trajectory_id is not None:
            raise ValueError("Use --trajectory or --camera_path with --case, not both")
        if input_image is not None or prompt is not None:
            raise ValueError("--case supplies the input image and prompt")
        return select_case_with_external_pose(
            manifest_path, case_id, camera_path, x_fov=x_fov, xi=xi
        )

    example = custom_example(input_image, prompt, camera_path, x_fov, xi)
    if example is not None:
        if case_id is not None or trajectory_id is not None:
            raise ValueError("Use either a manifest case or custom inputs, not both")
        return example
    return select_example(manifest_path, case_id, trajectory_id)


def load_pose(path: Path, num_frames: int) -> np.ndarray:
    """Load and validate a pose before any model weights are loaded."""
    pose = np.load(path, allow_pickle=False).astype(np.float32)
    if (
        pose.ndim != 3
        or pose.shape[0] != num_frames
        or pose.shape[1:] not in ((3, 4), (4, 4))
    ):
        raise ValueError(
            f"Expected pose [{num_frames},3,4] or [{num_frames},4,4], got {pose.shape}"
        )
    if not np.isfinite(pose).all():
        raise ValueError(f"Pose contains non-finite values: {path}")
    return pose[:, :3, :4]
