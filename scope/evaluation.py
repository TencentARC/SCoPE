"""Camera-trajectory metrics used to evaluate SCoPE generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np


class TrajectoryMetrics(TypedDict):
    rotation_error_degrees: float
    translation_error: float
    ate: float
    cammc: float
    scale_normalized_ate: float
    scale_normalized_cammc: float
    away_response: float
    return_translation_ratio: float
    return_rotation_degrees: float
    control_success: bool


def _as_homogeneous(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected poses [T,3,4] or [T,4,4], got {poses.shape}")
    if poses.shape[1:] == (4, 4):
        return poses.copy()
    bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (len(poses), 1, 4))
    return np.concatenate((poses, bottom), axis=1)


def _relative(poses: np.ndarray) -> np.ndarray:
    poses_44 = _as_homogeneous(poses)
    return np.linalg.inv(poses_44[0]) @ poses_44


def _normalize_translation(poses: np.ndarray) -> np.ndarray:
    output = poses.copy()
    scale = np.linalg.norm(output[:, :3, 3], axis=1).max()
    if scale > 1e-8:
        output[:, :3, 3] /= scale
    return output


def _rotation_angles_degrees(rotations: np.ndarray) -> np.ndarray:
    trace = np.trace(rotations, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def evaluate_trajectory(
    target_c2w: np.ndarray,
    predicted_c2w: np.ndarray,
    *,
    away_window: slice = slice(36, 41),
    return_window: slice = slice(76, 81),
) -> TrajectoryMetrics:
    """Evaluate one predicted camera path against its conditioning trajectory.

    The inputs must use the OpenCV camera-to-world convention. Both raw and
    independently scale-normalized translation metrics are returned.
    """
    target = _relative(target_c2w)
    predicted = _relative(predicted_c2w)
    if len(target) != len(predicted):
        raise ValueError(f"Pose lengths differ: target={len(target)}, predicted={len(predicted)}")
    if len(target) < max(away_window.stop or 0, return_window.stop or 0):
        raise ValueError("The default Revisit windows require at least 81 poses")

    target_normalized = _normalize_translation(target)
    predicted_normalized = _normalize_translation(predicted)
    relative_rotation = (
        np.swapaxes(target_normalized[:, :3, :3], 1, 2) @ predicted_normalized[:, :3, :3]
    )
    rotation_error = float(_rotation_angles_degrees(relative_rotation).mean())
    raw_translation_delta = predicted[:, :3, 3] - target[:, :3, 3]
    translation_error = float(np.linalg.norm(raw_translation_delta, axis=1).mean())
    raw_ate = float(np.sqrt(np.square(raw_translation_delta).sum(axis=1).mean()))
    raw_cammc = float(np.linalg.norm(predicted[:, :3, :4] - target[:, :3, :4], axis=(1, 2)).mean())
    normalized_translation_delta = predicted_normalized[:, :3, 3] - target_normalized[:, :3, 3]
    normalized_ate = float(np.sqrt(np.square(normalized_translation_delta).sum(axis=1).mean()))
    normalized_cammc = float(
        np.linalg.norm(
            predicted_normalized[:, :3, :4] - target_normalized[:, :3, :4],
            axis=(1, 2),
        ).mean()
    )

    predicted_distance = np.linalg.norm(predicted[:, :3, 3], axis=1)
    max_distance = float(predicted_distance.max())
    denominator = max(max_distance, 1e-8)
    away_response = float(predicted_distance[away_window].mean() / denominator)
    return_ratio = float(predicted_distance[return_window].mean() / denominator)
    return_rotation = float(_rotation_angles_degrees(predicted[return_window, :3, :3]).mean())
    success = away_response >= 0.5 and return_ratio <= 0.25 and return_rotation <= 5.0

    return {
        "rotation_error_degrees": rotation_error,
        "translation_error": translation_error,
        "ate": raw_ate,
        "cammc": raw_cammc,
        "scale_normalized_ate": normalized_ate,
        "scale_normalized_cammc": normalized_cammc,
        "away_response": away_response,
        "return_translation_ratio": return_ratio,
        "return_rotation_degrees": return_rotation,
        "control_success": bool(success),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one SCoPE camera trajectory")
    parser.add_argument("--target-pose", type=Path, required=True)
    parser.add_argument("--predicted-pose", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metrics = evaluate_trajectory(
        np.load(args.target_pose, allow_pickle=False),
        np.load(args.predicted_pose, allow_pickle=False),
    )
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
