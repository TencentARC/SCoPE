from __future__ import annotations

import numpy as np

from scope.evaluation import evaluate_trajectory


def _closed_loop() -> np.ndarray:
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (81, 4, 4)).copy()
    distance = np.concatenate(
        (
            np.linspace(0.0, 1.0, 41),
            np.linspace(0.975, 0.0, 40),
        )
    )
    poses[:, 0, 3] = distance
    return poses


def test_identical_closed_loop_is_successful() -> None:
    poses = _closed_loop()
    metrics = evaluate_trajectory(poses, poses)

    assert metrics["rotation_error_degrees"] == 0.0
    assert metrics["translation_error"] == 0.0
    assert metrics["ate"] == 0.0
    assert metrics["cammc"] == 0.0
    assert metrics["scale_normalized_ate"] == 0.0
    assert metrics["scale_normalized_cammc"] == 0.0
    assert metrics["away_response"] >= 0.5
    assert metrics["return_translation_ratio"] <= 0.25
    assert metrics["control_success"] is True


def test_pose_length_mismatch_is_rejected() -> None:
    poses = _closed_loop()
    try:
        evaluate_trajectory(poses, poses[:-1])
    except ValueError as error:
        assert "Pose lengths differ" in str(error)
    else:
        raise AssertionError("Expected a pose-length error")
