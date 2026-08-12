from __future__ import annotations

import json

import numpy as np
import pytest

from scope.data._pose import (
    first_camera_relative,
    load_near_depth_map,
    scale_translation,
    to_c2w_44,
)


def test_first_camera_relative_maps_first_to_identity() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    poses[0, 0, 3] = 10.0
    poses[1, 0, 3] = 12.0
    poses[2, 0, 3] = 15.0

    relative = first_camera_relative(poses)

    np.testing.assert_allclose(relative[0], np.eye(4, dtype=np.float32)[:3], atol=1e-6)
    np.testing.assert_allclose(relative[1, :, 3], [2.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(relative[2, :, 3], [5.0, 0.0, 0.0], atol=1e-6)


def test_first_camera_relative_accepts_3x4_input() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :3], 2, axis=0)
    poses[1, 1, 3] = 4.0
    relative = first_camera_relative(poses)
    assert relative.shape == (2, 3, 4)
    np.testing.assert_allclose(relative[1, :, 3], [0.0, 4.0, 0.0], atol=1e-6)


def test_scale_translation_divides_by_near_depth() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :3], 2, axis=0)
    poses[1, 0, 3] = 2.0
    scaled = scale_translation(poses, near_depth=2.0, trajectory_scale=3.0)
    np.testing.assert_allclose(scaled[1, :, 3], [3.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(scaled[1, :, :3], np.eye(3, dtype=np.float32), atol=1e-6)


def test_scale_translation_rejects_bad_near_depth() -> None:
    poses = np.eye(4, dtype=np.float32)[None, :3]
    with pytest.raises(ValueError):
        scale_translation(poses, near_depth=0.0)


def test_to_c2w_44_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        to_c2w_44(np.zeros((4, 2, 2), dtype=np.float32))


def test_load_near_depth_map_drops_invalid(tmp_path) -> None:
    payload = {
        "clip_a": {"near_depth": 2.5},
        "clip_b": {"near_depth": None},
        "clip_c": {"near_depth": -1.0},
        "clip_d": 4.0,
    }
    path = tmp_path / "near_depth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = load_near_depth_map(path)
    assert result == {"clip_a": 2.5, "clip_d": 4.0}


def test_load_near_depth_map_missing_returns_none(tmp_path) -> None:
    assert load_near_depth_map(None) is None
    assert load_near_depth_map(tmp_path / "nope.json") is None
