from __future__ import annotations

import json
from pathlib import Path

import pytest

from scope.case_inference import _output_path, _select_case_examples


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "scene",
                        "first_frame": "frame.png",
                        "trajectories": [
                            {"id": "left", "pose": "left.npy"},
                            {"id": "orbit", "pose": "orbit.npy"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_selects_every_trajectory_for_one_case(tmp_path: Path) -> None:
    examples = _select_case_examples(_manifest(tmp_path), "scene")

    assert [example["trajectory_id"] for example in examples] == ["left", "orbit"]
    assert examples[0]["first_frame"] == tmp_path / "frame.png"
    assert examples[1]["pose"] == tmp_path / "orbit.npy"
    assert _output_path(tmp_path, examples[1]) == tmp_path / "scene__orbit.mp4"


def test_unknown_case_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="Unknown case"):
        _select_case_examples(_manifest(tmp_path), "missing")
