from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scope.data.factory import _REGISTRY, build_dataset

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_rdpo_high_only.yaml"


def test_release_training_config_covers_four_datasets() -> None:
    config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    names = [spec["name"] for spec in config["data"]["datasets"]]
    assert set(names) == {"realestate10k", "dl3dv", "panshot", "omniworld"}
    # Best-scheme convention: scale is handled by the model's scale gate and
    # near_depth is only a translation preprocessing, so the dataset-level
    # trajectory_scale stays at 1.0 for every dataset.
    assert config["data"]["trajectory_scale"] == 1.0
    for spec in config["data"]["datasets"]:
        assert "near_depth_json" in spec


def test_build_dataset_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        build_dataset({"name": "not_a_dataset", "data_root": "/tmp"}, shared={})


def test_registry_matches_documented_datasets() -> None:
    assert set(_REGISTRY) == {"realestate10k", "dl3dv", "panshot", "omniworld"}
