"""Assemble the SCoPE training mixture from a config.

The released RDPO high-only recipe concatenates four datasets - RealEstate10K,
DL3DV, PanShot, and OmniWorld - each read by its own native loader but sharing
one camera convention (first-camera-relative poses, per-clip near-depth
translation preprocessing; scale is handled by the model's scale gate).
The unweighted mixture is a ``ConcatDataset``; each dataset's length therefore
determines its sampling proportion, matching the training setup.

Config schema (see configs/train_rdpo_high_only.yaml)::

    num_frames: 81
    height: 480
    width: 832
    trajectory_scale: 1.0
    datasets:
      - name: realestate10k
        data_root: /data/RealEstate10K
        split: train
        sample_stride: 4
        near_depth_json: /data/RealEstate10K/near_depth_train.json
      - name: dl3dv
        data_root: /data/DL3DV
        sample_stride: 1
        near_depth_json: /data/DL3DV/near_depth.json
      - name: panshot
        data_root: /data/PanShot
        split: train
        near_depth_json: /data/PanShot/near_depth_train.json
      - name: omniworld
        data_root: /data/OmniWorld
        index_path: /data/OmniWorld/valid_entries.json
        near_depth_json: /data/OmniWorld/near_depth.json
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import ConcatDataset

from scope.data._pose import load_near_depth_map
from scope.data.dl3dv import DL3DVDataset
from scope.data.omniworld import OmniWorldDataset
from scope.data.panshot import PanShotDataset
from scope.data.realestate10k import RealEstate10KDataset

_REGISTRY = {
    "realestate10k": RealEstate10KDataset,
    "dl3dv": DL3DVDataset,
    "panshot": PanShotDataset,
    "omniworld": OmniWorldDataset,
}

_SHARED_KEYS = ("num_frames", "height", "width", "trajectory_scale", "return_first_frame")


def build_dataset(spec: dict[str, Any], shared: dict[str, Any]):
    """Instantiate a single dataset from its spec plus shared defaults."""
    spec = dict(spec)
    name = spec.pop("name")
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {sorted(_REGISTRY)}.")
    params = {key: shared[key] for key in _SHARED_KEYS if key in shared}
    params.update(spec)
    params["near_depth_map"] = load_near_depth_map(params.pop("near_depth_json", None))
    return _REGISTRY[name](**params)


def build_training_dataset(config: dict[str, Any]) -> ConcatDataset:
    """Build the concatenated training mixture described by ``config``."""
    datasets = config.get("datasets")
    if not datasets:
        raise ValueError("config['datasets'] must list at least one dataset")
    shared = {key: config[key] for key in _SHARED_KEYS if key in config}
    shared.setdefault("return_first_frame", True)
    built = [build_dataset(spec, shared) for spec in datasets]
    total = sum(len(d) for d in built)
    print(f"[SCoPE mixture] {len(built)} datasets, {total} clips total")
    return ConcatDataset(built)
