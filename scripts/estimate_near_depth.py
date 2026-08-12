#!/usr/bin/env python
"""Per-clip near-depth estimator for SCoPE training.

Every SCoPE dataset scales its camera translation by
``trajectory_scale / near_depth``. This script estimates one ``near_depth`` per
clip so the four datasets - RealEstate10K, DL3DV, PanShot, OmniWorld - share a
comparable motion magnitude. Run it once per dataset and pass the resulting
JSON to the matching ``near_depth_json`` field in the training config.

Method (per clip): sample frame pairs, run RAFT optical flow with a
forward-backward consistency check, triangulate depth from the frame-pair
relative pose, and take ``near_depth = median`` of the per-pair 25th-percentile
depths. Because only *relative* poses between frames are used, ``near_depth`` is
invariant to how the clip is globally normalized downstream.

Usage::

    python scripts/estimate_near_depth.py --dataset realestate10k \
        --data_root /path/to/RealEstate10K --split train \
        --output near_depth/realestate10k_train.json
    python scripts/estimate_near_depth.py --dataset dl3dv \
        --data_root /path/to/DL3DV --output near_depth/dl3dv.json
    python scripts/estimate_near_depth.py --dataset panshot \
        --data_root /path/to/PanShot --split train \
        --output near_depth/panshot_train.json
    python scripts/estimate_near_depth.py --dataset omniworld \
        --data_root /path/to/OmniWorld --index /path/to/OmniWorld/valid_entries.json \
        --output near_depth/omniworld.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
except ImportError as error:  # pragma: no cover
    raise ImportError("torchvision>=0.12 with the optical_flow module is required") from error

SCHEMA_VERSION = 1
FLUSH_EVERY = 50
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
_FOV_XI_RE = re.compile(r"-fov([\d.]+)-xi([\d.]+)$")


# --------------------------------------------------------------------------- #
# JSON I/O - atomic flush + resume
# --------------------------------------------------------------------------- #
def load_existing(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as error:
            print(f"[warn] failed to load {path}: {error}; starting fresh")
    return {}


def atomic_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Pose helpers (near_depth is frame-pair relative, so global convention is
# irrelevant; recentring only stabilizes numeric ranges).
# --------------------------------------------------------------------------- #
def _recenter(c2w_34: np.ndarray) -> np.ndarray:
    """Subtract the first-frame translation from an OpenCV c2w ``(T, 3, 4)``."""
    out = np.array(c2w_34, dtype=np.float32, copy=True)
    out[:, :3, 3] -= out[0:1, :3, 3]
    return out


def convention_cos(c2w_34: np.ndarray) -> float:
    """cos(mean-forward, motion); positive implies OpenCV c2w."""
    if c2w_34.shape[0] < 2:
        return 0.0
    forward = c2w_34[:, :3, 2].mean(axis=0)
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    motion = c2w_34[-1, :3, 3] - c2w_34[0, :3, 3]
    norm = np.linalg.norm(motion)
    if norm < 1e-6:
        return 0.0
    return float(np.dot(forward, motion / norm))


# --------------------------------------------------------------------------- #
# RAFT wrapper
# --------------------------------------------------------------------------- #
class RAFTRunner:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2)
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _pad8(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, height, width = x.shape
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, pad_h, pad_w

    @torch.no_grad()
    def flow(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        x1, pad_h, pad_w = self._pad8(img1)
        x2, _, _ = self._pad8(img2)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=True):
            flows = self.model(x1, x2)
        flow = flows[-1].float()
        if pad_h or pad_w:
            flow = flow[..., : img1.shape[-2], : img1.shape[-1]]
        return flow


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _build_grid(height: int, width: int, device) -> torch.Tensor:
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)


def _fb_consistency(flow12: torch.Tensor, flow21: torch.Tensor) -> torch.Tensor:
    _, height, width = flow12.shape
    grid = _build_grid(height, width, flow12.device)
    warped = grid + flow12.permute(1, 2, 0)
    norm = warped.clone()
    norm[..., 0] = 2.0 * norm[..., 0] / max(width - 1, 1) - 1.0
    norm[..., 1] = 2.0 * norm[..., 1] / max(height - 1, 1) - 1.0
    sampled = F.grid_sample(
        flow21.unsqueeze(0),
        norm.unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0]
    return torch.linalg.norm(flow12 + sampled, dim=0)


def _skew(t: torch.Tensor) -> torch.Tensor:
    x, y, z = t.unbind(-1)
    zero = torch.zeros_like(x)
    row0 = torch.stack([zero, -z, y], dim=-1)
    row1 = torch.stack([z, zero, -x], dim=-1)
    row2 = torch.stack([-y, x, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _triangulate_closed_form(p1_pix, p2_pix, K, R_12, t_12):
    height, width, _ = p1_pix.shape
    device = p1_pix.device
    ones = torch.ones(height, width, 1, device=device, dtype=p1_pix.dtype)
    p1h = torch.cat([p1_pix, ones], dim=-1)
    p2h = torch.cat([p2_pix, ones], dim=-1)
    k_inv = torch.linalg.inv(K).to(p1h.dtype)
    r1 = p1h @ k_inv.T
    r2 = p2h @ k_inv.T
    rr1 = r1 @ R_12.T
    A = torch.stack([rr1, -r2], dim=-1)
    ata = A.transpose(-1, -2) @ A
    rhs = (-t_12).expand(height, width, 3)
    atb = (A.transpose(-1, -2) @ rhs.unsqueeze(-1)).squeeze(-1)
    det = ata[..., 0, 0] * ata[..., 1, 1] - ata[..., 0, 1] * ata[..., 1, 0]
    det_safe = det.clone()
    det_safe[det_safe.abs() < 1e-12] = 1e-12
    z1 = (ata[..., 1, 1] / det_safe) * atb[..., 0] + (-ata[..., 0, 1] / det_safe) * atb[..., 1]
    z2 = (-ata[..., 1, 0] / det_safe) * atb[..., 0] + (ata[..., 0, 0] / det_safe) * atb[..., 1]
    return z1, z2, det, det.abs() > 1e-8


def _epipolar_error_px(p1_pix, p2_pix, K, R_12, t_12):
    height, width, _ = p1_pix.shape
    device = p1_pix.device
    ones = torch.ones(height, width, 1, device=device, dtype=p1_pix.dtype)
    p1h = torch.cat([p1_pix, ones], dim=-1)
    p2h = torch.cat([p2_pix, ones], dim=-1)
    k_inv = torch.linalg.inv(K)
    fmat = k_inv.T @ _skew(t_12) @ R_12 @ k_inv
    l2 = p1h @ fmat.T
    n2 = torch.sqrt(l2[..., 0] ** 2 + l2[..., 1] ** 2 + 1e-12)
    d_fwd = (p2h * l2).sum(-1).abs() / n2
    l1 = p2h @ fmat
    n1 = torch.sqrt(l1[..., 0] ** 2 + l1[..., 1] ** 2 + 1e-12)
    d_bwd = (p1h * l1).sum(-1).abs() / n1
    return 0.5 * (d_fwd + d_bwd)


def _estimate_pair_p25(img1, img2, K, R_12, t_12, raft, central_crop_ratio, min_inliers=1000):
    device = img1.device
    height, width = img1.shape[-2], img1.shape[-1]
    f12 = raft.flow(img1, img2)[0]
    f21 = raft.flow(img2, img1)[0]
    fb = _fb_consistency(f12, f21)
    grid = _build_grid(height, width, device)
    p2 = grid + f12.permute(1, 2, 0)
    epi = _epipolar_error_px(grid, p2, K, R_12, t_12)
    z1, z2, _, ok_det = _triangulate_closed_form(grid, p2, K, R_12, t_12)
    thresh = 1.5 * max(height, width) / 832.0
    mask_base = (fb < 1.5) & (epi < thresh) & (z1 > 0) & (z2 > 0) & ok_det & (z1 < 1e4) & (z2 < 1e4)
    if central_crop_ratio < 1.0:
        pad_h = int(height * (1 - central_crop_ratio) / 2)
        pad_w = int(width * (1 - central_crop_ratio) / 2)
        cc = torch.zeros_like(mask_base)
        cc[pad_h : height - pad_h, pad_w : width - pad_w] = True
        mask = mask_base & cc
    else:
        mask = mask_base
    n_valid = int(mask.sum().item())
    if n_valid < min_inliers:
        return None
    p25 = float(torch.quantile(z1[mask], 0.25).item())
    if central_crop_ratio >= 1.0:
        denom = float(mask_base.numel())
    else:
        pad_h = int(height * (1 - central_crop_ratio) / 2)
        pad_w = int(width * (1 - central_crop_ratio) / 2)
        denom = float((height - 2 * pad_h) * (width - 2 * pad_w))
    if not np.isfinite(p25) or p25 <= 0:
        return None
    return p25, n_valid / max(denom, 1.0)


def _pair_indices(n_frames: int, stride: int, max_pairs: int) -> list[tuple[int, int]]:
    pairs, i = [], 0
    while i + stride < n_frames:
        pairs.append((i, i + stride))
        i += stride
        if len(pairs) >= max_pairs:
            break
    return pairs


def _rel_pose(c2w_i: np.ndarray, c2w_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_i, r_j = c2w_i[:3, :3], c2w_j[:3, :3]
    c_i, c_j = c2w_i[:3, 3], c2w_j[:3, 3]
    return (r_j.T @ r_i).astype(np.float32), (r_j.T @ (c_i - c_j)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def _pil_to_tensor(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    return tensor.to(device, non_blocking=True)


def _decode_mp4(path: Path) -> list[np.ndarray]:
    import imageio.v3 as iio

    return list(iio.imiter(str(path)))


# --------------------------------------------------------------------------- #
# Uniform per-clip descriptor + dataset iterators
# --------------------------------------------------------------------------- #
class ClipEntry:
    def __init__(self, clip_id, n_frames, load_frame_fn, pose_fn, intrinsics_fn):
        self.clip_id = clip_id
        self.n_frames = n_frames
        self.load_frame = load_frame_fn
        self.pose = pose_fn
        self.intrinsics = intrinsics_fn


def iter_realestate10k(data_root: Path, split: str) -> list[ClipEntry]:
    process_dir = data_root / "process" / split
    index_path = process_dir / f"{split}_index.json"
    if index_path.exists():
        clip_dirs = {k: Path(v) for k, v in json.loads(index_path.read_text()).items()}
    else:
        clip_dirs = {}
        for batch in sorted(process_dir.glob("*")):
            if batch.is_dir():
                for entry in batch.iterdir():
                    if (entry / "transforms.json").exists():
                        clip_dirs[entry.name] = entry
    return [
        _make_transforms_entry(clip_id, clip_dir, clip_dir, gl_to_cv=False)
        for clip_id, clip_dir in sorted(clip_dirs.items())
        if (clip_dir / "transforms.json").exists()
    ]


def iter_dl3dv(data_root: Path) -> list[ClipEntry]:
    index_path = data_root / "valid_video_dirs.json"
    if index_path.exists():
        scenes = [Path(p) for p in json.loads(index_path.read_text())]
    else:
        scenes = [
            scene
            for res_dir in sorted(data_root.iterdir())
            if res_dir.is_dir()
            for scene in sorted(res_dir.iterdir())
            if (scene / "transforms.json").exists()
        ]
    return [
        _make_transforms_entry(s.name, s, s / "images_4", gl_to_cv=True)
        for s in scenes
        if (s / "transforms.json").exists()
    ]


def _make_transforms_entry(
    clip_id: str, scene_dir: Path, image_dir: Path, gl_to_cv: bool
) -> ClipEntry:
    meta = json.loads((scene_dir / "transforms.json").read_text())
    frames = meta["frames"]
    w_json, h_json = int(meta.get("w", 0)), int(meta.get("h", 0))
    fl_x, fl_y = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta.get("cx", w_json / 2)), float(meta.get("cy", h_json / 2))

    def _load_frame(i, device):
        return _pil_to_tensor(image_dir / Path(frames[i]["file_path"]).name, device)

    def _pose(indices):
        poses = []
        for i in indices:
            c2w = np.asarray(frames[i]["transform_matrix"], dtype=np.float32)
            if gl_to_cv:
                c2w = c2w @ _GL2CV
            poses.append(c2w[:3])
        return _recenter(np.stack(poses))

    def _intrinsics(img_hw):
        height, width = img_hw
        sx = width / max(w_json, 1)
        sy = height / max(h_json, 1)
        return np.array(
            [[fl_x * sx, 0, cx * sx], [0, fl_y * sy, cy * sy], [0, 0, 1]], dtype=np.float32
        )

    return ClipEntry(clip_id, len(frames), _load_frame, _pose, _intrinsics)


def iter_panshot(data_root: Path, split: str) -> list[ClipEntry]:
    video_dir = data_root / f"videos-{split}"
    pose_dir = data_root / f"pose-{split}"
    entries: list[ClipEntry] = []
    with (data_root / f"captioned-{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            name = json.loads(line)["video"]
            match = _FOV_XI_RE.search(name)
            if match is None or float(match.group(2)) > 0:  # pinhole only
                continue
            mp4 = video_dir / f"{name}.mp4"
            pose = pose_dir / f"{_FOV_XI_RE.sub('', name)}.npy"
            if mp4.exists() and pose.exists():
                entries.append(_make_panshot_entry(name, mp4, pose, float(match.group(1))))
    return entries


def _make_panshot_entry(name: str, mp4_path: Path, pose_path: Path, fov_deg: float) -> ClipEntry:
    poses_np = np.load(pose_path).astype(np.float32)
    n = int(poses_np.shape[0])
    cache: dict[str, Any] = {}

    def _load_frame(i, device):
        if "frames" not in cache:
            cache["frames"] = _decode_mp4(mp4_path)
        frames = cache["frames"]
        arr = np.asarray(frames[max(0, min(i, len(frames) - 1))], dtype=np.float32) / 255.0
        return (torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).to(device)

    def _pose(indices):
        return _recenter(poses_np[np.clip(indices, 0, n - 1)].copy())

    def _intrinsics(img_hw):
        height, width = img_hw
        fl = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        return np.array([[fl, 0, width / 2], [0, fl, height / 2], [0, 0, 1]], dtype=np.float32)

    return ClipEntry(name, n, _load_frame, _pose, _intrinsics)


def iter_omniworld(data_root: Path, index_path: Path) -> list[ClipEntry]:
    entries = json.loads(index_path.read_text())
    return [_make_omniworld_entry(data_root, entry) for entry in entries]


def _make_omniworld_entry(data_root: Path, entry: dict[str, Any]) -> ClipEntry:
    scene = entry["scene"]
    split_idx = int(entry["split_idx"])
    frame_start = int(entry["frame_start"])
    local_start = int(entry["split_local_start"])
    n_frames = int(entry["frame_end"]) - frame_start + 1
    clip_id = f"{scene}_split{split_idx}_{frame_start:06d}"
    color_dir = data_root / scene / "color"
    droid_path = data_root / scene / "droidclib" / f"split_{split_idx}.json"
    cache: dict[str, Any] = {}

    def _droid():
        if "droid" not in cache:
            cache["droid"] = json.loads(droid_path.read_text())
        return cache["droid"]

    def _load_frame(i, device):
        return _pil_to_tensor(color_dir / f"{frame_start + i:06d}.png", device)

    def _pose(indices):
        extrinsics = np.asarray(_droid()["extrinsics"], dtype=np.float32)
        c2w = np.linalg.inv(extrinsics)  # DROID emits camera-from-world
        poses = np.stack([c2w[local_start + i][:3] for i in indices])
        return _recenter(poses)

    def _intrinsics(img_hw):
        height, width = img_hw
        intr = _droid().get("orig_intrinsic") or _droid().get("crop_intrinsic") or {}
        fx = float(intr.get("fx", width))
        fy = float(intr.get("fy", fx))
        cx = float(intr.get("cx", width / 2))
        cy = float(intr.get("cy", height / 2))
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    return ClipEntry(clip_id, n_frames, _load_frame, _pose, _intrinsics)


# --------------------------------------------------------------------------- #
# Per-clip driver
# --------------------------------------------------------------------------- #
def process_clip(entry, raft, device, pair_stride, max_pairs, central_crop_ratio):
    empty = {
        "near_depth": None,
        "n_pairs_total": 0,
        "n_pairs_valid": 0,
        "mean_inlier_ratio": 0.0,
        "version": SCHEMA_VERSION,
    }
    if entry.n_frames < pair_stride + 1:
        return {**empty, "note": f"n_frames={entry.n_frames} < pair_stride+1"}

    pairs = _pair_indices(entry.n_frames, pair_stride, max_pairs)
    needed = sorted({i for pair in pairs for i in pair})
    tensors: dict[int, torch.Tensor] = {}
    for i in needed:
        try:
            tensors[i] = entry.load_frame(i, device)
        except Exception as error:
            return {**empty, "n_pairs_total": len(pairs), "note": f"frame-load error {i}: {error}"}

    all_indices = list(tensors.keys())
    pose_rows = entry.pose(all_indices)
    idx_to_row = {fi: k for k, fi in enumerate(all_indices)}
    first = next(iter(tensors.values()))
    K = torch.from_numpy(entry.intrinsics((int(first.shape[-2]), int(first.shape[-1])))).to(device)

    p25_list: list[float] = []
    ratio_list: list[float] = []
    for i, j in pairs:
        r_np, t_np = _rel_pose(pose_rows[idx_to_row[i]], pose_rows[idx_to_row[j]])
        if float(np.linalg.norm(t_np)) < 1e-6:
            continue
        try:
            res = _estimate_pair_p25(
                tensors[i],
                tensors[j],
                K,
                torch.from_numpy(r_np).to(device),
                torch.from_numpy(t_np).to(device),
                raft=raft,
                central_crop_ratio=central_crop_ratio,
            )
        except Exception as error:
            print(f"  [pair {i}->{j}] error: {error}")
            continue
        if res is not None:
            p25_list.append(res[0])
            ratio_list.append(res[1])

    del tensors
    if not p25_list:
        return {**empty, "n_pairs_total": len(pairs)}
    return {
        "near_depth": round(float(np.median(p25_list)), 6),
        "n_pairs_total": len(pairs),
        "n_pairs_valid": len(p25_list),
        "mean_inlier_ratio": round(float(np.mean(ratio_list)), 4),
        "version": SCHEMA_VERSION,
    }


def _convention_sanity(entries: list[ClipEntry], max_clips: int = 10) -> None:
    cosines = []
    for entry in entries[:max_clips]:
        if entry.n_frames < 8:
            continue
        idxs = np.linspace(0, entry.n_frames - 1, 8, dtype=int).tolist()
        try:
            cosines.append(convention_cos(entry.pose(idxs)))
        except Exception:
            continue
    if cosines:
        median = float(np.median(cosines))
        print(f"[sanity] median cos(forward, motion) = {median:+.3f} (expect > 0.3 for OpenCV)")


_ITERATORS: dict[str, Callable[..., list[ClipEntry]]] = {
    "realestate10k": lambda root, split, index: iter_realestate10k(root, split),
    "dl3dv": lambda root, split, index: iter_dl3dv(root),
    "panshot": lambda root, split, index: iter_panshot(root, split),
    "omniworld": lambda root, split, index: iter_omniworld(root, index),
}
# DL3DV frames carry mild edge distortion, so restrict triangulation to the
# central crop; the other datasets use the full frame.
_CENTRAL_CROP = {"dl3dv": 0.8}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(_ITERATORS), required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--index", type=Path, default=None, help="OmniWorld valid_entries.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pair_stride", type=int, default=5)
    parser.add_argument("--max_pairs", type=int, default=20)
    parser.add_argument("--skip_sanity", action="store_true")
    args = parser.parse_args()

    if args.dataset == "omniworld" and args.index is None:
        parser.error("--index is required for the omniworld dataset")

    print(f"[estimator] dataset={args.dataset} root={args.data_root} output={args.output}")
    entries = _ITERATORS[args.dataset](args.data_root, args.split, args.index)
    central_crop_ratio = _CENTRAL_CROP.get(args.dataset, 1.0)
    print(f"[estimator] {len(entries)} clips, central_crop_ratio={central_crop_ratio}")
    if args.limit:
        entries = entries[: args.limit]
    if not args.skip_sanity:
        _convention_sanity(entries)

    existing = load_existing(args.output)
    done = set(existing)
    print(f"[estimator] {len(done)} clips already done in {args.output}")

    torch.backends.cudnn.benchmark = True
    raft = RAFTRunner(args.device)
    device = torch.device(args.device)

    processed, errors, null_count = 0, 0, 0
    start = time.time()
    for k, entry in enumerate(tqdm(entries)):
        if entry.clip_id in done:
            continue
        try:
            record = process_clip(
                entry,
                raft,
                device,
                pair_stride=args.pair_stride,
                max_pairs=args.max_pairs,
                central_crop_ratio=central_crop_ratio,
            )
        except Exception as error:
            errors += 1
            print(f"[{k + 1}/{len(entries)}] {entry.clip_id} ERROR: {error}")
            traceback.print_exc()
            continue
        existing[entry.clip_id] = record
        processed += 1
        null_count += int(record["near_depth"] is None)
        if processed % FLUSH_EVERY == 0:
            atomic_dump(args.output, existing)

    atomic_dump(args.output, existing)
    print(
        f"[done] wrote {len(existing)} clips to {args.output} "
        f"({time.time() - start:.1f}s, null={null_count}, err={errors})"
    )


if __name__ == "__main__":
    main()
