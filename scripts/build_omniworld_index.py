#!/usr/bin/env python3
"""Build the OmniWorld training index consumed by scope.data.OmniWorldDataset.

Each caption file (``text/<start>_<end>.json``) corresponds to one 81-frame
training window whose frame range is already aligned to a split. This script
enumerates every caption file and keeps a window when:

* the window lies entirely inside one split (contiguous frames),
* that split's ``reproj_error_after_refine`` is below ``--reproj-threshold``,
* the window does not intersect the split's ``invalid_frame`` set,
* the sampled PNG frames exist on disk.

Output entry schema (list of dict)::

    {"scene", "split_idx", "frame_start", "frame_end",
     "caption_file", "reproj_error", "split_local_start"}

Usage::

    python scripts/build_omniworld_index.py \
        --root /path/to/OmniWorld \
        --output /path/to/OmniWorld/valid_entries.json \
        --reproj-threshold 50.0
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

NUM_FRAMES = 81  # caption window size == training clip length


def parse_text_filename(name: str) -> tuple[int, int]:
    """``000196_000276.json`` -> (196, 276)."""
    start, end = name.replace(".json", "").split("_")
    return int(start), int(end)


def find_split_for_window(splits: list[list[int]], fs: int, fe: int):
    """Return (split_idx, local_start) for the split fully containing [fs, fe]."""
    for i, sp in enumerate(splits):
        if not sp:
            continue
        sp_set = set(sp)
        if fs in sp_set and fe in sp_set:
            try:
                local_start = sp.index(fs)
            except ValueError:
                continue
            if sp[local_start : local_start + (fe - fs + 1)] != list(range(fs, fe + 1)):
                continue
            return i, local_start
    return None


def scan_one_scene(args):
    scene_dir, reproj_threshold, verify_png_samples = args
    scene_dir = Path(scene_dir)
    scene_name = scene_dir.name
    out_entries: list[dict] = []
    stats = {
        "n_text": 0,
        "kept": 0,
        "drop_no_split": 0,
        "drop_reproj": 0,
        "drop_invalid_frame": 0,
        "drop_camera_missing": 0,
        "drop_droidclib_missing": 0,
        "drop_png_missing": 0,
    }

    try:
        info = json.load(open(scene_dir / "split_info.json"))
    except Exception as error:
        return scene_name, [], stats, f"split_info read fail: {error}"
    splits = info["split"]

    text_dir = scene_dir / "text"
    if not text_dir.is_dir():
        return scene_name, [], stats, "no text dir"

    camera_data: dict[int, dict] = {}
    cam_dir = scene_dir / "camera"
    droid_dir = scene_dir / "droidclib"
    color_dir = scene_dir / "color"

    for tname in sorted(text_dir.iterdir()):
        if tname.suffix != ".json":
            continue
        stats["n_text"] += 1
        try:
            fs, fe = parse_text_filename(tname.name)
        except Exception:
            continue
        if fe - fs + 1 != NUM_FRAMES:
            continue
        match = find_split_for_window(splits, fs, fe)
        if match is None:
            stats["drop_no_split"] += 1
            continue
        split_idx, local_start = match

        cam_path = cam_dir / f"split_{split_idx}.json"
        droid_path = droid_dir / f"split_{split_idx}.json"
        if not cam_path.is_file():
            stats["drop_camera_missing"] += 1
            continue
        if not droid_path.is_file():
            stats["drop_droidclib_missing"] += 1
            continue

        if split_idx not in camera_data:
            try:
                camera_data[split_idx] = json.load(open(cam_path))
            except Exception:
                stats["drop_camera_missing"] += 1
                continue
        cam = camera_data[split_idx]

        reproj = cam.get("reproj_error_after_refine", float("inf"))
        if not (reproj < reproj_threshold):
            stats["drop_reproj"] += 1
            continue

        invalid = set(cam.get("invalid_frame", []) or [])
        if invalid and (invalid & set(range(fs, fe + 1))):
            stats["drop_invalid_frame"] += 1
            continue

        if verify_png_samples > 0:
            check_idxs = (
                [fs] if verify_png_samples == 1 else [fs, (fs + fe) // 2, fe][:verify_png_samples]
            )
            if not all((color_dir / f"{i:06d}.png").is_file() for i in check_idxs):
                stats["drop_png_missing"] += 1
                continue

        out_entries.append(
            {
                "scene": scene_name,
                "split_idx": split_idx,
                "frame_start": fs,
                "frame_end": fe,
                "caption_file": f"text/{tname.name}",
                "reproj_error": float(reproj),
                "split_local_start": local_start,
            }
        )
        stats["kept"] += 1

    return scene_name, out_entries, stats, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="OmniWorld data root")
    parser.add_argument("--output", required=True, help="output valid_entries.json path")
    parser.add_argument("--reproj-threshold", type=float, default=50.0)
    parser.add_argument("--verify-png-samples", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"[ERROR] root not found: {root}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    scenes = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    print(f"[Index] scanning {len(scenes)} scenes from {root}")

    all_entries: list[dict] = []
    agg: dict[str, int] = {}
    tasks = [(str(s), args.reproj_threshold, args.verify_png_samples) for s in scenes]
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(scan_one_scene, t): t[0] for t in tasks}
        done = 0
        for future in as_completed(futures):
            scene_name, entries, stats, err = future.result()
            done += 1
            if err:
                print(f"[skip] {scene_name}: {err}")
            all_entries.extend(entries)
            for key, value in stats.items():
                agg[key] = agg.get(key, 0) + value
            if done % 50 == 0:
                print(f"  [{done}/{len(tasks)}] cumulative kept={agg.get('kept', 0)}")

    print("\n=== summary ===")
    for key, value in agg.items():
        print(f"  {key:24s}: {value}")
    print(f"  total kept entries     : {len(all_entries)}")
    output.write_text(json.dumps(all_entries), encoding="utf-8")
    print(f"[Index] wrote {len(all_entries)} entries -> {output}")


if __name__ == "__main__":
    main()
