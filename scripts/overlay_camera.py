#!/usr/bin/env python3
"""Overlay a camera-frustum HUD (and optional WASD keys) on a generated video.

This is the visualization tool used for the SCoPE demos: it composites a
generated video with a translucent camera-trajectory HUD in the bottom-right
corner and, optionally, a WASD key indicator in the bottom-left corner. Both are
derived from the driving camera pose, so the overlay always matches the motion
the model was asked to follow.

The WASD keys are inferred per frame from the pose (``--pose``): inter-frame
translation is projected into the first-camera frame, forward/back maps to W/S
and right/left maps to A/D. This keeps extra yaw/turn from flipping an intended
lateral move into the opposite strafe. Without a pose it falls back to a fixed
key set inferred from the motion name (``MOTION_KEYS``).

Requires ffmpeg/ffprobe on PATH and the ``viz`` extra (matplotlib). Install with
``pip install -e .[viz]``.

Examples
--------
Frustum HUD only (recommended for scenic / dolly / orbit trajectories)::

    python scripts/overlay_camera.py outputs/omni-misty-forest__truck_right.mp4 \\
        --pose examples/poses/omni-misty-forest/truck_right.npy \\
        --out_dir outputs/overlay --hide_wasd

Frustum HUD plus WASD keys (for keyboard-style / drone trajectories)::

    python scripts/overlay_camera.py outputs/clip.mp4 \\
        --pose path/to/pose.npy --out_dir outputs/overlay
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

# Fixed key fallback when no pose is provided.
MOTION_KEYS = {
    "truck_right": {"D"},
    "truck_left": {"A"},
    "dolly_in": {"W"},
    "dolly_out": {"S"},
    "pan_right": {"D"},
    "pan_left": {"A"},
}

ACCENT = (78, 205, 255)  # highlight cyan


# --------------------------------- video io ---------------------------------
def ffprobe_video(path: Path) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,r_frame_rate,avg_frame_rate",
        "-of", "json", str(path),
    ])
    s = json.loads(out)["streams"][0]
    rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or "16/1"
    num, den = (rate.split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else 16.0
    try:
        nb = int(s.get("nb_frames"))
    except (TypeError, ValueError):
        nb = _count_frames(path)
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "nb_frames": nb}


def _count_frames(path: Path) -> int:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nk=1:nw=1", str(path),
    ])
    return int(out.strip())


def _run_ffmpeg_to(out_mp4: Path, cmd_to_stage) -> None:
    fd, p = tempfile.mkstemp(suffix=".mp4", prefix="camera_overlay_")
    os.close(fd)
    stage = Path(p)
    try:
        subprocess.check_call(cmd_to_stage(stage))
        shutil.copyfile(str(stage), str(out_mp4))
    finally:
        if stage.exists():
            stage.unlink()


# ------------------------------- pose -> keys -------------------------------
def _load_poses(path: Path) -> np.ndarray:
    p = Path(path)
    arr = np.load(p) if p.suffix == ".npy" else np.asarray(json.loads(p.read_text()))
    arr = np.asarray(arr, dtype=np.float64)
    return arr[:, :3, :4].copy()


def keys_per_frame_from_pose(poses: np.ndarray) -> list[set[str]]:
    """Per-frame active key set from c2w (OpenCV) poses, decided by the
    inter-frame translation expressed in the first-camera frame.

    Uses a forward difference: frame ``i`` shows the motion that carries the
    camera to the next frame (key -> next frame, the world-model overlay
    convention); the last frame holds the previous one. This way idle frames at
    the start/end (camera not yet moving / already stopped) read as released
    keys instead of holding the main key all the way through.
    """
    R = poses[:, :3, :3]
    t = poses[:, :, 3]
    T = len(poses)
    d = np.zeros_like(t)
    if T > 1:
        d[:-1] = t[1:] - t[:-1]
        d[-1] = d[-2]
    # Synthetic controls are authored in the anchor frame; using R[i] flips
    # car-like arcs whose heading turns along the path tangent.
    local = (R[0].T @ d.T).T
    # Light smoothing to suppress jitter near the threshold; edge padding avoids
    # decaying/trailing the first and last velocities with zeros.
    if T >= 5:
        k = np.ones(5) / 5.0
        local = np.stack([
            np.convolve(np.pad(local[:, c], 2, mode="edge"), k, mode="valid")
            for c in range(3)
        ], axis=1)
    lx, lz = local[:, 0], local[:, 2]
    thr_x = max(0.25 * np.max(np.abs(lx)), 1e-6)
    thr_z = max(0.25 * np.max(np.abs(lz)), 1e-6)
    out = []
    for i in range(T):
        ks: set[str] = set()
        if lz[i] > thr_z:
            ks.add("W")
        elif lz[i] < -thr_z:
            ks.add("S")
        if lx[i] > thr_x:
            ks.add("D")
        elif lx[i] < -thr_x:
            ks.add("A")
        out.append(ks)
    return _despike_keys(out)


def _despike_keys(keys: list[set[str]]) -> list[set[str]]:
    """Only erase single-frame jitter bracketed by identical neighbors; leave
    the genuine idle segments at the start/end untouched."""
    keys = [set(k) for k in keys]
    for i in range(1, len(keys) - 1):
        if keys[i - 1] == keys[i + 1] and keys[i] != keys[i - 1]:
            keys[i] = set(keys[i - 1])
    return keys


def _resample(keys: list[set[str]], n: int) -> list[set[str]]:
    T = len(keys)
    if T == n:
        return keys
    return [keys[min(T - 1, round(i * (T - 1) / max(n - 1, 1)))] for i in range(n)]


# ------------------------------ HUD rendering ------------------------------
def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def trajectory_points_from_pose(poses: np.ndarray) -> np.ndarray:
    """Commanded c2w pose -> normalized 2D trajectory in frame-0 camera coordinates."""
    R0 = poses[0, :3, :3]
    t0 = poses[0, :, 3]
    rel = (R0.T @ (poses[:, :, 3] - t0).T).T
    pts = rel[:, [0, 2]].astype(np.float64)  # x=right, z=forward
    center = (pts.min(0) + pts.max(0)) / 2
    pts = pts - center
    radius = float(np.max(np.linalg.norm(pts, axis=1)))
    if radius < 1e-8:
        return np.zeros_like(pts)
    return pts / radius


def to_plot_space(p: np.ndarray) -> np.ndarray:
    """OpenCV (x-right, y-down, z-forward) -> plot space (x-right, y-depth, z-up)."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([x, z, -y], axis=-1)


def frustum_corners(center, right, up, forward, depth, half_w, half_h):
    """Return (apex, [4 far-plane corners]) for one camera in plot space."""
    fc = center + forward * depth
    corners = [
        fc + right * half_w + up * half_h,
        fc + right * half_w - up * half_h,
        fc - right * half_w - up * half_h,
        fc - right * half_w + up * half_h,
    ]
    return center, corners


def frustum_faces(apex, corners):
    """5 filled faces: 4 triangular sides + 1 far quad. Solid faces give an
    unambiguous occlusion cue (vs. a wireframe that flips like a Necker cube)."""
    faces = [[apex, corners[i], corners[(i + 1) % 4]] for i in range(4)]
    faces.append([corners[0], corners[1], corners[2], corners[3]])
    return faces


def _build_frustums(poses: np.ndarray, depth: float):
    """Per-frame (apex, far-corners) in plot space plus all points for axes limits."""
    centers_p = to_plot_space(poses[:, :, 3])
    half_w = 0.66 * depth
    half_h = 0.50 * depth

    def _dir(v):
        d = to_plot_space(v)
        return d / (np.linalg.norm(d) + 1e-8)

    frus = []
    for j in range(len(poses)):
        frus.append(frustum_corners(
            centers_p[j],
            _dir(poses[j, :, 0]),
            _dir(-poses[j, :, 1]),
            _dir(poses[j, :, 2]),
            depth,
            half_w,
            half_h,
        ))
    return centers_p, frus


def _set_axes_equal(ax, pts: np.ndarray, zoom: float = 1.05) -> None:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    radius = max(float((hi - lo).max()) * 0.5 * 1.06, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=zoom)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))


def _alpha_bbox(im: Image.Image):
    """Return (cx, cy, half) of the square that tightly bounds the non-empty alpha content."""
    alpha = np.asarray(im)[..., 3]
    ys, xs = np.where(alpha > 8)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0 + 1, y1 - y0 + 1) * 0.5
    return cx, cy, half


def _crop_square(im: Image.Image, box) -> Image.Image:
    """Crop im to a fixed square (cx, cy, half) onto a transparent canvas."""
    cx, cy, half = box
    side = int(round(half * 2))
    out = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    crop = im.crop((
        int(round(cx - half)),
        int(round(cy - half)),
        int(round(cx + half)),
        int(round(cy + half)),
    ))
    out.alpha_composite(crop, (0, 0))
    return out


# HUD elevation (degrees). Overridden by main() via --elev; the default keeps a
# half-top-down view that preserves a 3D sense of volume.
VIEW_ELEV = 35.0


def render_camera_frustum_rgba(poses: np.ndarray, frame_idx: int, px: int,
                               crop_box=None) -> Image.Image:
    """Render the commanded camera trajectory as a transparent rainbow frustum HUD.

    The matplotlib axes limits are fixed from the full trajectory, so geometry sits at a
    stable position/scale in the figure across frames. `crop_box` (cx, cy, half) must also be
    shared across frames so the post-render crop+resize does not make the frustum pulse.
    """
    T = len(poses)
    t = min(frame_idx, T - 1)
    centers_p0 = to_plot_space(poses[:, :, 3])
    extent = float(np.linalg.norm(centers_p0.max(0) - centers_p0.min(0)))
    ratio = np.clip(extent / 0.5, 0.2, 8.0)
    depth = 0.15 * ratio ** 0.5
    centers_p, frus = _build_frustums(poses, depth)
    all_pts = [centers_p] + [np.asarray([apex, *corners]) for apex, corners in frus]
    all_pts = np.concatenate(all_pts, axis=0)

    cmap = matplotlib.colormaps["turbo"]

    def col(j: int):
        return cmap(j / t) if t > 0 else cmap(1.0)

    fig = plt.figure(figsize=(px / 120, px / 120), dpi=120)
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax = fig.add_subplot(111, projection="3d")
    # Short focal length = strong perspective (near big / far small) so dolly-in
    # reads as moving away, not back. Matches the paper teaser renderer.
    ax.set_proj_type("persp", focal_length=0.2)
    ax.set_facecolor((0, 0, 0, 0))
    ax.patch.set_alpha(0.0)

    # Full path as a subtle guide.
    ax.plot(centers_p[:, 0], centers_p[:, 1], centers_p[:, 2],
            color=(1, 1, 1, 0.22), lw=0.9, zorder=1, solid_capstyle="round")
    if t >= 1:
        seg_line = [[centers_p[k], centers_p[k + 1]] for k in range(t)]
        ax.add_collection3d(Line3DCollection(seg_line, colors=[col(k) for k in range(t)],
                                             linewidths=1.3, zorder=2, capstyle="round"))

    # All elapsed frustums as translucent filled faces in ONE collection so
    # matplotlib depth-sorts them (correct occlusion); the current frame is red.
    faces, facecolors, edgecolors = [], [], []
    for j in range(t + 1):
        apex, corners = frus[j]
        fcs = frustum_faces(apex, corners)
        if j == t:
            fcol = mcolors.to_rgba("#ff4d4d", 0.50)
            ecol = mcolors.to_rgba("#ff2a2a", 0.95)
        else:
            base = col(j)
            fcol = mcolors.to_rgba(base, 0.16)
            ecol = mcolors.to_rgba(base, 0.60)
        faces.extend(fcs)
        facecolors.extend([fcol] * len(fcs))
        edgecolors.extend([ecol] * len(fcs))
    poly = Poly3DCollection(faces, facecolors=facecolors, edgecolors=edgecolors,
                            linewidths=0.5)
    poly.set_zsort("average")
    ax.add_collection3d(poly)

    _set_axes_equal(ax, all_pts, zoom=1.35)
    # Fixed view for EVERY video: azim=-90 -> forward = straight up, right =
    # straight right (no skew). elev keeps a half-top-down 3D body. When the
    # elevation is too high, forward (depth) motion projects as "up" and gets
    # confused with crane_up; individual cases can flatten the view via --elev.
    ax.view_init(elev=VIEW_ELEV, azim=-90.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, facecolor=(0, 0, 0, 0))
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    if crop_box is not None:
        return _crop_square(im, crop_box)
    return im


def compute_frustum_crop_box(poses: np.ndarray, px: int, margin: float = 0.16):
    """One-shot crop box from the full-trajectory render; shared by every frame for a stable size."""
    full = render_camera_frustum_rgba(poses, len(poses) - 1, px, crop_box=None)
    bb = _alpha_bbox(full)
    if bb is None:
        return None
    cx, cy, half = bb
    return cx, cy, half * (1.0 + margin)


def _draw_text_center(draw: ImageDraw.ImageDraw, box, text: str, font, fill) -> None:
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - tw) / 2 - bb[0], y0 + (y1 - y0 - th) / 2 - bb[1]),
              text, font=font, fill=fill)


def _line(draw: ImageDraw.ImageDraw, pts, fill, width: int) -> None:
    if len(pts) >= 2:
        draw.line([tuple(p) for p in pts], fill=fill, width=width, joint="curve")


def render_overlay_frame(width: int, height: int, active: set[str], camera_img=None,
                         frame_idx: int = 0, show_wasd: bool = True) -> Image.Image:
    """WASD keys on the left, glass camera-frustum HUD on the right."""
    scale = height / 480.0
    key = int(round(31 * scale))
    gap = int(round(7 * scale))
    margin = int(round(17 * scale))
    radius = int(round(6 * scale))
    font = _font(max(13, int(round(17 * scale))))

    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    if show_wasd:
        wasd_h = key * 2 + gap
        x0 = margin
        y0 = height - margin - wasd_h
        boxes = {
            "W": (x0 + key + gap, y0, x0 + key * 2 + gap, y0 + key),
            "A": (x0, y0 + key + gap, x0 + key, y0 + key * 2 + gap),
            "S": (x0 + key + gap, y0 + key + gap, x0 + key * 2 + gap, y0 + key * 2 + gap),
            "D": (x0 + key * 2 + gap * 2, y0 + key + gap, x0 + key * 3 + gap * 2,
                  y0 + key * 2 + gap),
        }
        for k in ("W", "A", "S", "D"):
            on = k in active
            box = boxes[k]
            if on:
                fill = (240, 240, 235, 235)
                outline = (255, 255, 255, 245)
                text = (20, 20, 18, 245)
            else:
                fill = (0, 0, 0, 92)
                outline = (255, 255, 255, 62)
                text = (244, 244, 240, 205)
            d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline,
                                width=max(1, int(round(1.1 * scale))))
            _draw_text_center(d, box, k, font, text)

    if camera_img is not None:
        panel = int(round(height * 0.25))
        pad = int(round(5 * scale))
        px = width - margin - panel
        py = height - margin - panel
        rr = int(round(13 * scale))

        # Same visual language as the WASD keys: dark translucent glass, soft blurred edge.
        glow = Image.new("RGBA", (panel + 24, panel + 24), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.rounded_rectangle((12, 12, panel + 12, panel + 12), radius=rr + 3,
                             fill=(0, 0, 0, 68))
        glow = glow.filter(ImageFilter.GaussianBlur(int(round(7 * scale))))
        im.alpha_composite(glow, (px - 12, py - 12))
        d.rounded_rectangle((px, py, px + panel, py + panel), radius=rr,
                            fill=(0, 0, 0, 36), outline=(255, 255, 255, 72),
                            width=max(1, int(round(scale))))

        cam = camera_img.resize((panel - pad * 2, panel - pad * 2), Image.Resampling.LANCZOS)
        im.alpha_composite(cam, (px + pad, py + pad))
    return im


def overlay_video(video: Path, out_mp4: Path, keys: list[set[str]], camera_poses=None,
                  show_wasd: bool = True) -> None:
    info = ffprobe_video(video)
    n = info["nb_frames"]
    keys = _resample(keys, n)
    if camera_poses is not None:
        cam_idx = [
            min(len(camera_poses) - 1, round(i * (len(camera_poses) - 1) / max(n - 1, 1)))
            for i in range(n)
        ]
        cam_px = int(round(info["height"] * 0.25))
        crop_box = compute_frustum_crop_box(camera_poses, cam_px)
    else:
        cam_idx = [0] * n
        cam_px = 0
        crop_box = None
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(tempfile.mkdtemp(prefix="camera_frames_"))
    try:
        for i in range(n):
            cam = None
            if camera_poses is not None:
                cam = render_camera_frustum_rgba(camera_poses, cam_idx[i], cam_px,
                                                 crop_box=crop_box)
            render_overlay_frame(info["width"], info["height"], keys[i], cam, i,
                                 show_wasd=show_wasd).save(frames_dir / f"hud_{i:05d}.png")
        _run_ffmpeg_to(out_mp4, lambda stage: [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video),
            "-framerate", f"{info['fps']}", "-start_number", "0",
            "-i", str(frames_dir / "hud_%05d.png"),
            "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto[v]",
            "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "16", "-movflags", "+faststart", "-c:a", "copy", str(stage),
        ])
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


def _infer_motion(video: Path) -> str | None:
    stem = video.stem
    for m in MOTION_KEYS:
        if f"__{m}" in stem or stem.endswith(m):
            return m
    return None


def main() -> None:
    global VIEW_ELEV
    ap = argparse.ArgumentParser(
        description="Overlay a camera-frustum HUD (and optional WASD keys) on a video."
    )
    ap.add_argument("videos", nargs="+", help="Input mp4 file(s)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pose", default=None,
                    help="Single video: derive per-frame keys from this c2w pose "
                         "(.npy/.json). Recommended; even snake trajectories line up.")
    ap.add_argument("--motion", default=None, choices=sorted(MOTION_KEYS),
                    help="Fixed keys when no pose is given; omit to infer from the filename.")
    ap.add_argument("--suffix", default="_hud")
    ap.add_argument("--hide_wasd", action="store_true",
                    help="Only show the bottom-right camera-trajectory HUD, no WASD keys.")
    ap.add_argument("--elev", type=float, default=VIEW_ELEV,
                    help="HUD elevation in degrees (default 35); lower = flatter view, "
                         "which mitigates forward motion reading as 'up'.")
    args = ap.parse_args()

    VIEW_ELEV = args.elev
    out_dir = Path(args.out_dir)
    for vp in args.videos:
        video = Path(vp)
        info = ffprobe_video(video)
        camera_poses = None
        if args.pose:
            poses = _load_poses(Path(args.pose))
            keys = keys_per_frame_from_pose(poses)
            camera_poses = poses
            src = f"pose={Path(args.pose).name}"
        else:
            motion = args.motion or _infer_motion(video)
            ks = MOTION_KEYS.get(motion, set())
            keys = [set(ks)] * info["nb_frames"]
            src = f"motion={motion}"
        stem = video.stem
        if stem.endswith("_pred"):
            stem = stem[: -len("_pred")]
        out = out_dir / f"{stem}{args.suffix}.mp4"
        print(f"[overlay] {video.name}  {src} -> {out.name}")
        overlay_video(video, out, keys, camera_poses=camera_poses, show_wasd=not args.hide_wasd)


if __name__ == "__main__":
    main()
