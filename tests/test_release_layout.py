from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_DIFFSYNTH_MODEL_SOURCES = (
    "__init__.py",
    "model_manager.py",
    "utils.py",
    "wan_video_camera_controller.py",
    "wan_video_dit.py",
    "wan_video_dit_s2v.py",
    "wan_video_image_encoder.py",
    "wan_video_motion_controller.py",
    "wan_video_text_encoder.py",
    "wan_video_vace.py",
    "wan_video_vae.py",
)


def test_vendored_diffsynth_model_sources_are_present() -> None:
    model_dir = _REPO_ROOT / "diffsynth" / "models"
    missing = [
        name for name in _REQUIRED_DIFFSYNTH_MODEL_SOURCES if not (model_dir / name).is_file()
    ]

    assert not missing, f"Missing vendored DiffSynth model sources: {missing}"
