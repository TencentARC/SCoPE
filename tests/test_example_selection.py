from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scope.example_selection import choose_case_trajectory, load_pose, resolve_example_inputs


class ExampleSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "id": "scene",
                            "first_frame": "frame.png",
                            "caption": "A test scene.",
                            "x_fov": 1.25,
                            "xi": 0.2,
                            "trajectories": [{"id": "left", "pose": "poses/left.npy"}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _resolve(self, **overrides: object) -> dict[str, object]:
        arguments = {
            "manifest_path": self.manifest,
            "case_id": None,
            "trajectory_id": None,
            "input_image": None,
            "prompt": None,
            "camera_path": None,
            "x_fov": None,
            "xi": None,
        }
        arguments.update(overrides)
        return resolve_example_inputs(**arguments)  # type: ignore[arg-type]

    def test_case_with_external_pose_inherits_case_inputs(self) -> None:
        external_pose = self.root / "external.npy"
        example = self._resolve(case_id="scene", camera_path=external_pose)

        self.assertEqual(example["first_frame"], self.root / "frame.png")
        self.assertEqual(example["caption"], "A test scene.")
        self.assertEqual(example["pose"], external_pose)
        self.assertEqual(example["trajectory_id"], "external")
        self.assertEqual(example["x_fov"], 1.25)
        self.assertEqual(example["xi"], 0.2)

    def test_interactive_selection_accepts_number(self) -> None:
        messages: list[str] = []

        selected = choose_case_trajectory(
            self.manifest,
            "scene",
            input_fn=lambda _: "1",
            output_fn=messages.append,
        )

        self.assertEqual(selected, "left")
        self.assertTrue(any("poses/left.npy" in message for message in messages))

    def test_interactive_selection_accepts_id_after_invalid_input(self) -> None:
        answers = iter(("99", "left"))
        messages: list[str] = []

        selected = choose_case_trajectory(
            self.manifest,
            "scene",
            input_fn=lambda _: next(answers),
            output_fn=messages.append,
        )

        self.assertEqual(selected, "left")
        self.assertTrue(any("Invalid selection" in message for message in messages))

    def test_registered_case_and_trajectory_still_resolve(self) -> None:
        example = self._resolve(case_id="scene", trajectory_id="left")

        self.assertEqual(example["first_frame"], self.root / "frame.png")
        self.assertEqual(example["pose"], self.root / "poses/left.npy")
        self.assertEqual(example["trajectory_id"], "left")

    def test_case_with_external_pose_allows_camera_overrides(self) -> None:
        example = self._resolve(
            case_id="scene",
            camera_path=self.root / "external.npy",
            x_fov=0.9,
            xi=0.0,
        )

        self.assertEqual(example["x_fov"], 0.9)
        self.assertEqual(example["xi"], 0.0)

    def test_case_rejects_two_pose_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "--trajectory or --camera_path"):
            self._resolve(
                case_id="scene",
                trajectory_id="left",
                camera_path=self.root / "external.npy",
            )

    def test_case_rejects_custom_image_or_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "supplies the input image and prompt"):
            self._resolve(
                case_id="scene",
                camera_path=self.root / "external.npy",
                prompt="Do not mix modes.",
            )

    def test_custom_mode_keeps_pinhole_default(self) -> None:
        example = self._resolve(
            input_image=self.root / "custom.png",
            prompt="A custom scene.",
            camera_path=self.root / "custom.npy",
            x_fov=1.1,
        )

        self.assertEqual(example["id"], "custom")
        self.assertEqual(example["xi"], 0.0)

    def test_pose_loader_accepts_homogeneous_pose(self) -> None:
        path = self.root / "pose.npy"
        pose = np.repeat(np.eye(4, dtype=np.float32)[None], 81, axis=0)
        np.save(path, pose)

        loaded = load_pose(path, 81)

        self.assertEqual(loaded.shape, (81, 3, 4))
        self.assertEqual(loaded.dtype, np.float32)

    def test_pose_loader_rejects_wrong_length(self) -> None:
        path = self.root / "pose.npy"
        np.save(path, np.zeros((80, 3, 4), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, r"Expected pose \[81,3,4\]"):
            load_pose(path, 81)

    def test_pose_loader_rejects_non_finite_values(self) -> None:
        path = self.root / "pose.npy"
        pose = np.zeros((81, 3, 4), dtype=np.float32)
        pose[10, 0, 3] = np.nan
        np.save(path, pose)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_pose(path, 81)


if __name__ == "__main__":
    unittest.main()
