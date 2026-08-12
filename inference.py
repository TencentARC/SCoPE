"""Generate camera-controlled videos with SCoPE."""

from __future__ import annotations

import sys


def main() -> None:
    if "--all_trajectories" in sys.argv:
        sys.argv.remove("--all_trajectories")
        from scope.case_inference import main as generate_case

        generate_case()
    else:
        from scope.inference import main as generate_video

        generate_video()


if __name__ == "__main__":
    main()
