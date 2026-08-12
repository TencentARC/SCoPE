#!/usr/bin/env bash
# Generate a camera-controlled video with SCoPE.
#
# Edit the variables below, then run:  bash scripts/inference.sh
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-checkpoints/SCoPE}"
CASE="${CASE:-omni-misty-forest}"
TRAJECTORY="${TRAJECTORY:-truck_right}"
OUTPUT="${OUTPUT:-outputs/${CASE}__${TRAJECTORY}.mp4}"

python inference.py \
  --model_path "${MODEL_PATH}" \
  --case "${CASE}" \
  --trajectory "${TRAJECTORY}" \
  --output_path "${OUTPUT}"

# Generate every bundled trajectory for one scene with a single model load:
#   python inference.py --model_path "${MODEL_PATH}" --case "${CASE}" \
#       --all_trajectories --output_dir "outputs/${CASE}"
