#!/usr/bin/env bash
# Launch SCoPE RDPO high-only training.
#
# 1. Prepare each dataset and estimate its per-clip near-depth, e.g.:
#      python scripts/estimate_near_depth.py --dataset realestate10k \
#          --data_root /path/to/RealEstate10K --split train \
#          --output near_depth/realestate10k_train.json
#    (repeat for dl3dv, panshot, and omniworld; build the OmniWorld index first
#     with scripts/build_omniworld_index.py).
# 2. Point configs/train_rdpo_high_only.yaml at your dataset roots and JSONs.
# 3. Run:  bash scripts/train.sh
set -euo pipefail

CONFIG="${CONFIG:-configs/train_rdpo_high_only.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"

python train.py \
  --config "${CONFIG}" \
  --num_gpus "${NUM_GPUS}" \
  "$@"
