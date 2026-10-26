#!/usr/bin/env bash
set -euo pipefail
SCGS_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCGS_PROJECT_DIR"
CUDA_VISIBLE_DEVICES=0 python train_gui.py --source_path YOUR/PATH/TO/DATASET/jumpingjacks --model_path outputs/jumpingjacks --deform_type node --node_num 512 --is_blender --eval --gui --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800
