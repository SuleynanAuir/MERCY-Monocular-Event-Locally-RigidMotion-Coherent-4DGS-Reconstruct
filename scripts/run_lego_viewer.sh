#!/usr/bin/env bash
# Separate loopback service: does not stop or replace the fan viewer on 8890.
set -euo pipefail
SCGS_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCGS_REPO_DIR"
source scripts/autodl_env.sh
exec python -u viewer.py \
  --sequence_path "${SCGS_LEGO_SEQUENCE:-/root/autodl-tmp/scgs-results/lego-moving-gaussians/lego_motion.npz}" \
  --port "${SCGS_LEGO_PORT:-8891}" --cache auto
