#!/usr/bin/env bash
# Public, precomputed Gaussian output from the Splatt3r authors; no training.
set -euo pipefail
SCGS_DATA_ROOT="${SCGS_DATA_ROOT:-/root/autodl-tmp}"
SCGS_MODEL="$SCGS_DATA_ROOT/scgs-downloads/splatt3r-example.ply"
SCGS_MODEL_SHA256=bae61a6c6d348fc05f882b9158b41c01df277a8424e2ad25db55e6ed2422d66e
SCGS_MODEL_REMOTE=brandonsmart/splatt3r_v1.0/resolve/main/demo_examples/in_the_wild_1.ply
mkdir -p "$(dirname -- "$SCGS_MODEL")"
if [[ -f "$SCGS_MODEL" ]] && printf '%s  %s\n' "$SCGS_MODEL_SHA256" "$SCGS_MODEL" | sha256sum --check --status; then
    printf 'Verified existing model: %s\n' "$SCGS_MODEL"
    exit 0
fi
if [[ -e "$SCGS_MODEL" ]]; then
    printf 'Existing file has a different checksum; refusing to overwrite: %s\n' "$SCGS_MODEL" >&2
    exit 1
fi
if ! curl -fL --connect-timeout 10 --max-time 180 "https://huggingface.co/$SCGS_MODEL_REMOTE" -o "$SCGS_MODEL.part"; then
    curl -fL --retry 2 --connect-timeout 15 --max-time 300 "https://hf-mirror.com/$SCGS_MODEL_REMOTE" -o "$SCGS_MODEL.part"
fi
printf '%s  %s\n' "$SCGS_MODEL_SHA256" "$SCGS_MODEL.part" | sha256sum --check
mv "$SCGS_MODEL.part" "$SCGS_MODEL"
