#!/usr/bin/env bash
# Use the AutoDL image's matching PyTorch/CUDA pair without changing conda base.
set -euo pipefail
SCGS_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCGS_DATA_ROOT="${SCGS_DATA_ROOT:-/root/autodl-tmp}"
SCGS_BASE_PYTHON="${SCGS_BASE_PYTHON:-/root/miniconda3/bin/python}"
"$SCGS_BASE_PYTHON" -m venv --system-site-packages "$SCGS_DATA_ROOT/scgs-env"
source "$SCGS_PROJECT_DIR/scripts/autodl_env.sh"
cd "$SCGS_PROJECT_DIR"
mkdir -p "$SCGS_DATA_ROOT/scgs-downloads" "$SCGS_DATA_ROOT/scgs-deps"
python -m pip install -r requirements-autodl.txt

fetch_source() {
    local url="$1" archive="$2" target="$3"
    if [[ ! -f "$archive" ]]; then
        curl -fL --retry 2 --connect-timeout 15 --max-time 300 "$url" -o "$archive.part"
        mv "$archive.part" "$archive"
    fi
    mkdir -p "$target"
    tar -xzf "$archive" --strip-components=1 -C "$target"
}

if [[ ! -f submodules/diff-gaussian-rasterization/setup.py ]]; then
    fetch_source \
        https://codeload.github.com/ashawkey/diff-gaussian-rasterization/tar.gz/d986da0d4cf2dfeb43b9a379b6e9fa0a7f3f7eea \
        "$SCGS_DATA_ROOT/scgs-downloads/rasterizer.tar.gz" submodules/diff-gaussian-rasterization
fi
if [[ ! -f submodules/diff-gaussian-rasterization/third_party/glm/glm/glm.hpp ]]; then
    fetch_source https://codeload.github.com/g-truc/glm/tar.gz/refs/tags/0.9.9.8 \
        "$SCGS_DATA_ROOT/scgs-downloads/glm.tar.gz" submodules/diff-gaussian-rasterization/third_party/glm
fi
if [[ ! -f submodules/simple-knn/setup.py ]]; then
    fetch_source https://gitlab.inria.fr/bkerbl/simple-knn/-/archive/44f7642/simple-knn-44f7642.tar.gz \
        "$SCGS_DATA_ROOT/scgs-downloads/simple-knn.tar.gz" submodules/simple-knn
fi
if [[ ! -f "$SCGS_DATA_ROOT/scgs-deps/pytorch3d-0.7.5/setup.py" ]]; then
    fetch_source https://codeload.github.com/facebookresearch/pytorch3d/tar.gz/refs/tags/v0.7.5 \
        "$SCGS_DATA_ROOT/scgs-downloads/pytorch3d-0.7.5.tar.gz" "$SCGS_DATA_ROOT/scgs-deps/pytorch3d-0.7.5"
fi

python -m pip install --no-build-isolation --no-deps ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
FORCE_CUDA=1 python -m pip install --no-build-isolation --no-deps "$SCGS_DATA_ROOT/scgs-deps/pytorch3d-0.7.5"
python -c 'import torch, pytorch3d.ops, diff_gaussian_rasterization, simple_knn._C; print(torch.cuda.get_device_name())'
