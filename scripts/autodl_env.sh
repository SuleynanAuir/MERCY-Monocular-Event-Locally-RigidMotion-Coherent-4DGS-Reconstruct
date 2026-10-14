#!/usr/bin/env bash
# Source this before using the AutoDL environment.
export SCGS_DATA_ROOT="${SCGS_DATA_ROOT:-/root/autodl-tmp}"
source "$SCGS_DATA_ROOT/scgs-env/bin/activate"
export PIP_CACHE_DIR="$SCGS_DATA_ROOT/scgs-cache/pip"
export TORCH_HOME="$SCGS_DATA_ROOT/scgs-cache/torch"
export TMPDIR="$SCGS_DATA_ROOT/scgs-cache/build"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export PATH="$CUDA_HOME/bin:$PATH"
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    # Instances can change GPU families (T4/2080 Ti: 7.5; RTX 3080: 8.6).
    # A caller-provided architecture still takes precedence for cross builds.
    TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}")')"
    export TORCH_CUDA_ARCH_LIST
fi
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME" "$TMPDIR"
