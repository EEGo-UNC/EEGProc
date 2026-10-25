#!/bin/bash
# Sourced by v16 jobs. Prepare venv312 once before submitting; never install in jobs.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the EEGProc repository root}}"
# smoke.sh/full.sh invoke task.sh in a child shell; pass the resolved path.
export PROJECT_DIR
cd "$PROJECT_DIR"
module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
[[ -x "$VENV_DIR/bin/python" ]] || { echo "Prepare $VENV_DIR before submitting." >&2; exit 1; }
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TF_GPU_ALLOCATOR=cuda_malloc_async
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
SIC16="$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv16"
MODULE_CUDA_ROOT="${EBROOTCUDA:-${CUDA_HOME:-}}"
if [[ -z "$MODULE_CUDA_ROOT" ]] && command -v nvcc >/dev/null 2>&1; then
    MODULE_CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi
CUDA_ROOTS=()
for candidate in "$MODULE_CUDA_ROOT" "$VENV_DIR/lib/python3.12/site-packages/nvidia" /usr/local/cuda /opt/cuda; do
    [[ -d "$candidate" ]] && CUDA_ROOTS+=("$candidate")
done
[[ ${#CUDA_ROOTS[@]} -gt 0 ]] || { echo "Cannot find CUDA." >&2; exit 1; }
LIBDEVICE_PATH="$(find "${CUDA_ROOTS[@]}" -type f -path '*/nvvm/libdevice/libdevice.10.bc' -print -quit 2>/dev/null || true)"
[[ -f "$LIBDEVICE_PATH" ]] || { echo "Cannot find libdevice.10.bc." >&2; exit 1; }
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=${LIBDEVICE_PATH%/nvvm/libdevice/libdevice.10.bc}"
if [[ -n "$MODULE_CUDA_ROOT" ]]; then
    export CUDA_HOME="$MODULE_CUDA_ROOT" CUDA_PATH="$MODULE_CUDA_ROOT"
fi
