#!/bin/bash
#SBATCH --job-name=20261008T144743Z-valence-smoke-d99fba30
#SBATCH --output=experiments/dreamer_3dcnn/runs/20261008T144743Z-valence-smoke-d99fba30/slurm-%j.out
#SBATCH --error=experiments/dreamer_3dcnn/runs/20261008T144743Z-valence-smoke-d99fba30/slurm-%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="${PROJECT_DIR:-$HOME/EEGProc}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/datasets}"
RUN_DIR="$PROJECT_DIR/experiments/dreamer_3dcnn/runs/20261008T144743Z-valence-smoke-d99fba30"

cd "$PROJECT_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Missing virtual environment: $VENV_DIR" >&2
    echo "Create it and install this checkout with the deep-learning extra." >&2
    exit 2
fi
if [[ ! -f "$DATA_DIR/dreamer_eeg.npy" || ! -f "$DATA_DIR/dreamer_labels.npy" ]]; then
    echo "Missing DREAMER arrays in $DATA_DIR" >&2
    exit 2
fi

source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR=cuda_malloc_async
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# TensorFlow/XLA does not automatically discover the libdevice shipped by
# Longleaf's CUDA module. Resolve it from the active module (or the Python CUDA
# packages) and pass its CUDA root explicitly before TensorFlow starts.
MODULE_CUDA_ROOT=""
if [[ -n "${EBROOTCUDA:-}" && -d "$EBROOTCUDA" ]]; then
    MODULE_CUDA_ROOT="$EBROOTCUDA"
elif [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME" ]]; then
    MODULE_CUDA_ROOT="$CUDA_HOME"
elif command -v nvcc >/dev/null 2>&1; then
    MODULE_CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi

find_libdevice() {
    local roots=()
    [[ -n "$MODULE_CUDA_ROOT" && -d "$MODULE_CUDA_ROOT" ]] && roots+=("$MODULE_CUDA_ROOT")
    [[ -d "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" ]] && roots+=("$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia")
    [[ -d /usr/local/cuda ]] && roots+=("/usr/local/cuda")
    [[ -d /opt/cuda ]] && roots+=("/opt/cuda")
    [[ ${#roots[@]} -eq 0 ]] && return 0
    find "${roots[@]}" -type f -path "*/nvvm/libdevice/libdevice.10.bc" -print -quit 2>/dev/null || true
}

LIBDEVICE_PATH="$(find_libdevice)"
if [[ -z "$LIBDEVICE_PATH" || ! -f "$LIBDEVICE_PATH" ]]; then
    echo "Unable to locate nvvm/libdevice/libdevice.10.bc." >&2
    echo "Loaded CUDA root: ${MODULE_CUDA_ROOT:-<none>}" >&2
    echo "Try: python -m pip install nvidia-cuda-nvcc-cu12" >&2
    exit 2
fi
CUDA_XLA_ROOT="${LIBDEVICE_PATH%/nvvm/libdevice/libdevice.10.bc}"
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_cuda_data_dir=$CUDA_XLA_ROOT"
[[ -n "$MODULE_CUDA_ROOT" ]] && export CUDA_HOME="$MODULE_CUDA_ROOT" CUDA_PATH="$MODULE_CUDA_ROOT"

echo "CUDA XLA root: $CUDA_XLA_ROOT"
echo "libdevice: $LIBDEVICE_PATH"

python experiments/dreamer_3dcnn/train.py \
    --config "$RUN_DIR/config.json" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RUN_DIR"
