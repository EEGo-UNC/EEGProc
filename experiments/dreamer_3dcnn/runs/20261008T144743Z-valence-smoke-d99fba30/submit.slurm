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

python experiments/dreamer_3dcnn/train.py \
    --config "$RUN_DIR/config.json" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RUN_DIR"
