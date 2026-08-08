#!/bin/bash
#SBATCH --job-name=mtlfuse_train
#SBATCH --output=mtlfuse_train_%j.out
#SBATCH --error=mtlfuse_train_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00

set -euo pipefail

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

# --- User-editable variables ---
PROJECT_DIR="$HOME/EEGProc"
VENV_DIR="$PROJECT_DIR/venv312"
PROCESSED_DIR="$PROJECT_DIR/processed_trials"
TASK="arousal"                  # or "valence"
OUT_DIR="$PROJECT_DIR/runs/mtlfusenet"
RUN_NAME="dreamer_${TASK}_mtlfusenet"
EPOCHS=50
BATCH_SIZE=64
N_JOBS=4
CPUS_PER_WORKER=4
VALIDATION_SUBJECTS=2
EARLY_STOPPING_PATIENCE=10
# -------------------------------

cd "$PROJECT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Install dependencies safely when jobs start concurrently
if command -v flock >/dev/null 2>&1; then
    (
        flock -x 9
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
        python -m pip install -e .
    ) 9>"$PROJECT_DIR/.venv312_install.lock"
else
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .
fi

# Find libdevice for XLA if needed (copied from smoke script)
MODULE_CUDA_ROOT=""
if [[ -n "${EBROOTCUDA:-}" && -d "${EBROOTCUDA}" ]]; then
    MODULE_CUDA_ROOT="$EBROOTCUDA"
elif [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
    MODULE_CUDA_ROOT="$CUDA_HOME"
elif command -v nvcc >/dev/null 2>&1; then
    MODULE_CUDA_ROOT="$ (
        dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")"
    )"
fi

find_libdevice() {
    local roots=()
    if [[ -n "$MODULE_CUDA_ROOT" && -d "$MODULE_CUDA_ROOT" ]]; then
        roots+=("$MODULE_CUDA_ROOT")
    fi
    if [[ -d "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" ]]; then
        roots+=("$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia")
    fi
    if [[ -d /usr/local/cuda ]]; then
        roots+=("/usr/local/cuda")
    fi
    if [[ -d /opt/cuda ]]; then
        roots+=("/opt/cuda")
    fi
    if [[ ${#roots[@]} -eq 0 ]]; then
        return 0
    fi
    find "${roots[@]}" \
        -type f \
        -path "*/nvvm/libdevice/libdevice.10.bc" \
        -print \
        -quit 2>/dev/null || true
}

LIBDEVICE_PATH="$(find_libdevice)"
if [[ -z "$LIBDEVICE_PATH" ]]; then
    echo "libdevice.10.bc was not found; installing nvidia-cuda-nvcc-cu12."
    if command -v flock >/dev/null 2>&1; then
        (
            flock -x 9
            python -m pip install --upgrade nvidia-cuda-nvcc-cu12
        ) 9>"$PROJECT_DIR/.venv312_install.lock"
    else
        python -m pip install --upgrade nvidia-cuda-nvcc-cu12
    fi
    LIBDEVICE_PATH="$(find_libdevice)"
fi

if [[ -z "$LIBDEVICE_PATH" || ! -f "$LIBDEVICE_PATH" ]]; then
    echo "ERROR: Unable to locate libdevice.10.bc."
    exit 1
fi

CUDA_XLA_ROOT="${LIBDEVICE_PATH%/nvvm/libdevice/libdevice.10.bc}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_XLA_ROOT}"
if [[ -n "$MODULE_CUDA_ROOT" ]]; then
    export CUDA_HOME="$MODULE_CUDA_ROOT"
    export CUDA_PATH="$MODULE_CUDA_ROOT"
fi

# Sanity checks
if [[ ! -d "$PROCESSED_DIR" ]]; then
    echo "ERROR: Cached MTLFuseNet trials are missing: $PROCESSED_DIR"
    exit 1
fi

N_TRIAL_FILES="$(find "$PROCESSED_DIR" -maxdepth 1 -name 'subj*_trial*.pkl' | wc -l)"
if [[ "$N_TRIAL_FILES" -lt 10 ]]; then
    echo "WARNING: Found only $N_TRIAL_FILES cached trial files in $PROCESSED_DIR."
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
echo "Virtual environment: $VIRTUAL_ENV"
echo "Processed trials: $PROCESSED_DIR ($N_TRIAL_FILES files)"
echo "XLA CUDA root: $CUDA_XLA_ROOT"
echo "libdevice: $LIBDEVICE_PATH"

python --version
nvidia-smi || true

# Run the training module (LOSO + final model). Adjust CLI flags as needed.
python -m eegproc.deep_learning.supervised.mtlfusenet.mtl_model_train \
    --processed-dir "$PROCESSED_DIR" \
    --task "$TASK" \
    --out-dir "$OUT_DIR" \
    --run-name "$RUN_NAME" \
    --max-folds 0 \
    --n-jobs $N_JOBS \
    --cpus-per-worker $CPUS_PER_WORKER \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --optimizer adam \
    --learning-rate 0.0001 \
    --weight-decay 0.0 \
    --validation-subjects $VALIDATION_SUBJECTS \
    --early-stopping-patience $EARLY_STOPPING_PATIENCE \
    --selection-level trial \
    --selection-metric accuracy \
    --prediction-latent-samples 0 \
    --latent-sampling-seed 42 \
    --prediction-batch-size 128 \
    --seed 42 \
    --final-epoch-strategy median \
    --outer-verbose 2 \
    --final-verbose 2 \
    --vae-latent 128 \
    --gcn-dim 32 \
    --gru-units 384 \
    --beta1 0.7 \
    --beta2 0.2 \
    --beta3 0.1 \
    --focal-alpha 0.7 \
    --focal-gamma 2.0 \
    --tc-margin 1.0 \
    --dropout 0.2 \
    --no-save-full-model

echo "Training finished: $(date)"
