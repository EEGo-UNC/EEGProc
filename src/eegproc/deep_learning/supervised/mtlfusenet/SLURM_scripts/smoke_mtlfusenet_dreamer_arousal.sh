#!/bin/bash
#SBATCH --job-name=smoke_mtlfusenet_arousal
#SBATCH --output=smoke_mtlfusenet_arousal_%j.out
#SBATCH --error=smoke_mtlfusenet_arousal_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00

set -euo pipefail

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="$HOME/EEGProc"
VENV_DIR="$PROJECT_DIR/venv312"
PROCESSED_DIR="$PROJECT_DIR/processed_trials"

cd "$PROJECT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Reuse the environment safely when multiple jobs start together.
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

MODULE_CUDA_ROOT=""
if [[ -n "${EBROOTCUDA:-}" && -d "${EBROOTCUDA}" ]]; then
    MODULE_CUDA_ROOT="$EBROOTCUDA"
elif [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
    MODULE_CUDA_ROOT="$CUDA_HOME"
elif command -v nvcc >/dev/null 2>&1; then
    MODULE_CUDA_ROOT="$(
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

if [[ ! -d "$PROCESSED_DIR" ]]; then
    echo "ERROR: Cached MTLFuseNet trials are missing: $PROCESSED_DIR"
    echo "Run mtl_preprocess.py before submitting this job."
    exit 1
fi

N_TRIAL_FILES="$(find "$PROCESSED_DIR" -maxdepth 1 -name 'subj*_trial*.pkl' | wc -l)"
if [[ "$N_TRIAL_FILES" -lt 2 ]]; then
    echo "ERROR: Found only $N_TRIAL_FILES cached trial files in $PROCESSED_DIR."
    exit 1
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
echo "Virtual environment: $VIRTUAL_ENV"
echo "Processed trials: $PROCESSED_DIR ($N_TRIAL_FILES files)"
echo "Module CUDA root: ${MODULE_CUDA_ROOT:-not found}"
echo "XLA CUDA root: $CUDA_XLA_ROOT"
echo "libdevice: $LIBDEVICE_PATH"

python --version
nvidia-smi

# Validate imports, GPUs, the packed-input adapter, and one GPU forward pass.
python - <<'TF_PY'
import inspect
import os
from pathlib import Path

import tensorflow as tf

from eegproc.deep_learning.supervised.mtlfusenet import mtl_model_train

print("TensorFlow version:", tf.__version__)
print("TensorFlow location:", tf.__file__)
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"))
print("Training module:", mtl_model_train.__file__)
print("Packed size:", mtl_model_train.PACKED_SIZE)
print("Adapter signature:", inspect.signature(mtl_model_train.EEGProcMTLFuseNet))

if not Path(mtl_model_train.__file__).is_file():
    raise RuntimeError("The local MTLFuseNet training module was not imported.")

gpus = tf.config.list_physical_devices("GPU")
print("Available GPUs:", gpus)
if not gpus:
    raise RuntimeError("TensorFlow cannot see the allocated GPUs.")

with tf.device("/GPU:0"):
    packed = tf.zeros((2, 1, mtl_model_train.PACKED_SIZE), tf.float32)
    model = mtl_model_train.EEGProcMTLFuseNet(prediction_batch_size=2)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    output = model(packed, training=False, sample_latent=False)

print("Probability shape:", output["probabilities"].shape)
print("Reconstruction shape:", output["recon"].shape)
if tuple(output["probabilities"].shape) != (2, 2):
    raise RuntimeError("Unexpected classifier output shape.")
TF_PY

# Two-fold arousal smoke test using the paper's DREAMER architecture defaults.
# The model uses deterministic posterior-mean inference for CV predictions.
python -m eegproc.deep_learning.supervised.mtlfusenet.mtl_model_train \
    --processed-dir "$PROCESSED_DIR" \
    --task arousal \
    --out-dir runs/smoke/mtlfusenet/DREAMER/arousal \
    --run-name dreamer_arousal_mtlfusenet \
    --max-folds 2 \
    --n-jobs 2 \
    --cpus-per-worker 4 \
    --epochs 5 \
    --batch-size 64 \
    --optimizer adam \
    --learning-rate 0.0001 \
    --weight-decay 0.0 \
    --validation-subjects 2 \
    --validation-seed 42 \
    --early-stopping-patience 3 \
    --early-stopping-min-delta 0.001 \
    --early-stopping-monitor val_accuracy \
    --early-stopping-mode max \
    --selection-level trial \
    --selection-metric accuracy \
    --decision-thresholds 0.5 \
    --threshold-selection-level trial \
    --threshold-selection-metric accuracy \
    --prediction-latent-samples 0 \
    --latent-sampling-seed 42 \
    --prediction-batch-size 128 \
    --prediction-diagnostics-every 1 \
    --prediction-diagnostics-samples 128 \
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
    --no-save-full-model \
    --hyperparameters-json '{
        "optimizer": ["adam"],
        "learning_rate": [0.0001],
        "weight_decay": [0.0],
        "vae_latent": [128],
        "gcn_dim": [32],
        "gru_units": [384],
        "beta1": [0.7],
        "beta2": [0.2],
        "beta3": [0.1],
        "focal_alpha": [0.7],
        "focal_gamma": [2.0],
        "tc_margin": [1.0],
        "dropout": [0.2],
        "prediction_batch_size": [128]
    }'
