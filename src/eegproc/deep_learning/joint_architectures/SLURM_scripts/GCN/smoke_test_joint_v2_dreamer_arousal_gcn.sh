#!/bin/bash
#SBATCH --job-name=smoke_dreamer_arousal_gcn
#SBATCH --output=smoke_dreamer_arousal_gcn_%j.out
#SBATCH --error=smoke_dreamer_arousal_gcn_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="$HOME/EEGProc"
VENV_DIR="$PROJECT_DIR/venv312"

cd "$PROJECT_DIR"

# Reuse the existing environment so the smoke test starts quickly. If it does
# not exist yet, create it and install the project requirements once.
CREATED_VENV=0
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python -m venv "$VENV_DIR"
    CREATED_VENV=1
fi

source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if [[ "$CREATED_VENV" -eq 1 ]]; then
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi

# Determine the CUDA toolkit root exposed by the Longleaf module.
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

# Some TensorFlow pip installations do not include libdevice. Install the
# CUDA NVCC support package only when the file is actually missing.
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
    echo "ERROR: Unable to locate libdevice.10.bc after CUDA setup."
    echo "EBROOTCUDA=${EBROOTCUDA:-not set}"
    echo "CUDA_HOME=${CUDA_HOME:-not set}"
    echo "MODULE_CUDA_ROOT=${MODULE_CUDA_ROOT:-not found}"
    echo "VIRTUAL_ENV=${VIRTUAL_ENV:-not set}"
    echo "nvcc=$(command -v nvcc || true)"
    exit 1
fi

# XLA expects the directory that contains nvvm/, not nvvm/libdevice itself.
CUDA_XLA_ROOT="${LIBDEVICE_PATH%/nvvm/libdevice/libdevice.10.bc}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_XLA_ROOT}"

# Preserve the module CUDA root for ordinary CUDA tools and libraries.
if [[ -n "$MODULE_CUDA_ROOT" ]]; then
    export CUDA_HOME="$MODULE_CUDA_ROOT"
    export CUDA_PATH="$MODULE_CUDA_ROOT"
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
echo "Virtual environment: $VIRTUAL_ENV"
echo "Module CUDA root: ${MODULE_CUDA_ROOT:-not found}"
echo "XLA CUDA root: $CUDA_XLA_ROOT"
echo "libdevice: $LIBDEVICE_PATH"
echo "XLA_FLAGS: $XLA_FLAGS"

python --version
nvidia-smi

# Validate TensorFlow, GPU visibility, and the exp-gradient operation that
# previously exposed a missing-libdevice XLA failure.
python - <<'TF_PY'
import os
import site
import sys
from pathlib import Path

import tensorflow as tf

print("Python executable:", sys.executable)
print("Virtual environment:", sys.prefix)
print("User site enabled:", site.ENABLE_USER_SITE)
print("TensorFlow version:", tf.__version__)
print("TensorFlow location:", tf.__file__)
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"))

gpus = tf.config.list_physical_devices("GPU")
print("Available GPUs:", gpus)

if not gpus:
    raise RuntimeError("TensorFlow cannot see the allocated GPU.")

xla_flags = os.environ.get("XLA_FLAGS", "")
prefix = "--xla_gpu_cuda_data_dir="
xla_root = next(
    (
        token[len(prefix):]
        for token in xla_flags.split()
        if token.startswith(prefix)
    ),
    None,
)

if not xla_root:
    raise RuntimeError("XLA_FLAGS does not contain xla_gpu_cuda_data_dir.")

libdevice = Path(xla_root) / "nvvm" / "libdevice" / "libdevice.10.bc"
print("Resolved libdevice:", libdevice)

if not libdevice.is_file():
    raise RuntimeError(f"libdevice does not exist at {libdevice}")

with tf.device("/GPU:0"):
    x = tf.Variable([0.0, 1.0], dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(tf.exp(x))
    gradient = tape.gradient(loss, x)

print("GPU exp-gradient test:", gradient.numpy())
TF_PY

# Fast end-to-end smoke test matching the official GCN run configuration.
# One fold, one hyperparameter configuration, and two latent samples keep it quick.
python -m src.eegproc.deep_learning.joint_architectures.joint_v2_autoencoder_vc_train \
    --raw-eeg-npy datasets/dreamer_eeg.npy \
    --raw-labels-npy datasets/dreamer_labels.npy \
    --label-dimension arousal \
    --encoder-type gcn \
    --n-channels 14 \
    --n-bands 4 \
    --out-dir runs/joint_autoencoder_vc_v2/GCN \
    --run-name dreamer_arousal_vaevc_gcn \
    --max-folds 2 \
    --n-jobs 1 \
    --cpus-per-worker 2 \
    --outer-verbose 2 \
    --final-verbose 2 \
    --prediction-latent-samples 5 \
    --latent-sampling-seed 42 \
    --seed 42 \
    --validation-subjects 4 \
    --validation-seed 42 \
    --early-stopping-patience 40 \
    --early-stopping-min-delta 0.002 \
    --window-sec 4.0 \
    --window-overlap 0.0 \
    --label-threshold-mode subject_median \
    --no-class-weight \
    --batch-size 64 \
    --selection-level window \
    --selection-metric f1 \
    --early-stopping-monitor val_window_accuracy \
    --early-stopping-mode max \
    --final-epoch-strategy median \
    --hyperparameters-json '{
        "epochs": [
            300
        ],
        "batch_size": [
            64
        ],
        "learning_rate": [
            0.0001
        ],
        "ae_loss_weight": [
            0.3
        ],
        "vc_loss_weight": [
            1.0
        ],
        "vc_alpha": [
            1.0
        ],
        "vc_beta": [
            0.2
        ],
        "vc_gamma": [
            0.0
        ],
        "vc_lambda": [
            0.1
        ],
        "vae_beta": [
            0.3
        ],
        "t_down": [
            2
        ],
        "emb_dim": [
            32
        ],
        "dropout": [
            0.3
        ],
        "gcn_units": [
            [
                32,
                16
            ]
        ],
        "temporal_pool_sizes": [
            [
                2
            ]
        ],
        "activation": [
            "relu"
        ],
        "use_batch_norm": [
            false
        ],
        "bilstm_units": [
            64
        ],
        "bilstm_layers": [
            1
        ],
        "bilstm_dropout": [
            0.3
        ]
    }'