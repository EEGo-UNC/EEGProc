#!/bin/bash
#SBATCH --job-name=smoke_v4_dreamer_arousal
#SBATCH --output=smoke_v4_dreamer_arousal_%j.out
#SBATCH --error=smoke_v4_dreamer_arousal_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:30:00

set -euo pipefail

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="$HOME/EEGProc"
VENV_DIR="$PROJECT_DIR/venv312"
CLASSIFICATION_LEVEL="${CLASSIFICATION_LEVEL:-trial}"

if [[ "$CLASSIFICATION_LEVEL" == "trial" ]]; then
    BATCH_SIZE=8
else
    BATCH_SIZE=64
fi

cd "$PROJECT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if command -v flock >/dev/null 2>&1; then
    (
        flock -x 9
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
    ) 9>"$PROJECT_DIR/.venv312_install.lock"
else
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi

MODULE_CUDA_ROOT=""
if [[ -n "${EBROOTCUDA:-}" && -d "${EBROOTCUDA}" ]]; then
    MODULE_CUDA_ROOT="$EBROOTCUDA"
elif [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
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
if [[ -z "$LIBDEVICE_PATH" ]]; then
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

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Classification level: $CLASSIFICATION_LEVEL"
echo "Batch size: $BATCH_SIZE"

python --version
nvidia-smi

python - <<'TF_PY'
import inspect
import tensorflow as tf
from src.eegproc.deep_learning.joint_architectures.joint_v4_sts import joint_sts_model

print("TensorFlow:", tf.__version__)
print("V4 module:", joint_sts_model.__file__)
print("Builder:", inspect.signature(joint_sts_model.build_joint_sts_model))
if "classification_level" not in inspect.signature(
    joint_sts_model.build_joint_sts_model
).parameters:
    raise RuntimeError("joint_v4_sts builder is missing classification_level.")
if not tf.config.list_physical_devices("GPU"):
    raise RuntimeError("TensorFlow cannot see the allocated GPU.")
TF_PY

python -m src.eegproc.deep_learning.joint_architectures.joint_v4_sts.joint_sts_model_train \
    --cv-strategy loso \
    --classification-level "$CLASSIFICATION_LEVEL" \
    --validation-subjects 4 \
    --raw-eeg-npy datasets/remove_gamma/dreamer_eeg.npy \
    --raw-labels-npy datasets/remove_gamma/dreamer_labels.npy \
    --label-dimension arousal \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir "runs/smoke/joint_v4_sts/DREAMER/arousal/${CLASSIFICATION_LEVEL}" \
    --run-name "dreamer_arousal_joint_v4_${CLASSIFICATION_LEVEL}_smoke" \
    --max-folds 2 \
    --n-jobs 2 \
    --cpus-per-worker 2 \
    --outer-verbose 2 \
    --final-verbose 2 \
    --seed 42 \
    --label-threshold-mode global \
    --median-label 3 \
    --window-sec 4.0 \
    --window-overlap 0.0 \
    --window-normalization global_rms \
    --no-class-weight \
    --selection-level trial \
    --selection-metric accuracy \
    --decision-thresholds 0.5 \
    --threshold-selection-level trial \
    --threshold-selection-metric accuracy \
    --early-stopping-patience 10 \
    --early-stopping-min-delta 0.002 \
    --early-stopping-monitor val_accuracy \
    --early-stopping-mode max \
    --final-epoch-strategy median \
    --skip-no-validation-loso-before-final \
    --batch-size "$BATCH_SIZE" \
    --hyperparameters-json '{
        "epochs": [20],
        "optimizer": ["adamw"],
        "classification_learning_rate": [0.0001],
        "weight_decay": [0.00005],
        "gcn_units": [[64, 32]],
        "spectral_emb_dim": [64],
        "gcn_dropout": [0.2],
        "gcn_activation": ["relu"],
        "gcn_use_batch_norm": [false],
        "graph_self_loop_bias": [2.0],
        "graph_identity_mix": [0.0],
        "graph_adjacency_reg_weight": [0.0001],
        "t_down": [2],
        "temporal_pool_sizes": [[2]],
        "bilstm_units": [64],
        "bilstm_layers": [1],
        "bilstm_dropout": [0.3],
        "classification_hidden_units": [64],
        "classification_dropout": [0.3],
        "activation": ["relu"],
        "focal_gamma": [0.0],
        "focal_alpha": null
    }'
