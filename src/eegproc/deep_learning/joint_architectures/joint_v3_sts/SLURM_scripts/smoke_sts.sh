#!/bin/bash
#SBATCH --job-name=smoke_sts_dreamer_arousal
#SBATCH --output=smoke_sts_dreamer_arousal_%j.out
#SBATCH --error=smoke_sts_dreamer_arousal_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00

set -euo pipefail

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="$HOME/EEGProc"
VENV_DIR="$PROJECT_DIR/venv312"

cd "$PROJECT_DIR"

# Reuse the virtual environment instead of recreating it for every job.
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Serialize environment updates so concurrent jobs do not modify the same
# virtual environment simultaneously.
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

# Some TensorFlow pip installations omit libdevice. Install the CUDA NVCC
# support package only when the file is missing.
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

# XLA expects the directory containing nvvm/, not nvvm/libdevice itself.
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
import inspect
import os
import site
import sys
from pathlib import Path

import tensorflow as tf

from src.eegproc.deep_learning.joint_architectures.joint_v3_sts \
    import joint_sts_model

print("Python executable:", sys.executable)
print("Virtual environment:", sys.prefix)
print("User site enabled:", site.ENABLE_USER_SITE)
print("TensorFlow version:", tf.__version__)
print("TensorFlow location:", tf.__file__)
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"))
print("STS model module:", joint_sts_model.__file__)
print(
    "STS builder signature:",
    inspect.signature(joint_sts_model.build_joint_sts_model),
)

required_builder_parameters = {
    "decoder_temporal_units",
    "decoder_bilstm_layers",
    "decoder_graph_output_units",
    "decoder_branch_feature_dim",
    "decoder_fusion_units",
    "decoder_dropout",
}
actual_builder_parameters = set(
    inspect.signature(
        joint_sts_model.build_joint_sts_model
    ).parameters
)
missing_builder_parameters = (
    required_builder_parameters - actual_builder_parameters
)
if missing_builder_parameters:
    raise RuntimeError(
        "The imported STS model is not the dual-path decoder version. "
        f"Missing builder parameters: {sorted(missing_builder_parameters)}"
    )

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

# Official DREAMER arousal LOSO grid for the dual-path joint STS model.
#
# Each train_step alternates:
#   1. classification + subject adversity + cross-subject SupCon
#   2. fused VAE reconstruction + posterior KL
#
# Four outer-fold workers are requested for the four allocated L40S GPUs.
python -m src.eegproc.deep_learning.joint_architectures.joint_v3_sts.joint_sts_model_train \
    --raw-eeg-npy datasets/remove_gamma/dreamer_eeg.npy \
    --raw-labels-npy datasets/remove_gamma/dreamer_labels.npy \
    --label-dimension arousal \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir runs/smoke/joint_sts/DREAMER/arousal \
    --run-name dreamer_arousal_joint_sts \
    --max-folds 2 \
    --n-jobs 2 \
    --cpus-per-worker 4 \
    --outer-verbose 2 \
    --final-verbose 2 \
    --prediction-latent-samples 20 \
    --latent-sampling-seed 42 \
    --seed 42 \
    --validation-subjects 6 \
    --validation-seed 42 \
    --label-threshold-mode global \
    --median-label 3 \
    --early-stopping-patience 20 \
    --early-stopping-min-delta 0.002 \
    --window-sec 4.0 \
    --window-overlap 0.0 \
    --window-normalization global_rms \
    --no-class-weight \
    --early-stopping-monitor val_accuracy \
    --early-stopping-mode max \
    --selection-level trial \
    --selection-metric accuracy \
    --decision-thresholds \
        0.20 0.25 0.30 0.35 0.40 0.45 0.50 \
        0.55 0.60 0.65 0.70 0.75 0.80 \
    --threshold-selection-level trial \
    --threshold-selection-metric accuracy \
    --final-epoch-strategy median \
    --hyperparameters-json '{
        "epochs": [
            100
        ],
        "batch_size": [
            64
        ],
        "optimizer": [
            "adamw"
        ],
        "classification_learning_rate": [
            0.00003
        ],
        "vae_learning_rate": [
            0.00002
        ],
        "weight_decay": [
            0.00005
        ],
        "classification_steps_per_batch": [
            2
        ],
        "vae_steps_per_batch": [
            1
        ],

        "use_subject_adversarial": [
            true
        ],
        "subject_adversarial_weight": [
            0.1
        ],
        "subject_loss_weight": [
            0.0
        ],
        "subject_hidden_units": [
            128
        ],
        "subject_dropout": [
            0.0
        ],
        "subject_latent_mode": [
            "mean"
        ],

        "use_supcon": [
            false
        ],
        "supcon_weight": [
            0.0
        ],
        "supcon_temperature": [
            0.1
        ],
        "supcon_cross_subject_only": [
            true
        ],

        "classification_loss_weight": [
            1.0
        ],
        "vae_loss_weight": [
            0.0
        ],
        "vae_beta": [
            0.3
        ],
        "label_smoothing": [
            0.0
        ],

        "classifier_head": [
            "hybrid"
        ],

        "classification_hidden_units": [
            128
        ],
        "classification_dropout": [
            0.3
        ],

        focal_alpha: [0.725, 0.275],
        focal_gamma: 1.0,
        "vc_alpha": [
            1.0
        ],
        "vc_beta": [
            0.0, 0.5
        ],
        "vc_gamma": [
            0.0
        ],
        "vc_lambda": [
            0.0
        ],

        "t_down": [
            2
        ],
        "temporal_pool_sizes": [
            [
                2
            ]
        ],

        "bilstm_units": [
            256
        ],
        "bilstm_layers": [
            1
        ],
        "bilstm_dropout": [
            0.3
        ],
        "temporal_emb_dim": [
            64
        ],

        "gcn_units": [
            [
                128,
                64
            ]
        ],
        "spectral_emb_dim": [
            128
        ],
        "gcn_dropout": [
            0.2
        ],
        "gcn_activation": [
            "relu"
        ],
        "gcn_use_batch_norm": [
            false
        ],
        "graph_self_loop_bias": [
            2.0
        ],
        "graph_identity_mix": [
            0.0
        ],
        "graph_adjacency_reg_weight": [
            0.0001
        ],

        "fusion_dim": [
            256
        ],
        "latent_features": [
            128
        ],
        "fusion_dropout": [
            0.2
        ],
        "activation": [
            "relu"
        ],

        "decoder_temporal_units": [
            128
        ],
        "decoder_bilstm_layers": [
            1
        ],
        "decoder_graph_output_units": [
            32
        ],
        "decoder_branch_feature_dim": [
            64
        ],
        "decoder_fusion_units": [
            128
        ],
        "decoder_dropout": [
            0.2
        ],
        "reconstruction_loss": [
            "mse"
        ]
    }'
    