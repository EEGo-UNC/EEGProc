#!/bin/bash
#SBATCH --job-name=smoke_v4_dreamer_valence
#SBATCH --output=smoke_v4_dreamer_valence_%j.out
#SBATCH --error=smoke_v4_dreamer_valence_%j.err
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
USE_MLDG="${USE_MLDG:-false}"
if [[ "$USE_MLDG" == "true" ]]; then
    MLDG_FLAG="--use-mldg"
else
    MLDG_FLAG="--no-mldg"
fi

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
import numpy as np
import tensorflow as tf

from src.eegproc.deep_learning.joint_architectures.joint_v4_sts import (
    joint_sts_model,
)

print("TensorFlow:", tf.__version__)
print("V4 module:", joint_sts_model.__file__)
signature = inspect.signature(joint_sts_model.build_joint_sts_model)
print("Builder:", signature)

required = {
    "classification_level",
    "bilstm_emb_dim",
    "use_vae",
    "use_subject_adversarial",
    "use_mldg",
}
actual = set(signature.parameters)
missing = required - actual
api_version = getattr(
    joint_sts_model,
    "JOINT_STS_BUILDER_API_VERSION",
    0,
)
print("V4 builder API version:", api_version)

if missing or api_version < 6:
    raise RuntimeError(
        "joint_v4_sts builder is stale/incomplete. "
        f"Missing={sorted(missing)}, API version={api_version}; expected >=6."
    )

encoder_module = joint_sts_model.BandSeparatedGCNEncoder.__module__
print("BandSeparatedGCNEncoder module:", encoder_module)
if "GCN_band_separated" not in encoder_module:
    raise RuntimeError(
        "joint_v4_sts is not importing BandSeparatedGCNEncoder from "
        "GCN_band_separated.py."
    )

gpus = tf.config.list_physical_devices("GPU")
print("Available GPUs:", gpus)
if not gpus:
    raise RuntimeError("TensorFlow cannot see the allocated GPU.")

# Runtime architecture preflight: trial-level classifier.
trial_shape = (3, 8, 42)
x_trial = tf.zeros((2, *trial_shape), dtype=tf.float32)

baseline = joint_sts_model.build_joint_sts_model(
    input_shape=trial_shape,
    classification_level="trial",
    n_classes=2,
    n_channels=14,
    n_bands=3,
    t_down=2,
    temporal_pool_sizes=(2,),
    gcn_units=(8,),
    spectral_emb_dim=8,
    bilstm_units=4,
    n_bilstm_layers=1,
    bilstm_emb_dim=4,
    classification_hidden_units=4,
    use_vae=False,
    use_subject_adversarial=False,
    use_mldg=False,
)
baseline_logits = baseline(x_trial, training=False)
print("Baseline trial logits shape:", baseline_logits.shape)
if tuple(baseline_logits.shape) != (2, 2):
    raise RuntimeError(
        f"Unexpected baseline logits shape: {baseline_logits.shape}"
    )

# Runtime VAE preflight.
vae_model = joint_sts_model.build_joint_sts_model(
    input_shape=trial_shape,
    classification_level="trial",
    n_classes=2,
    n_channels=14,
    n_bands=3,
    t_down=2,
    temporal_pool_sizes=(2,),
    gcn_units=(8,),
    spectral_emb_dim=8,
    bilstm_units=4,
    n_bilstm_layers=1,
    bilstm_emb_dim=4,
    classification_hidden_units=4,
    use_vae=True,
    vae_loss_weight=0.1,
    vae_beta=0.05,
    vae_learning_rate=5e-5,
    use_subject_adversarial=False,
    use_mldg=False,
)
vae_losses, vae_outputs = vae_model._vae_losses(
    x_trial,
    training=False,
)
print(
    "VAE preflight:",
    {key: float(value.numpy()) for key, value in vae_losses.items()},
)
print("VAE reconstruction shape:", vae_outputs["reconstruction"].shape)

# Runtime subject-adversarial preflight. This specifically verifies that the
# fold-local subject head is created before Keras locks model state.
subject_model = joint_sts_model.build_joint_sts_model(
    input_shape=trial_shape,
    classification_level="trial",
    n_classes=2,
    n_channels=14,
    n_bands=3,
    t_down=2,
    temporal_pool_sizes=(2,),
    gcn_units=(8,),
    spectral_emb_dim=8,
    bilstm_units=4,
    n_bilstm_layers=1,
    bilstm_emb_dim=4,
    classification_hidden_units=4,
    use_vae=False,
    use_subject_adversarial=True,
    subject_adversarial_weight=0.3,
    subject_loss_weight=0.3,
    subject_hidden_units=4,
    use_mldg=False,
)
prepared = subject_model.prepare_fit_inputs(
    np.zeros((2, *trial_shape), dtype=np.float32),
    np.asarray([10, 11], dtype=np.int32),
)
subject_losses, subject_outputs = subject_model._classification_losses(
    prepared,
    tf.constant([0, 1], dtype=tf.int32),
    training=False,
)
print("Subject loss:", float(subject_losses["subject_loss"].numpy()))
print("Subject logits shape:", subject_outputs["subject_logits"].shape)

# Runtime first-order MLDG preflight.
mldg_model = joint_sts_model.build_joint_sts_model(
    input_shape=trial_shape,
    classification_level="trial",
    n_classes=2,
    n_channels=14,
    n_bands=3,
    t_down=2,
    temporal_pool_sizes=(2,),
    gcn_units=(8,),
    spectral_emb_dim=8,
    bilstm_units=4,
    n_bilstm_layers=1,
    bilstm_emb_dim=4,
    classification_hidden_units=4,
    use_vae=False,
    use_subject_adversarial=False,
    use_mldg=True,
    mldg_inner_learning_rate=1e-4,
    mldg_meta_test_weight=1.0,
)
mldg_metrics = mldg_model._mldg_train_step(
    (
        {
            "meta_train": x_trial[:1],
            "meta_test": x_trial[1:],
        },
        {
            "meta_train": tf.constant([0], dtype=tf.int32),
            "meta_test": tf.constant([1], dtype=tf.int32),
        },
    )
)
print(
    "MLDG preflight metrics:",
    {key: float(value.numpy()) for key, value in mldg_metrics.items()},
)
print("V4 runtime preflight: PASS")
TF_PY

python -m src.eegproc.deep_learning.joint_architectures.joint_v4_sts.joint_sts_model_train \
    --cv-strategy loso \
    --classification-level "$CLASSIFICATION_LEVEL" \
    "$MLDG_FLAG" \
    --mldg-inner-learning-rate 0.0001 \
    --mldg-meta-test-weight 1.0 \
    --mldg-meta-train-subjects 6 \
    --mldg-meta-test-subjects 2 \
    --mldg-samples-per-subject 4 \
    --mldg-seed 42 \
    --validation-subjects 4 \
    --raw-eeg-npy datasets/remove_gamma/dreamer_eeg.npy \
    --raw-labels-npy datasets/remove_gamma/dreamer_labels.npy \
    --label-dimension valence \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir "runs/smoke/joint_v4_sts/DREAMER/valence/${CLASSIFICATION_LEVEL}" \
    --run-name "dreamer_valence_joint_v4_${CLASSIFICATION_LEVEL}_smoke" \
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
        "epochs": [30],
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
        "bilstm_emb_dim": [64],
        "use_vae": [
            false
        ],
        "vae_loss_weight": [
            0.1
        ],
        "vae_beta": [
            0.05
        ],
        "vae_learning_rate": [
            0.00005
        ],
        "use_subject_adversarial": [
            false
        ],
        "subject_adversarial_weight": [
            0.3
        ],
        "subject_loss_weight": [
            0.3
        ],
        "subject_hidden_units": [
            64
        ],
        "subject_dropout": [
            0.0
        ],

        "classification_hidden_units": [64],
        "classification_dropout": [0.3],
        "activation": ["relu"],
        "focal_gamma": [0.0],
        "focal_alpha": null
    }'
