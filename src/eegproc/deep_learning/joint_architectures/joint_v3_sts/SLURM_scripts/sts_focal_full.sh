#!/bin/bash
#SBATCH --job-name=loso_sts_restore075
#SBATCH --output=loso_sts_restore075_%j.out
#SBATCH --error=loso_sts_restore075_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

set -euo pipefail

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="$HOME/EEGProc"
VENV_DIR="$PROJECT_DIR/venv312"

cd "$PROJECT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: Python environment not found at $VENV_DIR"
    echo "Create/install the environment before submitting the full experiment."
    exit 1
fi

source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=1

# Resolve libdevice for TensorFlow/XLA using the loaded CUDA module first.
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
if [[ -z "$LIBDEVICE_PATH" || ! -f "$LIBDEVICE_PATH" ]]; then
    echo "ERROR: Unable to locate libdevice.10.bc."
    echo "Install nvidia-cuda-nvcc-cu12 in $VENV_DIR or check the CUDA module."
    exit 1
fi

CUDA_XLA_ROOT="${LIBDEVICE_PATH%/nvvm/libdevice/libdevice.10.bc}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_XLA_ROOT}"

if [[ -n "$MODULE_CUDA_ROOT" ]]; then
    export CUDA_HOME="$MODULE_CUDA_ROOT"
    export CUDA_PATH="$MODULE_CUDA_ROOT"
fi

EEG_PATH="datasets/remove_gamma/dreamer_eeg.npy"
LABEL_PATH="datasets/remove_gamma/dreamer_labels.npy"

for required_path in "$EEG_PATH" "$LABEL_PATH"; do
    if [[ ! -f "$required_path" ]]; then
        echo "ERROR: Required dataset file does not exist: $required_path"
        exit 1
    fi
done

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
echo "Virtual environment: $VIRTUAL_ENV"
echo "CUDA module root: ${MODULE_CUDA_ROOT:-not found}"
echo "XLA CUDA root: $CUDA_XLA_ROOT"
echo "libdevice: $LIBDEVICE_PATH"
python --version
nvidia-smi

# Full 23-fold DREAMER arousal LOSOCV using the configuration selected by the
# 0.7500 mean trial-accuracy smoke test.
#
# Important distinction:
#   --no-alternate-subject-sets disables the newer two-environment subject-set
#   sequence. The current JointSTSModel still performs its original separate
#   classifier and VAE optimizer updates inside train_step (2 classifier, 1 VAE),
#   because the current Python model does not expose a joint-update/no-alternation
#   mode.
python -m src.eegproc.deep_learning.joint_architectures.joint_v3_sts.joint_sts_model_train \
    --cv-strategy loso \
    --no-alternate-subject-sets \
    --skip-no-validation-loso-before-final \
    --raw-eeg-npy "$EEG_PATH" \
    --raw-labels-npy "$LABEL_PATH" \
    --label-dimension arousal \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir runs/FULL_LOSO_FOCAL_TRAIN_VAL/DREAMER/AROUSAL \
    --run-name dreamer_arousal_restored_075 \
    --n-jobs 4 \
    --cpus-per-worker 2 \
    --outer-verbose 2 \
    --final-verbose 2 \
    --prediction-latent-samples 20 \
    --latent-sampling-seed 42 \
    --seed 42 \
    --validation-subjects 6 \
    --validation-seed 42 \
    --label-threshold-mode global \
    --median-label 3 \
    --window-sec 4.0 \
    --window-overlap 0.0 \
    --window-normalization global_rms \
    --no-class-weight \
    --early-stopping-patience 20 \
    --early-stopping-min-delta 0.002 \
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
        "epochs": [100],
        "batch_size": [64],
        "optimizer": ["adamw"],
        "classification_learning_rate": [0.00003],
        "vae_learning_rate": [0.00002],
        "weight_decay": [0.00005],
        "classification_steps_per_batch": [2],
        "vae_steps_per_batch": [1],

        "use_subject_adversarial": [true],
        "subject_adversarial_weight": [0.1],
        "subject_loss_weight": [0.5],
        "subject_hidden_units": [128],
        "subject_dropout": [0.0],
        "subject_latent_mode": ["mean"],

        "use_supcon": [false],
        "supcon_weight": [0.0],
        "supcon_temperature": [0.1],
        "supcon_cross_subject_only": [true],

        "classification_loss_weight": [1.0],
        "vae_loss_weight": [0.0, 0.5],
        "vae_beta": [0.3],
        "label_smoothing": [0.05],

        "classifier_head": ["hybrid"],
        "classification_hidden_units": [128],
        "classification_dropout": [0.3],

        "focal_gamma": [2.0],
        "focal_alpha": [
            [0.275, 0.725],
            [0.55, 0.45]
            ],
        "vc_alpha": [1.0],
        "vc_beta": [0.5],
        "vc_gamma": [0.0],
        "vc_lambda": [0.0],

        "t_down": [2],
        "temporal_pool_sizes": [[2]],

        "bilstm_units": [256],
        "bilstm_layers": [1],
        "bilstm_dropout": [0.3],
        "temporal_emb_dim": [64],

        "gcn_units": [[128, 64]],
        "spectral_emb_dim": [128],
        "gcn_dropout": [0.2],
        "gcn_activation": ["relu"],
        "gcn_use_batch_norm": [false],
        "graph_self_loop_bias": [2.0],
        "graph_identity_mix": [0.0],
        "graph_adjacency_reg_weight": [0.0001],

        "fusion_dim": [256],
        "latent_features": [128],
        "fusion_dropout": [0.2],
        "activation": ["relu"],

        "decoder_temporal_units": [128],
        "decoder_bilstm_layers": [1],
        "decoder_graph_output_units": [32],
        "decoder_branch_feature_dim": [64],
        "decoder_fusion_units": [128],
        "decoder_dropout": [0.2],
        "reconstruction_loss": ["mse"]
    }'
