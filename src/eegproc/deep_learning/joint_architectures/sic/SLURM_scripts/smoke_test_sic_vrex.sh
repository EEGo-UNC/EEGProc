#!/bin/bash
#SBATCH --job-name=smoke_sic_vrex_grid_val
#SBATCH --output=smoke_sic_vrex_grid_val_%j.out
#SBATCH --error=smoke_sic_vrex_grid_val_%j.err
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

# The winning grid configuration and the best epoch within each fold are both
# selected using this accuracy type. Valid values: accuracy, balanced_accuracy.
SELECTION_METRIC="balanced_accuracy"
BEST_EPOCH_METRIC="balanced_accuracy"

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

# Scalar candidate lists are grid dimensions. Sequence-valued model settings
# such as gcn_units remain fixed unless given an extra nesting level.
MODEL_GRID="$(python - <<'PY'
import json

print(json.dumps({
    "optimizer_name": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 5e-5,
    "t_down": 2,
    "temporal_pool_sizes": [2],
    "gcn_units": [32],
    "gcn_dropout": 0.20,
    "gcn_activation": "relu",
    "gcn_use_batch_norm": False,
    "spectral_gru_units": 384,
    "spectral_gru_dropout": 0.2,
    "mi_n_neighbors": 3,
    "mi_random_state": 42,
    "mi_zero_diagonal": False,
    "mi_band_reduction": "mean",
    "mi_max_observations": 15000,
    "bilstm_units": 128,
    "n_bilstm_layers": 1,
    "bilstm_dropout": 0.3,
    "architecture_mode": "feature_fusion",
    "fusion_units": 128,
    "fusion_dropout": 0.2,

    # Two configurations: gamma=0 is the CE-equivalent baseline; gamma=1
    # applies focal modulation.
    "focal_gamma": [1.0],
    "focal_alpha": None,

    "z_dim": 128,
    "classification_hidden_units": [128],
    "classification_dropout": 0.20,
    "vc_loss_weight": 1.0,
    "vc_alpha": 1.0,
    "vc_beta": 2.0,
    "vc_gamma": 0.0,
    "vc_lambda": 0.0,
    "update_vc_discriminator": False,
    "vae_loss_weight": 0.10,
    "vae_beta": 0.05,
    "decoder_dropout": 0.10,
    "use_vrex": True,
    "vrex_penalty_weight": 10.0,
    "use_subject_adversarial": True,
    "subject_adversarial_weight": 0.6,
    "subject_loss_weight": 1.0,
    "subject_hidden_units": 64,
    "subject_dropout": 0.0,
    "calibration_unfreeze_layers": 2,
    "calibration_use_vc_target": True,
    "use_class_weight": False,
}))
PY
)"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Grid-selection metric: ${SELECTION_METRIC}"
echo "Best-epoch metric: ${BEST_EPOCH_METRIC}"
python --version
nvidia-smi

# Verify that the JSON contains the intended two-configuration grid.
python - "$MODEL_GRID" <<'PY'
import json
import sys

grid = json.loads(sys.argv[1])
focal_gammas = grid.get("focal_gamma")
if focal_gammas != [0.0, 1.0]:
    raise RuntimeError(
        "Expected focal_gamma grid [0.0, 1.0]; "
        f"received {focal_gammas!r}."
    )
print(f"Grid preflight: PASS ({len(focal_gammas)} configurations)")
PY

# Verify the SIC builder, feature-fusion window mode, and preservation of
# source-subject IDs required by V-REx.

python -m src.eegproc.deep_learning.joint_architectures.sic.sic_model_train \
    --training-protocol loso_validation \
    --raw-eeg-npy datasets/remove_gamma/dreamer_eeg.npy \
    --raw-labels-npy datasets/remove_gamma/dreamer_labels.npy \
    --label-dimension valence \
    --classification-level window \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir runs/smoke/sic_vrex_grid_validation/DREAMER/valence \
    --run-name dreamer_valence_sic_vrex_grid_validation_smoke \
    --source-epochs 100 \
    --source-batch-size 256 \
    --validation-subjects 4 \
    --validation-seed 42 \
    --early-stopping-patience 20 \
    --early-stopping-min-delta 0.001 \
    --selection-metric "$SELECTION_METRIC" \
    --best-epoch-metric "$BEST_EPOCH_METRIC" \
    --decision-threshold 0.5 \
    --prediction-latent-samples 15 \
    --latent-sampling-seed 42 \
    --max-subjects 4 \
    --n-jobs 2 \
    --gpu-ids 0 1 \
    --cpus-per-worker 2 \
    --verbose 2 \
    --seed 42 \
    --label-threshold-mode global \
    --median-label 3 \
    --window-sec 1.0 \
    --window-overlap 0.0 \
    --window-normalization global_rms \
    --hyperparameters-json "$MODEL_GRID"
