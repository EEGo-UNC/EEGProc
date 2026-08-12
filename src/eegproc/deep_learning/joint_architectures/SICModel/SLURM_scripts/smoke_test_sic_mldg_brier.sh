#!/bin/bash
#SBATCH --job-name=smoke_sic_v8
#SBATCH --output=smoke_sic_v8_%A_%a.out
#SBATCH --error=smoke_sic_v8_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --array=0-1%2

set -euo pipefail

# ---------------------------------------------------------------------------
# Smoke-test purpose
# ---------------------------------------------------------------------------
# Each array task validates one latent layout:
#   task 0: full encoder,       z = concat(z_gcn[64], z_bilstm[64]) -> 128
#   task 1: no BiLSTM branch,   z = z_gcn[64]                       -> 64
# This catches joint/single-branch decoder, classifier, adversary, calibration,
# and MLDG shape errors without running every full ablation.

SMOKE_PROFILES=(
    full
    no_bilstm
)

PROFILE_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( PROFILE_INDEX < 0 || PROFILE_INDEX >= ${#SMOKE_PROFILES[@]} )); then
    echo "ERROR: array task $PROFILE_INDEX is outside the smoke profile range."
    exit 2
fi
ABLATION_PROFILE="${SMOKE_PROFILES[$PROFILE_INDEX]}"

# ---------------------------------------------------------------------------
# Cluster environment and user-overridable smoke budget
# ---------------------------------------------------------------------------
module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="${PROJECT_DIR:-$HOME/EEGProc}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-3}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-3}"
MAX_SUBJECTS="${MAX_SUBJECTS:-2}"
MLDG_STEPS_PER_EPOCH="${MLDG_STEPS_PER_EPOCH:-2}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"

# Group every task from the same array under one suite directory.
SUITE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"

# Brier score is minimized. The smoke test evaluates only 3- and 6-shot
# calibration, and ranks its tiny grid at 6 shots.
SELECTION_METRIC="brier_score"
HYPERPARAMETER_SELECTION_LEVEL="calibration"
CALIBRATION_SELECTION_SHOTS=6
CALIBRATION_LEVEL_ARGS=(
    --calibration-level 3 2
    --calibration-level 6 2
)

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Create/reuse the shared virtual environment safely across array tasks
# ---------------------------------------------------------------------------
if command -v flock >/dev/null 2>&1; then
    (
        flock -x 9
        if [[ ! -x "$VENV_DIR/bin/python" ]]; then
            python -m venv "$VENV_DIR"
            "$VENV_DIR/bin/python" -m pip install --upgrade pip
            "$VENV_DIR/bin/python" -m pip install -r requirements.txt
        elif [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
            "$VENV_DIR/bin/python" -m pip install -r requirements.txt
        fi
    ) 9>"$PROJECT_DIR/.venv312_install.lock"
else
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        python -m venv "$VENV_DIR"
        "$VENV_DIR/bin/python" -m pip install --upgrade pip
        "$VENV_DIR/bin/python" -m pip install -r requirements.txt
    elif [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
        "$VENV_DIR/bin/python" -m pip install -r requirements.txt
    fi
fi
source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Locate CUDA libdevice for TensorFlow/XLA
# ---------------------------------------------------------------------------
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
    echo "ERROR: unable to locate libdevice.10.bc."
    exit 1
fi

CUDA_XLA_ROOT="${LIBDEVICE_PATH%/nvvm/libdevice/libdevice.10.bc}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_XLA_ROOT}"
if [[ -n "$MODULE_CUDA_ROOT" ]]; then
    export CUDA_HOME="$MODULE_CUDA_ROOT"
    export CUDA_PATH="$MODULE_CUDA_ROOT"
fi

# ---------------------------------------------------------------------------
# Resolve the selected ablation into explicit branch switches
# ---------------------------------------------------------------------------
use_gcn_gru=true
use_bilstm=true
use_decoder=true
remove_median=false

case "$ABLATION_PROFILE" in
    full)
        ;;
    no_bilstm)
        use_bilstm=false
        ;;
    *)
        echo "ERROR: unknown smoke profile: $ABLATION_PROFILE"
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Build one fixed smoke configuration as valid JSON
# ---------------------------------------------------------------------------
# z_dim is PER ACTIVE BRANCH. The corrected encoder creates an independent
# Gaussian posterior in each branch, samples each z, and concatenates those z
# values directly. There is no architecture_mode or fusion projection.
MODEL_CONFIG="$(python - \
    "$use_gcn_gru" \
    "$use_bilstm" \
    "$use_decoder" \
    "$remove_median" \
    "$MLDG_STEPS_PER_EPOCH" <<'PY'
import json
import sys

use_gcn_gru, use_bilstm, use_decoder, remove_median = (
    value.lower() == "true" for value in sys.argv[1:5]
)
mldg_steps_per_epoch = int(sys.argv[5])

print(json.dumps({
    # Overall source-optimization method.
    "training_method": "mldg",
    "optimizer_name": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 5e-5,

    # First-order MLDG: full VC/VAE/adversarial loss on meta-train subjects;
    # focal emotion loss on the temporarily adapted meta-test subjects.
    "mldg_meta_train_subjects": 18,
    "mldg_meta_test_subjects": 4,
    "mldg_trials_per_subject": 1,
    "mldg_steps_per_epoch": mldg_steps_per_epoch,
    "mldg_inner_learning_rate": 1e-4,
    "mldg_meta_test_weight": 1.0,
    "mldg_seed": 42,

    # GCN-GRU spatial/spectral branch.
    "t_down": 2,
    "temporal_pool_sizes": {"fixed": [2]},
    "gcn_units": {"fixed": [128, 64]},
    "gcn_dropout": 0.20,
    "gcn_activation": "relu",
    "gcn_use_batch_norm": False,
    "spectral_gru_units": 384,
    "spectral_gru_dropout": 0.20,
    "mi_n_neighbors": 3,
    "mi_random_state": 42,
    "mi_zero_diagonal": False,
    "mi_band_reduction": "mean",
    "mi_max_observations": 15000,

    # Independent temporal branch. Units are per direction, so 63 + 63 gives
    # a 126-wide deterministic BiLSTM representation before its posterior.
    "bilstm_units": 63,
    "n_bilstm_layers": 1,
    "bilstm_dropout": 0.30,

    # Each active branch maps to its own 64-D variational posterior. With both
    # branches enabled, the classifier/decoder/adversary receive 128 features.
    "z_dim": 64,
    "z_log_var_clip_min": -20.0,
    "z_log_var_clip_max": 20.0,
    "vae_loss_weight": 0.10,
    "vae_beta": 0.05,
    "decoder_dropout": 0.10,

    # Emotion classifier and VC regularizers.
    "focal_gamma": 1.5,
    "focal_alpha": {"fixed": None},
    "classification_hidden_units": {"fixed": [128, 64]},
    "classification_dropout": 0.20,
    "vc_loss_weight": 1.0,
    "vc_alpha": 1.0,
    "vc_beta": {"grid": [0.5, 1.5, 2.5]},
    "vc_gamma": 0.0,
    "vc_lambda": 0.05,
    "update_vc_discriminator": False,

    # Subject-invariance objective on the concatenated posterior means.
    "use_subject_adversarial": True,
    "subject_adversarial_weight": 0.60,
    "subject_loss_weight": 1.0,
    "subject_hidden_units": 64,
    "subject_dropout": 0.0,

    # Selected architecture/data ablation.
    "use_gcn_gru_branch": use_gcn_gru,
    "use_bilstm_branch": use_bilstm,
    "use_decoder": use_decoder,
    "remove_median_label": remove_median,

    # Subject calibration updates only the selected classifier suffix.
    "calibration_unfreeze_layers": 2,
    "calibration_use_vc_target": True,
    "use_class_weight": False,
}))
PY
)"

# ---------------------------------------------------------------------------
# Record the exact allocation and experiment mapping
# ---------------------------------------------------------------------------
echo "Suite ID: $SUITE_ID"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node: $(hostname)"
echo "Smoke profile: $ABLATION_PROFILE"
echo "Branches: GCN-GRU=$use_gcn_gru BiLSTM=$use_bilstm decoder=$use_decoder"
echo "Per-branch z_dim: 64"
echo "BiLSTM: 63 units/direction, one layer, dropout 0.30, LayerNorm enabled"
echo "MLDG: A=18 B=4 steps/epoch=$MLDG_STEPS_PER_EPOCH"
echo "Selection: minimize 6-shot calibrated Brier score"
echo "Scope: $MAX_SUBJECTS targets, $SOURCE_EPOCHS source epochs, $CALIBRATION_EPOCHS calibration epochs"
python --version
nvidia-smi

# ---------------------------------------------------------------------------
# Run the selected smoke profile
# ---------------------------------------------------------------------------
python -m src.eegproc.deep_learning.joint_architectures.SICModel.sic_model_train \
    --training-protocol loso_validation \
    --raw-eeg-npy datasets/dreamer_eeg.npy \
    --raw-labels-npy datasets/dreamer_labels.npy \
    --label-dimension valence \
    --classification-level window \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir "runs/smoke/sic_v8_mldg_brier/suite_${SUITE_ID}/${ABLATION_PROFILE}" \
    --run-name "dreamer_valence_sic_v8_${ABLATION_PROFILE}_smoke" \
    --source-epochs "$SOURCE_EPOCHS" \
    --source-batch-size 512 \
    --validation-subjects 0 \
    --no-early-stopping \
    --calibration-epochs "$CALIBRATION_EPOCHS" \
    --calibration-batch-size 32 \
    "${CALIBRATION_LEVEL_ARGS[@]}" \
    --calibration-selection-shots "$CALIBRATION_SELECTION_SHOTS" \
    --calibration-learning-rate 0.0001 \
    --calibration-optimizer adamw \
    --calibration-weight-decay 0.00005 \
    --calibration-seed 42 \
    --selection-metric "$SELECTION_METRIC" \
    --hyperparameter-selection-level "$HYPERPARAMETER_SELECTION_LEVEL" \
    --decision-threshold 0.5 \
    --prediction-latent-samples 5 \
    --latent-sampling-seed 42 \
    --ece-bins 15 \
    --max-subjects "$MAX_SUBJECTS" \
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
    --hyperparameters-json "$MODEL_CONFIG"
