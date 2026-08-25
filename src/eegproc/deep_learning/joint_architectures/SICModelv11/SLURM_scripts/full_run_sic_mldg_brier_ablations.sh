#!/bin/bash
#SBATCH --job-name=sic_v11_abl
#SBATCH --output=sic_v11_abl_%A_%a.out
#SBATCH --error=sic_v11_abl_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=47:00:00
#SBATCH --array=0

set -euo pipefail

# ---------------------------------------------------------------------------
# Full experiment organization
# ---------------------------------------------------------------------------
# This worker targets SIC builder API v11: deterministic window encoders plus
# a trial-level BiGRU classifier with no averaging across EEG windows.
# One Slurm array task runs one one-factor-at-a-time ablation. The %2 limit
# allows two profiles (four GPUs total) to run concurrently by default.
ABLATION_PROFILES=(
    full
)

PROFILE_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( PROFILE_INDEX < 0 || PROFILE_INDEX >= ${#ABLATION_PROFILES[@]} )); then
    echo "ERROR: array task $PROFILE_INDEX is outside 0-$((${#ABLATION_PROFILES[@]} - 1))."
    exit 2
fi
ABLATION_PROFILE="${ABLATION_PROFILES[$PROFILE_INDEX]}"

# ---------------------------------------------------------------------------
# Cluster environment and user-overridable full-run budget
# ---------------------------------------------------------------------------
module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="${PROJECT_DIR:-$HOME/EEGProc}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-40}"
MLDG_STEPS_PER_EPOCH="${MLDG_STEPS_PER_EPOCH:-25}"
SOURCE_BATCH_SIZE="${SOURCE_BATCH_SIZE:-32}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-16}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"
TARGET_DIMENSION="${TARGET_DIMENSION:-valence}"
TRAINING_METHOD="${TRAINING_METHOD:-mldg}"
VREX_PENALTY_WEIGHT="${VREX_PENALTY_WEIGHT:-1.0}"
PREDICTION_DIAGNOSTICS_METRIC="${PREDICTION_DIAGNOSTICS_METRIC:-brier_score}"
PREDICTION_DIAGNOSTICS_EVERY_N_EPOCHS="${PREDICTION_DIAGNOSTICS_EVERY_N_EPOCHS:-10}"
PREDICTION_DIAGNOSTICS_MAX_SAMPLES="${PREDICTION_DIAGNOSTICS_MAX_SAMPLES:-10000}"
EXPECTED_SIC_API_VERSION=11
SUITE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"

EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"

if [[ "$TARGET_DIMENSION" != "valence" && "$TARGET_DIMENSION" != "arousal" ]]; then
    echo "ERROR: TARGET_DIMENSION must be valence or arousal."
    exit 2
fi
if [[ "$TRAINING_METHOD" != "erm" && "$TRAINING_METHOD" != "vrex" && "$TRAINING_METHOD" != "mldg" ]]; then
    echo "ERROR: TRAINING_METHOD must be erm, vrex, or mldg."
    exit 2
fi

# Brier score is minimized. Every shot level is reported, and 12-shot
# calibrated performance ranks hyperparameter configurations.
SELECTION_METRIC="brier_score"
HYPERPARAMETER_SELECTION_LEVEL="calibration"
CALIBRATION_SELECTION_SHOTS=12
CALIBRATION_LEVEL_ARGS=(
    --calibration-level 3 6
    --calibration-level 6 3
    --calibration-level 9 2
    --calibration-level 12 3
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
# Resolve one array profile into explicit branch/data switches
# ---------------------------------------------------------------------------
use_gcn_gru=true
use_bilstm=true
remove_median=false

case "$ABLATION_PROFILE" in
    full)
        ;;
    no_gcn_gru)
        use_gcn_gru=false
        ;;
    no_bilstm)
        use_bilstm=false
        ;;
    remove_median)
        remove_median=true
        ;;
    *)
        echo "ERROR: unknown ablation profile: $ABLATION_PROFILE"
        exit 2
        ;;
esac

# Search BiLSTM width only when that branch is active. This avoids running
# three duplicate configurations in the no-BiLSTM task.
search_bilstm_width="$use_bilstm"

# ---------------------------------------------------------------------------
# Build the profile-specific Cartesian grid as valid JSON
# ---------------------------------------------------------------------------
# The full run searches BiLSTM width (42/63/96 per direction), focal gamma,
# and VC beta. GCN-GRU keeps all 384 output features and the BiLSTM keeps
# 2*bilstm_units features. Their per-window concatenations form the ordered
# sequence consumed by the trial BiGRU. Focal gamma and VC beta are searched.
MODEL_GRID="$(python - \
    "$use_gcn_gru" \
    "$use_bilstm" \
    "$remove_median" \
    "$MLDG_STEPS_PER_EPOCH" \
    "$search_bilstm_width" \
    "$VREX_PENALTY_WEIGHT" <<'PY'
import json
import sys

use_gcn_gru, use_bilstm, remove_median = (
    value.lower() == "true" for value in sys.argv[1:4]
)
mldg_steps_per_epoch = int(sys.argv[4])
search_bilstm_width = sys.argv[5].lower() == "true"
vrex_penalty_weight = float(sys.argv[6])

print(json.dumps({
    # Source optimizer. The method itself is selected by --training-method.
    "optimizer_name": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 5e-5,
    "vrex_penalty_weight": vrex_penalty_weight,

    # First-order MLDG. Meta-train uses the VC/classification and subject-
    # adversarial objectives; meta-test measures focal generalization.
    "mldg_meta_train_subjects": 10,
    "mldg_meta_test_subjects": 5,
    "mldg_trials_per_subject": 2,
    "mldg_steps_per_epoch": mldg_steps_per_epoch,
    "mldg_inner_learning_rate": 1e-4,
    "mldg_meta_test_weight": 1.0,
    "mldg_seed": 42,

    # GCN-GRU spatial/spectral branch.
    "t_down": 2,
    "temporal_pool_sizes": {"fixed": [2]},
    "gcn_units": {"fixed": [128, 64]},
    "gcn_dropout": 0.3,
    "gcn_activation": "relu",
    "gcn_use_batch_norm": False,
    "spectral_gru_units": 384,
    "spectral_gru_dropout": 0.3,
    "mi_n_neighbors": 3,
    "mi_random_state": 42,
    "mi_zero_diagonal": False,
    "mi_band_reduction": "mean",
    "mi_max_observations": 15000,

    # Independent temporal branch. Width is searched per direction; therefore
    # candidates 42/63/96 produce complete outputs of 84/126/192 features.
    "bilstm_units": 63,
    "n_bilstm_layers": 1,
    "bilstm_dropout": 0.3,

    # Trial classifier. Every ordered one-second window embedding is processed
    # by the BiGRU; only its final bidirectional state enters the VC head.
    # There is no arithmetic mean across the trial's window axis.
    "classifier_rnn_type": "bigru",
    "classifier_rnn_units": {"fixed": [128, 64]},
    "n_classifier_rnn_layers": 2,
    "classifier_rnn_dropout": 0.4,

    # VariationalClassifier-head regularizers. The VC is the sole logits head.
    "focal_gamma": 1.5,
    "focal_alpha": None,
    "vc_loss_weight": 1.0,
    "vc_alpha": 1.0,
    "vc_beta": 1.0,
    "vc_gamma": 0.0,
    "vc_lambda": 0.05,
    "update_vc_discriminator": False,

    # Subject-invariance objective on the learned trial representation.
    "use_subject_adversarial": False,
    "subject_adversarial_weight": 0.0,
    "subject_loss_weight": 0.0,
    "subject_hidden_units": 2,
    "subject_dropout": 0.0,

    # One-factor-at-a-time architecture/data ablation.
    "use_gcn_gru_branch": use_gcn_gru,
    "use_bilstm_branch": use_bilstm,
    "remove_median_label": remove_median,

    # Two layers means BiGRU + VC during target-subject calibration.
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
echo "Ablation profile: $ABLATION_PROFILE"
echo "Target: $TARGET_DIMENSION"
echo "Training method: $TRAINING_METHOD"
echo "Branches: GCN-GRU=$use_gcn_gru BiLSTM=$use_bilstm"
echo "Remove median trials: $remove_median"
echo "Trial classifier: BiGRU [128,64] units/direction, final width 128, no cross-window averaging"
echo "Prediction diagnostics: metric=$PREDICTION_DIAGNOSTICS_METRIC every=$PREDICTION_DIAGNOSTICS_EVERY_N_EPOCHS epoch(s) max_samples=$PREDICTION_DIAGNOSTICS_MAX_SAMPLES"
echo "Selection: minimize 12-shot calibrated Brier score"
echo "Calibration: 3-shot/6-fold, 6-shot/3-fold, 9-shot/2-fold, 12-shot/3-fold"
echo "Scope: all LOSO targets, $SOURCE_EPOCHS source epochs, $CALIBRATION_EPOCHS calibration epochs"
python --version
nvidia-smi

# ---------------------------------------------------------------------------
# Run the selected full ablation profile
# ---------------------------------------------------------------------------
python -m src.eegproc.deep_learning.joint_architectures.SICModelv11.sic_model_train \
    --training-protocol loso_validation \
    --dataset dreamer \
    --raw-eeg-npy "$EEG_PATH" \
    --raw-labels-npy "$LABELS_PATH" \
    --label-dimension "$TARGET_DIMENSION" \
    --classification-level trial \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir "runs/full/sic_trial_bigru_v11_${TRAINING_METHOD}_brier_ablation/DREAMER/${TARGET_DIMENSION}/suite_${SUITE_ID}/${ABLATION_PROFILE}" \
    --run-name "dreamer_${TARGET_DIMENSION}_sic_trial_bigru_v11_${TRAINING_METHOD}_${ABLATION_PROFILE}_full" \
    --training-method "$TRAINING_METHOD" \
    --source-epochs "$SOURCE_EPOCHS" \
    --source-batch-size "$SOURCE_BATCH_SIZE" \
    --validation-subjects 0 \
    --no-early-stopping \
    --calibration-epochs "$CALIBRATION_EPOCHS" \
    --calibration-batch-size "$CALIBRATION_BATCH_SIZE" \
    "${CALIBRATION_LEVEL_ARGS[@]}" \
    --calibration-selection-shots "$CALIBRATION_SELECTION_SHOTS" \
    --calibration-learning-rate 0.0001 \
    --calibration-optimizer adamw \
    --calibration-weight-decay 0.00005 \
    --calibration-seed 42 \
    --selection-metric "$SELECTION_METRIC" \
    --hyperparameter-selection-level "$HYPERPARAMETER_SELECTION_LEVEL" \
    --decision-threshold 0.5 \
    --prediction-diagnostics \
    --prediction-diagnostics-metric "$PREDICTION_DIAGNOSTICS_METRIC" \
    --prediction-diagnostics-every-n-epochs "$PREDICTION_DIAGNOSTICS_EVERY_N_EPOCHS" \
    --prediction-diagnostics-max-samples "$PREDICTION_DIAGNOSTICS_MAX_SAMPLES" \
    --prediction-diagnostics-threshold-tolerance 0.01 \
    --prediction-diagnostics-seed 42 \
    --ece-bins 15 \
    --n-jobs 4 \
    --gpu-ids 0 1 2 3 \
    --cpus-per-worker 2 \
    --verbose 2 \
    --seed 42 \
    --label-threshold-mode global \
    --median-label 3 \
    --window-sec 1.0 \
    --window-overlap 0.0 \
    --window-normalization global_rms \
    --hyperparameters-json "$MODEL_GRID"
