#!/bin/bash
#SBATCH --job-name=sic_mldg_abl_val
#SBATCH --output=sic_mldg_abl_val_%A_%a.out
#SBATCH --error=sic_mldg_abl_val_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-4

set -euo pipefail

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="${PROJECT_DIR:-$HOME/EEGProc}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"

# Full-run settings. Override these at submission time with --export.
SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-30}"
MLDG_STEPS_PER_EPOCH="${MLDG_STEPS_PER_EPOCH:-10}"

# Each array task runs one one-factor-at-a-time ablation. SupCon weight is
# searched independently inside every profile.
ABLATION_PROFILES=(
    full
    no_gcn_gru
    no_bilstm
    no_decoder
    remove_median
)

PROFILE_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( PROFILE_INDEX < 0 || PROFILE_INDEX >= ${#ABLATION_PROFILES[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$PROFILE_INDEX is outside 0-$((${#ABLATION_PROFILES[@]} - 1))."
    exit 2
fi
ABLATION_PROFILE="${ABLATION_PROFILES[$PROFILE_INDEX]}"

# Brier score is minimized. The 12-shot aggregate ranks hyperparameter
# configurations; zero-shot and every requested calibration level are reported.
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

# Array tasks may begin simultaneously, so venv creation and package updates
# are serialized.
if command -v flock >/dev/null 2>&1; then
    (
        flock -x 9
        if [[ ! -x "$VENV_DIR/bin/python" ]]; then
            python -m venv "$VENV_DIR"
        fi
        "$VENV_DIR/bin/python" -m pip install --upgrade pip
        "$VENV_DIR/bin/python" -m pip install -r requirements.txt
    ) 9>"$PROJECT_DIR/.venv312_install.lock"
else
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        python -m venv "$VENV_DIR"
    fi
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
    "$VENV_DIR/bin/python" -m pip install -r requirements.txt
fi
source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

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

make_model_grid() {
    local use_gcn_gru="$1"
    local use_bilstm="$2"
    local use_decoder="$3"
    local remove_median="$4"

    python - "$use_gcn_gru" "$use_bilstm" "$use_decoder" "$remove_median" "$MLDG_STEPS_PER_EPOCH" <<'PY'
import json
import sys

use_gcn_gru, use_bilstm, use_decoder, remove_median = (
    value.lower() == "true" for value in sys.argv[1:5]
)
mldg_steps_per_epoch = int(sys.argv[5])

print(json.dumps({
    "training_method": "mldg",
    "optimizer_name": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 5e-5,

    # First-order MLDG. With a strict DREAMER LOSO fold, all 22 source
    # subjects participate in each rotating 18-subject/4-subject episode.
    "mldg_meta_train_subjects": 18,
    "mldg_meta_test_subjects": 4,
    "mldg_trials_per_subject": 1,
    "mldg_steps_per_epoch": mldg_steps_per_epoch,
    "mldg_inner_learning_rate": 1e-4,
    "mldg_meta_test_weight": 1.0,
    "mldg_seed": 42,

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

    # Units are per direction: 63 forward + 63 backward = 126 features.
    "bilstm_units": 63,
    "n_bilstm_layers": 1,
    "bilstm_dropout": 0.30,
    "architecture_mode": "feature_fusion",
    "fusion_units": 128,
    "fusion_dropout": 0.20,

    "focal_gamma": 1.5,
    "focal_alpha": {"fixed": None},
    "z_dim": 128,
    "classification_hidden_units": {"fixed": [128, 64]},
    "classification_dropout": 0.20,
    "vc_loss_weight": 1.0,
    "vc_alpha": 1.0,
    "vc_beta": 0.5,
    "vc_gamma": 0.0,
    "vc_lambda": 0.2,
    "update_vc_discriminator": False,

    # A weight of 0.0 is the SupCon-loss ablation. Positive candidates test
    # how strongly SupCon should regularize the shared classifier embedding.
    "use_supcon": True,
    "supcon_loss_weight": {"grid": [0.0, 0.05, 0.10]},
    "supcon_temperature": 0.10,
    "supcon_cross_subject_only": True,

    "vae_loss_weight": 0.10,
    "vae_beta": 0.05,
    "decoder_dropout": 0.10,
    "use_subject_adversarial": True,
    "subject_adversarial_weight": 0.60,
    "subject_loss_weight": 1.0,
    "subject_hidden_units": 64,
    "subject_dropout": 0.0,

    "use_gcn_gru_branch": use_gcn_gru,
    "use_bilstm_branch": use_bilstm,
    "use_decoder": use_decoder,
    "remove_median_label": remove_median,

    "calibration_unfreeze_layers": 2,
    "calibration_use_vc_target": True,
    "use_class_weight": False,
}))
PY
}

use_gcn_gru=true
use_bilstm=true
use_decoder=true
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
    no_decoder)
        use_decoder=false
        ;;
    remove_median)
        remove_median=true
        ;;
    *)
        echo "ERROR: Unknown ablation profile: $ABLATION_PROFILE"
        exit 2
        ;;
esac

MODEL_GRID="$(make_model_grid "$use_gcn_gru" "$use_bilstm" "$use_decoder" "$remove_median")"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node: $(hostname)"
echo "Ablation profile: $ABLATION_PROFILE"
echo "GCN-GRU branch: $use_gcn_gru"
echo "BiLSTM branch: $use_bilstm"
echo "Decoder: $use_decoder"
echo "Remove median label: $remove_median"
echo "Source training: first-order MLDG (A=18, B=4, steps/epoch=$MLDG_STEPS_PER_EPOCH)"
echo "SupCon weights: 0.0, 0.05, 0.10"
echo "Selection: minimize 12-shot calibrated Brier score"
echo "Calibration: 3-shot/6-fold, 6-shot/3-fold, 9-shot/2-fold, 12-shot/3-fold"
echo "Scope: all LOSO target subjects, $SOURCE_EPOCHS source epochs, $CALIBRATION_EPOCHS calibration epochs"
python --version
nvidia-smi

python -m src.eegproc.deep_learning.joint_architectures.sic.sic_model_train \
    --training-protocol loso_validation \
    --raw-eeg-npy datasets/remove_gamma/dreamer_eeg.npy \
    --raw-labels-npy datasets/remove_gamma/dreamer_labels.npy \
    --label-dimension valence \
    --classification-level window \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir "runs/full/sic_mldg_supcon_brier_ablation/DREAMER/valence/$ABLATION_PROFILE" \
    --run-name "dreamer_valence_sic_mldg_supcon_brier_${ABLATION_PROFILE}_full" \
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
    --prediction-latent-samples 20 \
    --latent-sampling-seed 42 \
    --ece-bins 15 \
    --n-jobs 2 \
    --gpu-ids 0 1 \
    --cpus-per-worker 2 \
    --verbose 2 \
    --seed 42 \
    --label-threshold-mode global \
    --median-label 3 \
    --window-sec 4.0 \
    --window-overlap 0.0 \
    --window-normalization global_rms \
    --hyperparameters-json "$MODEL_GRID"
