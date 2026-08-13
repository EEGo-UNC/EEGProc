#!/bin/bash
#SBATCH --job-name=sic_v8_abl_val
#SBATCH --output=sic_v8_abl_val_%A_%a.out
#SBATCH --error=sic_v8_abl_val_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-4%2

set -euo pipefail

# ---------------------------------------------------------------------------
# Full experiment organization
# ---------------------------------------------------------------------------
# One Slurm array task runs one one-factor-at-a-time ablation. The %2 limit
# allows two profiles (four GPUs total) to run concurrently by default.
ABLATION_PROFILES=(
    full
    no_gcn_gru
    no_bilstm
    no_decoder
    remove_median
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
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-30}"
MLDG_STEPS_PER_EPOCH="${MLDG_STEPS_PER_EPOCH:-10}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"
TARGET_DIMENSION="${TARGET_DIMENSION:-valence}"
TRAINING_METHOD="${TRAINING_METHOD:-mldg}"
VREX_PENALTY_WEIGHT="${VREX_PENALTY_WEIGHT:-1.0}"
SUITE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"

# One worker supports valence/arousal and ERM/V-REx/MLDG without hand-editing.
# MLDG remains the default. ERM ignores method-specific settings; V-REx uses
# VREX_PENALTY_WEIGHT to scale the variance of subject-specific focal risks.
case "$TARGET_DIMENSION" in
    valence|arousal) ;;
    *) echo "ERROR: TARGET_DIMENSION must be valence or arousal."; exit 2 ;;
esac
case "$TRAINING_METHOD" in
    erm|vrex|mldg) ;;
    *) echo "ERROR: TRAINING_METHOD must be erm, vrex, or mldg."; exit 2 ;;
esac

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
# The full run searches only BiLSTM width (42/63/96 per direction). One layer,
# dropout 0.30, and z_dim 64 per branch stay fixed. This is a controlled
# three-configuration search, not a combinatorial depth/dropout/z search.
MODEL_GRID="$(python - \
    "$use_gcn_gru" \
    "$use_bilstm" \
    "$use_decoder" \
    "$remove_median" \
    "$MLDG_STEPS_PER_EPOCH" \
    "$search_bilstm_width" \
    "$TRAINING_METHOD" \
    "$VREX_PENALTY_WEIGHT" <<'PY'
import json
import sys

use_gcn_gru, use_bilstm, use_decoder, remove_median = (
    value.lower() == "true" for value in sys.argv[1:5]
)
mldg_steps_per_epoch = int(sys.argv[5])
search_bilstm_width = sys.argv[6].lower() == "true"
training_method = sys.argv[7]
vrex_penalty_weight = float(sys.argv[8])

bilstm_units = {"grid": [42, 63, 96]} if search_bilstm_width else 63

print(json.dumps({
    # ERM pools source risk; V-REx also penalizes risk variance across users;
    # MLDG performs subject-disjoint meta-train/meta-test episodes.
    "training_method": training_method,
    "vrex_penalty_weight": vrex_penalty_weight,

    # AdamW step size and decoupled L2-style parameter shrinkage.
    "optimizer_name": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 5e-5,

    # First-order MLDG. The meta-train loss includes the full VC, VAE, and
    # subject-adversarial objectives; meta-test measures focal generalization.
    "mldg_meta_train_subjects": 18,   # A: subjects used for the inner loss.
    "mldg_meta_test_subjects": 4,     # B: rotating virtual-unseen subjects.
    "mldg_trials_per_subject": 1,     # Complete trials sampled per subject.
    "mldg_steps_per_epoch": mldg_steps_per_epoch, # Episodes per epoch.
    "mldg_inner_learning_rate": 1e-4, # Temporary A-step learning rate.
    "mldg_meta_test_weight": 1.0,      # Weight of B's outer gradient.
    "mldg_seed": 42,

    # GCN-GRU spatial/spectral branch.
    "t_down": 2,                       # Halve encoder time resolution.
    "temporal_pool_sizes": {"fixed": [2]}, # GCN pooling schedule.
    "gcn_units": {"fixed": [128, 64]},     # Graph layer widths.
    "gcn_dropout": 0.20,
    "gcn_activation": "relu",
    "gcn_use_batch_norm": False,
    "spectral_gru_units": 384,         # Spectral embedding width.
    "spectral_gru_dropout": 0.20,
    "mi_n_neighbors": 3,               # k for source-only kNN MI graph.
    "mi_random_state": 42,
    "mi_zero_diagonal": False,
    "mi_band_reduction": "mean",      # Average channel MI across bands.
    "mi_max_observations": 15000,      # Bound graph-estimation cost.

    # Independent temporal branch. Width is searched per direction; therefore
    # candidates 42/63/96 produce total outputs 84/126/192 before the posterior.
    "bilstm_units": bilstm_units,       # Units per forward/backward direction.
    "n_bilstm_layers": 1,              # One temporal recurrent block.
    "bilstm_dropout": 0.30,            # Applied after LayerNorm.

    # Independent branch posteriors and joint reconstruction. z_dim is per
    # active branch: 64 + 64 = 128 for the full encoder.
    "z_dim": 64,                       # Latent width PER active branch.
    "z_log_var_clip_min": -20.0,
    "z_log_var_clip_max": 20.0,
    "vae_loss_weight": 0.10,           # Scale reconstruction + branch KL.
    "vae_beta": 0.05,                  # KL weight inside the VAE term.
    "decoder_dropout": 0.10,

    # Emotion classifier and VC regularizers.
    "focal_gamma": 1.5,                # Emphasize difficult examples.
    "focal_alpha": {"fixed": None},
    "classification_hidden_units": {"fixed": [128, 64]}, # Head widths.
    "classification_dropout": 0.20,
    "vc_loss_weight": 1.0,              # Overall focal + VC term weight.
    "vc_alpha": 1.0,                    # Focal scale inside VC objective.
    "vc_beta": 0.5,                     # VC posterior-KL coefficient.
    "vc_gamma": 0.0,                    # VC discriminator-KL disabled.
    "vc_lambda": 0.20,                  # Class-prior regularization.
    "update_vc_discriminator": False,

    # Subject-invariance objective on concatenated posterior means.
    "use_subject_adversarial": True,
    "subject_adversarial_weight": 0.60, # Gradient-reversal strength.
    "subject_loss_weight": 1.0,         # Scale of adversarial loss.
    "subject_hidden_units": 64,         # Subject discriminator width.
    "subject_dropout": 0.0,

    # One-factor-at-a-time architecture/data ablation.
    "use_gcn_gru_branch": use_gcn_gru,
    "use_bilstm_branch": use_bilstm,
    "use_decoder": use_decoder,
    "remove_median_label": remove_median,

    # Subject calibration updates only the selected classifier suffix.
    "calibration_unfreeze_layers": 2,   # Last hidden block + logits only.
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
echo "DREAMER target: $TARGET_DIMENSION"
echo "Training method: $TRAINING_METHOD"
echo "Branches: GCN-GRU=$use_gcn_gru BiLSTM=$use_bilstm decoder=$use_decoder"
echo "Remove median trials: $remove_median"
echo "Per-branch z_dim: 64"
if [[ "$search_bilstm_width" == "true" ]]; then
    echo "BiLSTM grid: 42, 63, 96 units/direction; one layer; dropout 0.30; LayerNorm enabled"
else
    echo "BiLSTM branch disabled; no duplicate BiLSTM-width grid"
fi
if [[ "$TRAINING_METHOD" == "mldg" ]]; then
    echo "MLDG: A=18 B=4 steps/epoch=$MLDG_STEPS_PER_EPOCH"
elif [[ "$TRAINING_METHOD" == "vrex" ]]; then
    echo "V-REx penalty weight: $VREX_PENALTY_WEIGHT"
fi
echo "Selection: minimize 12-shot calibrated Brier score"
echo "Calibration: 3-shot/6-fold, 6-shot/3-fold, 9-shot/2-fold, 12-shot/3-fold"
echo "Scope: all LOSO targets, $SOURCE_EPOCHS source epochs, $CALIBRATION_EPOCHS calibration epochs"
python --version
nvidia-smi

# ---------------------------------------------------------------------------
# Run the selected full ablation profile
# ---------------------------------------------------------------------------
# source-epochs is the fixed population-training budget; validation subjects and
# early stopping are disabled. source-batch-size applies to ERM/V-REx, whereas
# MLDG step size comes from complete-trial episodes. Calibration epochs/LR apply
# only to the unfrozen classifier suffix. Twenty posterior samples are averaged
# for probability reporting. n-jobs=2 maps concurrent LOSO folds to GPUs 0/1.
python -m src.eegproc.deep_learning.joint_architectures.sic.sic_model_train \
    --training-protocol loso_validation \
    --raw-eeg-npy datasets/remove_gamma/dreamer_eeg.npy \
    --raw-labels-npy datasets/remove_gamma/dreamer_labels.npy \
    --label-dimension "$TARGET_DIMENSION" \
    --classification-level window \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir "runs/full/sic_v8_${TRAINING_METHOD}_brier_ablation/DREAMER/${TARGET_DIMENSION}/suite_${SUITE_ID}/${ABLATION_PROFILE}" \
    --run-name "dreamer_${TARGET_DIMENSION}_sic_v8_${TRAINING_METHOD}_${ABLATION_PROFILE}_full" \
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
