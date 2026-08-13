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
TARGET_DIMENSION="${TARGET_DIMENSION:-valence}"
TRAINING_METHOD="${TRAINING_METHOD:-mldg}"
VREX_PENALTY_WEIGHT="${VREX_PENALTY_WEIGHT:-1.0}"

# The same worker supports both DREAMER targets and all source optimizers.
# MLDG is the default used by this smoke suite. ERM ignores the MLDG/V-REx
# settings; V-REx uses VREX_PENALTY_WEIGHT to scale variance across subject risk.
case "$TARGET_DIMENSION" in
    valence|arousal) ;;
    *) echo "ERROR: TARGET_DIMENSION must be valence or arousal."; exit 2 ;;
esac
case "$TRAINING_METHOD" in
    erm|vrex|mldg) ;;
    *) echo "ERROR: TRAINING_METHOD must be erm, vrex, or mldg."; exit 2 ;;
esac

# Group every task from the same array under one suite directory.
SUITE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"

# Brier score is minimized. The smoke test evaluates only 3- and 6-shot
# calibration, and ranks its tiny grid at 6 shots.
SELECTION_METRIC="brier_score"
HYPERPARAMETER_SELECTION_LEVEL="calibration"
CALIBRATION_SELECTION_SHOTS=9
CALIBRATION_LEVEL_ARGS=(
    --calibration-level 6 3
    --calibration-level 9 2
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
    "$MLDG_STEPS_PER_EPOCH" \
    "$TRAINING_METHOD" \
    "$VREX_PENALTY_WEIGHT" <<'PY'
import json
import sys

use_gcn_gru, use_bilstm, use_decoder, remove_median = (
    value.lower() == "true" for value in sys.argv[1:5]
)
mldg_steps_per_epoch = int(sys.argv[5])
training_method = sys.argv[6]
vrex_penalty_weight = float(sys.argv[7])

print(json.dumps({
    # Source optimizer: ERM uses the pooled source loss; V-REx adds variance
    # across subject-specific focal risks; MLDG simulates unseen subjects.
    "training_method": training_method,
    "vrex_penalty_weight": vrex_penalty_weight,

    # AdamW step size and decoupled L2-style parameter shrinkage.
    "optimizer_name": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 5e-5,

    # First-order MLDG: full VC/VAE/adversarial loss on meta-train subjects;
    # focal emotion loss on the temporarily adapted meta-test subjects.
    "mldg_meta_train_subjects": 18,   # A: subjects used for the inner loss.
    "mldg_meta_test_subjects": 4,     # B: rotating virtual-unseen subjects.
    "mldg_trials_per_subject": 1,     # Complete trials sampled per subject.
    "mldg_steps_per_epoch": mldg_steps_per_epoch,  # Episodes per epoch.
    "mldg_inner_learning_rate": 1e-4, # Temporary A-step learning rate.
    "mldg_meta_test_weight": 1.0,      # Weight of B's outer gradient.
    "mldg_seed": 42,                   # Reproducible episode sampling.

    # GCN-GRU spatial/spectral branch.
    "t_down": 2,                       # Halve the encoder time resolution.
    "temporal_pool_sizes": {"fixed": [2]}, # GCN temporal pool schedule.
    "gcn_units": {"fixed": [128, 64]},     # Two graph-convolution widths.
    "gcn_dropout": 0.20,               # Regularization inside the graph path.
    "gcn_activation": "relu",
    "gcn_use_batch_norm": False,       # Avoid batch-dependent subject effects.
    "spectral_gru_units": 384,         # Width after integrating EEG bands.
    "spectral_gru_dropout": 0.20,
    "mi_n_neighbors": 3,               # k for source-only kNN MI adjacency.
    "mi_random_state": 42,
    "mi_zero_diagonal": False,         # Retain self-information before loops.
    "mi_band_reduction": "mean",      # Average MI estimates across bands.
    "mi_max_observations": 15000,      # Cap MI computation cost.

    # Independent temporal branch. Units are per direction, so 63 + 63 gives
    # a 126-wide deterministic BiLSTM representation before its posterior.
    "bilstm_units": 63,                # Per direction: 126 total features.
    "n_bilstm_layers": 1,              # One temporal recurrent block.
    "bilstm_dropout": 0.30,            # Applied after LayerNorm.

    # Each active branch maps to its own 64-D variational posterior. With both
    # branches enabled, the classifier/decoder/adversary receive 128 features.
    "z_dim": 64,                       # Latent width PER active branch.
    "z_log_var_clip_min": -20.0,
    "z_log_var_clip_max": 20.0,
    "vae_loss_weight": 0.10,           # Weight of reconstruction + KL.
    "vae_beta": 0.05,                  # KL strength inside the VAE loss.
    "decoder_dropout": 0.10,           # Joint decoder regularization.

    # Emotion classifier and VC regularizers.
    "focal_gamma": 1.5,                # Focus learning on difficult labels.
    "focal_alpha": {"fixed": None},
    "classification_hidden_units": {"fixed": [128, 64]}, # Head widths.
    "classification_dropout": 0.20,
    "vc_loss_weight": 1.0,              # Weight of focal + VC regularizers.
    "vc_alpha": 1.0,                    # Focal term scale inside VC loss.
    "vc_beta": 0.5,                     # VC posterior-KL coefficient.
    "vc_gamma": 0.0,                    # VC discriminator-KL disabled.
    "vc_lambda": 0.20,                  # Class-prior regularization strength.
    "update_vc_discriminator": False,

    # Subject-invariance objective on the concatenated posterior means.
    "use_subject_adversarial": True,
    "subject_adversarial_weight": 0.60, # Gradient-reversal strength.
    "subject_loss_weight": 1.0,         # Scale of subject-adversarial loss.
    "subject_hidden_units": 64,         # Subject discriminator width.
    "subject_dropout": 0.0,

    # Selected architecture/data ablation.
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
echo "Smoke profile: $ABLATION_PROFILE"
echo "DREAMER target: $TARGET_DIMENSION"
echo "Training method: $TRAINING_METHOD"
echo "Branches: GCN-GRU=$use_gcn_gru BiLSTM=$use_bilstm decoder=$use_decoder"
echo "Per-branch z_dim: 64"
echo "BiLSTM: 63 units/direction, one layer, dropout 0.30, LayerNorm enabled"
if [[ "$TRAINING_METHOD" == "mldg" ]]; then
    echo "MLDG: A=18 B=4 steps/epoch=$MLDG_STEPS_PER_EPOCH"
elif [[ "$TRAINING_METHOD" == "vrex" ]]; then
    echo "V-REx penalty weight: $VREX_PENALTY_WEIGHT"
fi
echo "Selection: minimize 6-shot calibrated Brier score"
echo "Scope: $MAX_SUBJECTS targets, $SOURCE_EPOCHS source epochs, $CALIBRATION_EPOCHS calibration epochs"
python --version
nvidia-smi

# ---------------------------------------------------------------------------
# Run the selected smoke profile
# ---------------------------------------------------------------------------
# Source epochs are a fixed source-training budget because validation subjects
# and early stopping are disabled. source-batch-size controls ERM/V-REx; MLDG
# instead builds complete-trial episodes from its mldg_* settings. Calibration
# epochs/LR control classifier-only target adaptation. prediction-latent-samples
# averages five posterior draws for probability reporting. n-jobs=2 assigns one
# LOSO worker to each requested GPU.
python -m src.eegproc.deep_learning.joint_architectures.sic.sic_model_train \
    --training-protocol loso_validation \
    --raw-eeg-npy datasets/remove_gamma/dreamer_eeg.npy \
    --raw-labels-npy datasets/remove_gamma/dreamer_labels.npy \
    --label-dimension "$TARGET_DIMENSION" \
    --classification-level window \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir "runs/smoke/sic_v8_${TRAINING_METHOD}_brier/DREAMER/${TARGET_DIMENSION}/suite_${SUITE_ID}/${ABLATION_PROFILE}" \
    --run-name "dreamer_${TARGET_DIMENSION}_sic_v8_${TRAINING_METHOD}_${ABLATION_PROFILE}_smoke" \
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
    --window-sec 4.0 \
    --window-overlap 0.0 \
    --window-normalization global_rms \
    --hyperparameters-json "$MODEL_CONFIG"
