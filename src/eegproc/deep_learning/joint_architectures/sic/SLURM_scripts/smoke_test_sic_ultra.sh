#!/bin/bash
#SBATCH --job-name=smoke_sic_mldg_supcon_brier
#SBATCH --output=smoke_sic_mldg_supcon_brier_%j.out
#SBATCH --error=smoke_sic_mldg_supcon_brier_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00

set -euo pipefail

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="$HOME/EEGProc"
VENV_DIR="$PROJECT_DIR/venv312"

# Smoke defaults can be overridden at submission time, for example:
#   SOURCE_EPOCHS=30 MAX_SUBJECTS=4 sbatch <this-script>
SOURCE_EPOCHS="${SOURCE_EPOCHS:-10}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-10}"
MAX_SUBJECTS="${MAX_SUBJECTS:-4}"
MLDG_STEPS_PER_EPOCH="${MLDG_STEPS_PER_EPOCH:-10}"

# Brier score is the primary report and hyperparameter-selection metric.
# It is minimized. The 12-shot result ranks configurations; all four shot
# levels are still evaluated and written to JSON/CSV for every configuration.
SELECTION_METRIC="brier_score"
HYPERPARAMETER_SELECTION_LEVEL="calibration"
CALIBRATION_SELECTION_SHOTS=9

# Full shot-level continuation requested for DREAMER. Each source model is
# trained once per target subject and restored independently for every fold.
CALIBRATION_LEVEL_ARGS=(
    --calibration-level 6 3
    --calibration-level 9 2
)

# One-factor-at-a-time ablations. These are separate searches because a plain
# Cartesian grid over the two encoder switches would also create the invalid
# configuration with both branches disabled.
# Override with, for example:
#   ABLATION_PROFILES="full no_decoder" sbatch <this-script>
# ABLATION_PROFILES="${ABLATION_PROFILES:-full no_gcn_gru no_bilstm no_decoder remove_median}"
ABLATION_PROFILES="${ABLATION_PROFILES:-full remove_median}"
read -r -a REQUESTED_ABLATIONS <<< "$ABLATION_PROFILES"

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

    # First-order MLDG: all 22 outer-source subjects participate in every
    # episode as 18 meta-train (A) and four rotating meta-test (B) subjects.
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

    # The current builder defines units per direction, so 63 + 63 gives a
    # 126-wide concatenated bidirectional output.
    "bilstm_units": 63,
    "n_bilstm_layers": 1,
    "bilstm_dropout": 0.30,
    "architecture_mode": "feature_fusion",
    "fusion_units": 128,
    "fusion_dropout": 0.20,

    "focal_gamma": {"grid": [0.5, 1.5]},
    "focal_alpha": {"fixed": None},
    "z_dim": 128,
    "classification_hidden_units": {"fixed": [128, 64]},
    "classification_dropout": 0.20,
    "vc_loss_weight": 1.0,
    "vc_alpha": 1.0,
    "vc_beta": 0.5,
    "vc_gamma": 0.0,
    "vc_lambda": 0.01,
    "update_vc_discriminator": False,

    # SupCon is applied to the existing classifier embedding. Weight 0.0 is
    # the loss-off ablation; positive candidates test the new objective.
    "use_supcon": True,
    "supcon_loss_weight": {"grid": [0.02, 0.2]},
    "supcon_temperature": {"grid": [0.05, 0.15]},,
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

run_ablation() {
    local profile="$1"
    local use_gcn_gru=true
    local use_bilstm=true
    local use_decoder=true
    local remove_median=false

    case "$profile" in
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
            echo "ERROR: Unknown ablation profile: $profile"
            echo "Valid profiles: full no_gcn_gru no_bilstm no_decoder remove_median"
            exit 2
            ;;
    esac

    local model_grid
    model_grid="$(make_model_grid "$use_gcn_gru" "$use_bilstm" "$use_decoder" "$remove_median")"

    echo
    echo "Ablation profile: $profile"
    echo "  GCN-GRU branch: $use_gcn_gru"
    echo "  BiLSTM branch: $use_bilstm"
    echo "  Decoder: $use_decoder"
    echo "  Remove median label: $remove_median"
    echo "  SupCon weights: 0.0, 0.05, 0.10"
    echo "  Primary/selection metric: $SELECTION_METRIC (minimize)"
    echo "  Selection source: ${CALIBRATION_SELECTION_SHOTS}-shot calibrated aggregate"

    python -m src.eegproc.deep_learning.joint_architectures.sic.sic_model_train \
        --training-protocol loso_validation \
        --raw-eeg-npy datasets/remove_gamma/dreamer_eeg.npy \
        --raw-labels-npy datasets/remove_gamma/dreamer_labels.npy \
        --label-dimension valence \
        --classification-level window \
        --n-channels 14 \
        --n-bands 3 \
        --out-dir "runs/smoke/sic_mldg_supcon_brier_ablation/DREAMER/valence/$profile" \
        --run-name "dreamer_valence_sic_mldg_supcon_brier_${profile}_smoke" \
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
        --hyperparameters-json "$model_grid"
}

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Optimization: first-order MLDG, A=18, B=4, steps/epoch=$MLDG_STEPS_PER_EPOCH"
echo "Primary report metric: Brier score"
echo "Hyperparameter selection: minimize 12-shot calibrated Brier score"
echo "Calibration plan: 3-shot/6-fold, 6-shot/3-fold, 9-shot/2-fold, 12-shot/3-fold"
echo "Ablations: ${REQUESTED_ABLATIONS[*]}"
echo "Smoke scope: $MAX_SUBJECTS LOSO targets, $SOURCE_EPOCHS source epochs, $CALIBRATION_EPOCHS calibration epochs"
python --version
nvidia-smi

for profile in "${REQUESTED_ABLATIONS[@]}"; do
    run_ablation "$profile"
done
