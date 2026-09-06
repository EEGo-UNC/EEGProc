#!/bin/bash
#SBATCH --job-name=sicv11_ar_best
#SBATCH --output=sicv11_ar_best_%j.out
#SBATCH --error=sicv11_ar_best_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=09:00:00

set -euo pipefail

# Full DREAMER arousal LOSO run using the rank-1 hyperparameters selected by
# 12-shot calibrated Brier score in the completed SICModelv11 valence run:
# runs/full/sic_trial_bigru_v11_mldg_brier_ablation/DREAMER/valence/
# suite_65452590/full/
# dreamer_valence_sic_trial_bigru_v11_mldg_full_full_20260827_213158/
# hyperparameter_search_summary.csv
#
# DREAMER arousal includes subjects with only one low-arousal trial. Use two
# trials per selected subject so every balanced draw is feasible, and expand
# each episode from 12 to 18 subjects to preserve 36 total trials per update.

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="${PROJECT_DIR:-$HOME/EEGProc}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"

# Match the winning valence run's full training and calibration budget.
SOURCE_EPOCHS="${SOURCE_EPOCHS:-3}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-20}"
SOURCE_BATCH_SIZE="${SOURCE_BATCH_SIZE:-64}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-64}"
PREDICTION_DIAGNOSTICS_MAX_SAMPLES="${PREDICTION_DIAGNOSTICS_MAX_SAMPLES:-10000}"
SUITE_ID="${SLURM_JOB_ID:-manual}"

CALIBRATION_LEVEL_ARGS=(
    --calibration-level 3 6
    --calibration-level 6 3
    --calibration-level 9 2
)

cd "$PROJECT_DIR"

# Create or reuse the shared Python 3.12 environment. The lock prevents
# simultaneous jobs from modifying it at the same time.
if command -v flock >/dev/null 2>&1; then
    (
        flock -x 9
        if [[ ! -x "$VENV_DIR/bin/python" ]]; then
            python -m venv "$VENV_DIR"
            "$VENV_DIR/bin/python" -m pip install --upgrade pip
            "$VENV_DIR/bin/python" -m pip install -e . tensorflow==2.20.0
        elif [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
            "$VENV_DIR/bin/python" -m pip install -e . tensorflow==2.20.0
        fi
    ) 9>"$PROJECT_DIR/.venv312_install.lock"
else
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        python -m venv "$VENV_DIR"
        "$VENV_DIR/bin/python" -m pip install --upgrade pip
        "$VENV_DIR/bin/python" -m pip install -e . tensorflow==2.20.0
    elif [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
        "$VENV_DIR/bin/python" -m pip install -e . tensorflow==2.20.0
    fi
fi
source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
# Let TensorFlow reuse fragmented GPU memory during MLDG decoder gradients.
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Locate CUDA libdevice for TensorFlow/XLA.
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

# One fixed configuration. Wrapping the layer-width lists in "fixed" keeps
# them as architectures rather than treating their entries as search values.
MODEL_CONFIG="$(python - <<'PY'
import json

print(json.dumps({
    "optimizer_name": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 5e-5,
    "vrex_penalty_weight": 1.0,

    "mldg_meta_train_subjects": 12,
    "mldg_meta_test_subjects": 6,
    "mldg_trials_per_subject": 2,
    "mldg_steps_per_epoch": 20,
    "mldg_inner_learning_rate": 1e-4,
    "mldg_meta_test_weight": 1.0,
    "mldg_seed": 42,

    "gcn_units": {"fixed": [128, 64]},
    "gcn_dropout": 0.1,
    "gcn_activation": "relu",
    "gcn_use_batch_norm": False,
    "spectral_gru_units": 384,
    "spectral_gru_dropout": 0.2,
    "mi_n_neighbors": 3,
    "mi_random_state": 42,
    "mi_zero_diagonal": False,
    "mi_band_reduction": "mean",
    "mi_max_observations": 15000,

    "cnn3d_filters": {"fixed": [32, 64, 128]},
    "cnn3d_temporal_kernel_size": 7,
    "cnn3d_spatial_kernel_size": 3,
    "cnn3d_spatial_pool_sizes": {"fixed": [2, 2, 1]},
    "cnn3d_dropout": 0.2,
    "cnn3d_grid_size": 9,

    "classifier_rnn_type": "bigru",
    "classifier_rnn_units": {"fixed": [128, 64]},
    "n_classifier_rnn_layers": 2,
    "classifier_rnn_dropout": 0.4,

    "focal_gamma": 0.5,
    "focal_alpha": None,
    "vc_loss_weight": 1.0,
    "vc_alpha": 1.0,
    "vc_beta": 0.3,
    "vc_gamma": 0.0,
    "vc_lambda": 0.05,
    "update_vc_discriminator": False,

    "use_subject_adversarial": True,
    "subject_adversarial_weight": 0.6,
    "subject_loss_weight": 1.0,
    "subject_hidden_units": 64,
    "subject_dropout": 0.0,

    "use_gcn_gru_branch": True,
    "use_cnn3d_branch": True,
    "use_decoder": True,
    "reconstruction_loss_weight": 0.1,
    "decoder_dropout": 0.1,
    "remove_median_label": False,

    "calibration_unfreeze_layers": 2,
    "calibration_use_vc_target": True,
    "calibration_vc_alpha": 1.0,
    "calibration_vc_beta": 0.3,
    "calibration_vc_gamma": 0.0,
    "calibration_vc_lambda": 0.05,
}))
PY
)"

echo "SIC builder: v11"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Dataset/target: DREAMER arousal"
echo "Scope: all 23 LOSO target subjects"
echo "Training: MLDG, $SOURCE_EPOCHS source epochs, 12+6 subjects x 2 trials = 36 trials/episode"
echo "Calibration: $CALIBRATION_EPOCHS epochs at 3/6/9/12 shots"
echo "Selection/reporting metric: 12-shot calibrated Brier score"
echo "Configuration source: rank 1 from v11 valence suite 65452590"
echo "TensorFlow GPU allocator: $TF_GPU_ALLOCATOR"
python --version
nvidia-smi

python -m src.eegproc.deep_learning.joint_architectures.SICModelv11.sic_model_train \
    --training-protocol loso_validation \
    --dataset dreamer \
    --raw-eeg-npy "$EEG_PATH" \
    --raw-labels-npy "$LABELS_PATH" \
    --label-dimension arousal \
    --classification-level trial \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir "runs/full/sic_trial_bigru_v11_mldg_best_valence_hps/DREAMER/arousal/suite_${SUITE_ID}" \
    --run-name "dreamer_arousal_sic_trial_bigru_v11_mldg_best_valence_hps_full" \
    --training-method mldg \
    --source-epochs "$SOURCE_EPOCHS" \
    --source-batch-size "$SOURCE_BATCH_SIZE" \
    --validation-subjects 0 \
    --no-early-stopping \
    --calibration-epochs "$CALIBRATION_EPOCHS" \
    --calibration-batch-size "$CALIBRATION_BATCH_SIZE" \
    "${CALIBRATION_LEVEL_ARGS[@]}" \
    --calibration-selection-shots 12 \
    --calibration-learning-rate 0.0001 \
    --calibration-optimizer adamw \
    --calibration-weight-decay 0.00005 \
    --calibration-seed 42 \
    --selection-metric brier_score \
    --hyperparameter-selection-level calibration \
    --decision-threshold 0.5 \
    --prediction-diagnostics \
    --prediction-diagnostics-metric brier_score \
    --prediction-diagnostics-every-n-epochs 1 \
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
    --hyperparameters-json "$MODEL_CONFIG"
