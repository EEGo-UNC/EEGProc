#!/bin/bash
#SBATCH --job-name=smoke_valence_0_3_grid
#SBATCH --output=smoke_valence_0_3_grid_%A_%a.out
#SBATCH --error=smoke_valence_0_3_grid_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --array=0-1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00

set -euo pipefail

# Grid-search smoke run for DREAMER valence targets 0, 1, 2, and 3.
# Two Slurm array tasks test VC logit scales (inverse temperatures) 64 and 16.
# Each task runs the fixed focal-gamma/VC-weight smoke configuration and uses
# the same deterministic per-target initialization for a fair comparison.
# Selection maximizes mean zero-shot LOSO balanced accuracy.
#
# SICModelv15 uses the learned convex joint reconstruction with initial
# alpha=0.5 and auxiliary branch weight=0.25. Four allocated GPUs run two
# folds concurrently on pairs (0,1) and (2,3). To isolate the label dimension,
# preserve the arousal smoke run's 12 meta-train + 6 meta-test subjects x 2
# distinct trials setup: 24/12 trials globally and 12/6 per GPU.

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="${PROJECT_DIR:-$HOME/EEGProc}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"

# Match the arousal smoke run: 4 source and 10 calibration epochs.
SOURCE_EPOCHS="${SOURCE_EPOCHS:-4}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-10}"
SOURCE_BATCH_SIZE="${SOURCE_BATCH_SIZE:-64}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-64}"
PREDICTION_DIAGNOSTICS_MAX_SAMPLES="${PREDICTION_DIAGNOSTICS_MAX_SAMPLES:-10000}"
TRAINING_SEED="${TRAINING_SEED:-42}"
TEMPERATURES=(64)
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "$TASK_INDEX" =~ ^[0-9]+$ ]] || (( TASK_INDEX >= ${#TEMPERATURES[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID must be between 0 and $((${#TEMPERATURES[@]} - 1)); got $TASK_INDEX."
    exit 1
fi
VC_LOGIT_SCALE="${TEMPERATURES[$TASK_INDEX]}"
export VC_LOGIT_SCALE
SUITE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
TARGET_SUBJECTS=(0 1 2 3)

CALIBRATION_LEVEL_ARGS=(
    --calibration-level 3 6
    --calibration-level 6 3
    --calibration-level 9 2
    --calibration-level 12 3
)

cd "$PROJECT_DIR"

# Create or reuse the shared Python 3.12 environment. A lock prevents
# simultaneous jobs from modifying it at the same time.
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
export TRAINING_SEED
export PYTHONHASHSEED="$TRAINING_SEED"
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Use CUDA's asynchronous allocator so TensorFlow can reuse fragmented GPU
# memory for the large decoder-gradient temporaries created during MLDG.
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

# Six configurations vary only focal gamma and VC classification weight.
# Reconstruction, subject-adversarial, and other VC regularization weights
# remain fixed so zero-shot balanced accuracy isolates this small search.
MODEL_CONFIG="$(python - <<'PY'
import json
import os

seed = int(os.environ["TRAINING_SEED"])

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
    "mldg_seed": seed,

    "gcn_units": {"fixed": [128, 64]},
    "gcn_dropout": 0.1,
    "gcn_activation": "relu",
    "gcn_use_batch_norm": False,
    "spectral_gru_units": 384,
    "spectral_gru_dropout": 0.2,
    "mi_n_neighbors": 3,
    "mi_random_state": seed,
    "mi_zero_diagonal": False,
    "mi_band_reduction": "mean",
    "mi_max_observations": 15000,

    "bilstm_units": 63,
    "n_bilstm_layers": 1,
    "bilstm_dropout": 0.2,

    "classifier_rnn_type": "bigru",
    "classifier_rnn_units": {"fixed": [128, 64]},
    "n_classifier_rnn_layers": 2,
    "classifier_rnn_dropout": 0.4,

    "focal_gamma": {"grid": [0.8]},
    "focal_alpha": None,
    "vc_loss_weight": 1.0,
    "vc_alpha": {"grid": [2.0]},
    "vc_beta": 0.3,
    "vc_gamma": 0.0,
    "vc_lambda": 0.05,
    "vc_logit_scale": float(os.environ["VC_LOGIT_SCALE"]),
    "update_vc_discriminator": False,

    "use_subject_adversarial": True,
    "subject_adversarial_weight": 0.6,
    "subject_loss_weight": 0.2,
    "subject_hidden_units": 64,
    "subject_dropout": 0.0,

    "use_gcn_gru_branch": True,
    "use_bilstm_branch": True,
    "use_decoder": True,
    "reconstruction_loss_weight": 0.6,
    "decoder_dropout": 0.1,
    "joint_reconstruction_auxiliary_weight": 0.25,
    "joint_reconstruction_initial_alpha": 0.5,
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

echo "SIC builder: v15"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array task: $TASK_INDEX"
echo "Node: $(hostname)"
echo "Dataset/target: DREAMER valence"
echo "Target subjects: ${TARGET_SUBJECTS[*]}"
echo "Training: MLDG, $SOURCE_EPOCHS source epochs, 12+6 subjects x 2 trials = 36 trials/episode"
echo "Parallelism: 2 folds x 2 GPUs; episode trials: 24 meta-train / 12 meta-test"
echo "Per GPU: 12 meta-train / 6 meta-test trials; full-episode VC statistics"
echo "Valence uses the identical 2-distinct-trials-per-subject episode design."
echo "Calibration: $CALIBRATION_EPOCHS epochs at 3/6/9/12 shots"
echo "Temperature grid: vc_logit_scale=$VC_LOGIT_SCALE (task values: 64,16)"
echo "Within-task configuration: focal_gamma=1.0; vc_alpha=2.0; reconstruction=0.6"
echo "Selection: maximize zero-shot LOSO balanced accuracy"
echo "Subject loss weight: 0.2"
echo "Joint reconstruction: weight=0.6 initial alpha=0.5 auxiliary branch weight=0.25"
echo "Configurations: 1 per task, 2 across the array; subject loss weight fixed at 0.2"
echo "Deterministic training: enabled; base seed=$TRAINING_SEED; subject seed=base+target ID"
echo "TensorFlow GPU allocator: $TF_GPU_ALLOCATOR"
echo "Git commit: $(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "WARNING: tracked worktree changes are present; preserve this exact checkout with the results."
fi
python --version
nvidia-smi

# Verify the allocation and the two-GPU training path before the smoke grid.
python - <<'PY_GPU'
import tensorflow as tf
n = len(tf.config.list_physical_devices("GPU"))
if n != 4:
    raise SystemExit(f"This smoke grid requires exactly 4 allocated GPUs; visible={n}")
PY_GPU
EEGPROC_TEST_GPUS=1 python -m src.tests.test_sic_v15_multi_gpu

python -m src.eegproc.deep_learning.joint_architectures.SICModelv15.sic_model_train \
    --training-protocol loso_validation \
    --dataset dreamer \
    --raw-eeg-npy "$EEG_PATH" \
    --raw-labels-npy "$LABELS_PATH" \
    --label-dimension valence \
    --classification-level trial \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir "runs/smoke/sic_v15_valence_grid/DREAMER/valence/suite_${SUITE_ID}/temperature_${VC_LOGIT_SCALE}/users_0_3" \
    --run-name "smoke_valence_0_3_grid_temperature_${VC_LOGIT_SCALE}" \
    --training-method mldg \
    --source-epochs "$SOURCE_EPOCHS" \
    --source-batch-size "$SOURCE_BATCH_SIZE" \
    --validation-subjects 0 \
    --validation-seed "$TRAINING_SEED" \
    --no-early-stopping \
    --calibration-epochs "$CALIBRATION_EPOCHS" \
    --calibration-batch-size "$CALIBRATION_BATCH_SIZE" \
    "${CALIBRATION_LEVEL_ARGS[@]}" \
    --calibration-selection-shots 12 \
    --calibration-learning-rate 0.0001 \
    --calibration-optimizer adamw \
    --calibration-weight-decay 0.00005 \
    --calibration-seed "$TRAINING_SEED" \
    --selection-metric balanced_accuracy \
    --hyperparameter-selection-level losocv \
    --decision-threshold 0.5 \
    --prediction-diagnostics \
    --prediction-diagnostics-metric balanced_accuracy \
    --prediction-diagnostics-every-n-epochs 1 \
    --prediction-diagnostics-max-samples "$PREDICTION_DIAGNOSTICS_MAX_SAMPLES" \
    --prediction-diagnostics-threshold-tolerance 0.01 \
    --prediction-diagnostics-seed "$TRAINING_SEED" \
    --ece-bins 15 \
    --max-subjects 4 \
    --target-subjects "${TARGET_SUBJECTS[@]}" \
    --n-jobs 2 \
    --gpus-per-fold 2 \
    --gpu-ids 0 1 2 3 \
    --cpus-per-worker 4 \
    --verbose 2 \
    --seed "$TRAINING_SEED" \
    --deterministic-training \
    --label-threshold-mode global \
    --median-label 3 \
    --window-sec 1.0 \
    --fs 128 \
    --window-overlap 0.0 \
    --window-normalization global_rms \
    --hyperparameters-json "$MODEL_CONFIG"
