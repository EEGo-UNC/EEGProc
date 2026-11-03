#!/bin/bash
#SBATCH --job-name=full_run_v15_valence
#SBATCH --output=full_run_v15_valence_%A_%a.out
#SBATCH --error=full_run_v15_valence_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --array=0
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

# One fixed configuration: run 5168991_12, all 23 LOSO targets.
# Select one common epoch maximizing mean held-out-user balanced accuracy.
# Reload that epoch for final zero-shot evaluation and 3-shot calibration.
#
# SICModelv15 adds the learned convex joint reconstruction. Its v15 defaults
# are made explicit below: initial alpha=0.5 and auxiliary branch weight=0.25.
# Each task's four GPUs run two folds concurrently on pairs (0,1) and (2,3).
# Each episode uses 12 meta-train + 6 meta-test subjects x 2 distinct trials:
# 24/12 trials globally and 12/6 per GPU.

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-}"
if [[ "$TASK_INDEX" != "0" ]]; then
    echo "ERROR: submit this script with sbatch; SLURM_ARRAY_TASK_ID must be 0 (one fixed configuration)." >&2
    exit 1
fi

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="${PROJECT_DIR:-$HOME/EEGProc}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"

# Full run defaults: 12 source epochs and 1 calibration epoch.
SOURCE_EPOCHS="${SOURCE_EPOCHS:-12}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-1}"
SOURCE_BATCH_SIZE="${SOURCE_BATCH_SIZE:-64}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-64}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-82800}"
TRAINING_SEED="${TRAINING_SEED:-42}"
SUITE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"

SIC_DIR="$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv15"
CONFIGS="$SIC_DIR/sic_v15_valence_run_5168991_12.json"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/runs/full/sic_v15_valence_oracle_5168991_12/suite_${SUITE_ID}/task_${TASK_INDEX}}"

cd "$PROJECT_DIR"

# Create or reuse the shared Python 3.12 environment. A lock prevents
# simultaneous jobs from modifying it at the same time.
if command -v flock >/dev/null 2>&1; then
    (
        flock -x 9
        if [[ ! -x "$VENV_DIR/bin/python" ]]; then
            python -m venv "$VENV_DIR"
            "$VENV_DIR/bin/python" -m pip install --upgrade pip
            "$VENV_DIR/bin/python" -m pip install -e ".[deep-learning]"
        elif [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
            "$VENV_DIR/bin/python" -m pip install -e ".[deep-learning]"
        fi
    ) 9>"$PROJECT_DIR/.venv312_install.lock"
else
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        python -m venv "$VENV_DIR"
        "$VENV_DIR/bin/python" -m pip install --upgrade pip
        "$VENV_DIR/bin/python" -m pip install -e ".[deep-learning]"
    elif [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
        "$VENV_DIR/bin/python" -m pip install -e ".[deep-learning]"
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

# Print the same single fixed configuration passed to the oracle controller.
MODEL_CONFIG="$(python "$SIC_DIR/sic_v15_valence_sweep.py" list --configs "$CONFIGS" --seed "$TRAINING_SEED")"

echo "SIC builder: v15"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array: $SUITE_ID task $TASK_INDEX (one fixed configuration: run 5168991_12)"
echo "Node: $(hostname)"
echo "Dataset/target: DREAMER valence"
echo "Scope: all 23 LOSO target subjects"
echo "Training: MLDG, $SOURCE_EPOCHS source epochs, 10 steps/epoch"
echo "Parallelism: 2 folds x 2 GPUs; episode trials: 24 meta-train / 12 meta-test"
echo "Calibration: $CALIBRATION_EPOCHS epochs at 3 shots (6 folds)"
echo "Oracle selection: one common epoch maximizing mean held-out-user balanced accuracy at threshold 0.5"
echo "Fixed settings: focal gamma=0.5 subject loss=0.3 reconstruction loss=0.0"
echo "Selected model configuration: $MODEL_CONFIG"
echo "VC settings: vc_alpha=2.0 vc_beta=0.6 vc_lambda=0.05 vc_logit_scale=16.0"
echo "Joint reconstruction: initial alpha=0.5 auxiliary branch weight=0.25"
echo "Deterministic training: enabled; base seed=$TRAINING_SEED; subject seed=base+target ID"
echo "TensorFlow GPU allocator: $TF_GPU_ALLOCATOR"
echo "Git commit: $(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "WARNING: tracked worktree changes are present; preserve this exact checkout with the results."
fi
python --version
nvidia-smi

# Verify the allocation and the two-GPU training path before the full run.
python - <<'PY_GPU'
import tensorflow as tf
n = len(tf.config.list_physical_devices("GPU"))
if n != 4:
    raise SystemExit(f"This full run requires exactly 4 allocated GPUs; visible={n}")
PY_GPU
EEGPROC_TEST_GPUS=1 python -m src.tests.test_sic_v15_multi_gpu

# Source fits use all 22 other users, with no validation holdout. Every
# completed epoch saves its actual model. Selection waits for all 23 users.
python "$SIC_DIR/sic_v15_valence_sweep.py" run \
    --configs "$CONFIGS" \
    --config-index 0 \
    --output-dir "$OUTPUT_DIR" \
    --eeg "$EEG_PATH" --labels "$LABELS_PATH" \
    --source-epochs "$SOURCE_EPOCHS" \
    --source-batch-size "$SOURCE_BATCH_SIZE" \
    --seed "$TRAINING_SEED" \
    --deterministic-training \
    --time-budget-seconds "$TIME_BUDGET_SECONDS"

# Use only the selected common-epoch model set. A failed or incomplete source
# run stops the script before this step. Final metrics are checked against the
# selected oracle score and written to OUTPUT_DIR/final_metrics.json.
python "$SIC_DIR/sic_v15_valence_sweep.py" calibrate "$OUTPUT_DIR" \
    --calibration-epochs "$CALIBRATION_EPOCHS" \
    --calibration-batch-size "$CALIBRATION_BATCH_SIZE"
