#!/bin/bash
#SBATCH --job-name=sicv15_valence_oracle
#SBATCH --output=sicv15_valence_oracle_%A_%a.out
#SBATCH --error=sicv15_valence_oracle_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --array=0-15%2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

# Running with bash submits one array; direct sbatch also works. No jobs are
# submitted with DRY_RUN=1. Hyperparameter lists are fixed architecture widths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$SCRIPT_DIR")}}"
CONFIGS="${CONFIGS:-$PROJECT_DIR/scripts/sic_v15_valence_configs.json}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-6}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-21000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ "${1:-}" == "--task" ]]; then
        echo "ERROR: --task requires SLURM_ARRAY_TASK_ID." >&2
        exit 1
    fi
    [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid MAX_CONCURRENT" >&2; exit 1; }
    SUBMIT_PYTHON="${SUBMIT_PYTHON:-python3}"
    COUNT="$("$SUBMIT_PYTHON" "$PROJECT_DIR/scripts/sic_v15_valence_sweep.py" list --configs "$CONFIGS" --count)"
    # Slurm executes a spool copy of this file, so persist the real checkout.
    export PROJECT_DIR CONFIGS VENV_DIR
    SUBMIT=(sbatch --array="0-$((COUNT-1))%${MAX_CONCURRENT}" --chdir="$PROJECT_DIR" "$SCRIPT_PATH" --task)
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf 'Submission: '; printf '%q ' "${SUBMIT[@]}"; printf '\n'
        echo "$COUNT configurations; 4 GPUs each; 06:00:00 per task; at most $MAX_CONCURRENT tasks concurrently."
        exit 0
    fi
    exec "${SUBMIT[@]}"
fi

cd "$PROJECT_DIR"
module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0
[[ -x "$VENV_DIR/bin/python" ]] || { echo "Prepare the environment at $VENV_DIR first." >&2; exit 1; }
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TF_GPU_ALLOCATOR=cuda_malloc_async
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Use the same XLA libdevice discovery as the SICv15 full run. Array workers
# reuse the prepared environment and do not install packages concurrently.
MODULE_CUDA_ROOT="${EBROOTCUDA:-${CUDA_HOME:-}}"
if [[ -z "$MODULE_CUDA_ROOT" ]] && command -v nvcc >/dev/null 2>&1; then
    MODULE_CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi
CUDA_ROOTS=()
for candidate in "$MODULE_CUDA_ROOT" "$VENV_DIR/lib/python3.12/site-packages/nvidia" /usr/local/cuda /opt/cuda; do
    [[ -d "$candidate" ]] && CUDA_ROOTS+=("$candidate")
done
[[ ${#CUDA_ROOTS[@]} -gt 0 ]] || { echo "Cannot find a CUDA installation." >&2; exit 1; }
LIBDEVICE_PATH="$(find "${CUDA_ROOTS[@]}" -type f -path '*/nvvm/libdevice/libdevice.10.bc' -print -quit 2>/dev/null || true)"
[[ -f "$LIBDEVICE_PATH" ]] || { echo "Cannot find libdevice.10.bc; prepare CUDA before submission." >&2; exit 1; }
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${LIBDEVICE_PATH%/nvvm/libdevice/libdevice.10.bc}"
if [[ -n "$MODULE_CUDA_ROOT" ]]; then
    export CUDA_HOME="$MODULE_CUDA_ROOT" CUDA_PATH="$MODULE_CUDA_ROOT"
fi

TASK_INDEX="$SLURM_ARRAY_TASK_ID"
[[ "$TASK_INDEX" =~ ^[0-9]+$ ]] || { echo "Invalid array task ID." >&2; exit 1; }
SUITE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs/sweeps/sic_v15_valence_oracle}"
TARGET_ARGS=()
if [[ -n "${TARGET_SUBJECTS:-}" ]]; then
    read -r -a TARGET_LIST <<< "$TARGET_SUBJECTS"
    TARGET_ARGS=(--target-subjects "${TARGET_LIST[@]}")
fi
echo "SICv15 valence oracle: array $SUITE_ID task $TASK_INDEX; 2 folds x 2 GPUs"
echo "Source epochs: $SOURCE_EPOCHS; soft budget: $TIME_BUDGET_SECONDS seconds; Slurm hard limit: 6 hours"
echo "Oracle: maximize mean subject balanced accuracy at one common epoch, threshold 0.5"
echo "Targets: ${TARGET_SUBJECTS:-all 23}; no target calibration during this sweep"
nvidia-smi
python - <<'PY_GPU'
import tensorflow as tf
count = len(tf.config.list_physical_devices("GPU"))
if count != 4:
    raise SystemExit(f"Each task requires exactly four allocated GPUs; visible={count}.")
PY_GPU
if [[ "${GPU_PREFLIGHT:-1}" == 1 ]]; then
    EEGPROC_TEST_GPUS=1 python -m src.tests.test_sic_v15_multi_gpu
fi
exec python "$PROJECT_DIR/scripts/sic_v15_valence_sweep.py" run \
    --configs "$CONFIGS" \
    --config-index "$TASK_INDEX" \
    --output-dir "$OUTPUT_ROOT/suite_${SUITE_ID}/configuration_${TASK_INDEX}" \
    --eeg "$EEG_PATH" --labels "$LABELS_PATH" \
    --source-epochs "$SOURCE_EPOCHS" \
    --time-budget-seconds "$TIME_BUDGET_SECONDS" \
    "${TARGET_ARGS[@]}"
