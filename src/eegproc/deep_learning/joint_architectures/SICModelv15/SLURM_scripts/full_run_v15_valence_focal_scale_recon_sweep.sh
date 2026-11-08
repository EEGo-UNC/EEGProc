#!/bin/bash
#SBATCH --job-name=v15_valence_focal_scale_recon
#SBATCH --output=v15_valence_focal_scale_recon_%A_%a.out
#SBATCH --error=v15_valence_focal_scale_recon_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --array=0-8%2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -Eeuo pipefail
trap 'status=$?; printf "ERROR: line %s exited %s: %s\n" "$LINENO" "$status" "$BASH_COMMAND" >&2; exit "$status"' ERR

# Three logit-scale configurations (8/16/32), each with 48 source epochs.
# Fixed: focal gamma 0.5, reconstruction weight 0.4,
# and 8 MLDG episodes per epoch. Subject loss stays 0.3.
# Keep the legacy 24-configuration manifest for previously queued jobs.
# Three source jobs cover each configuration's targets: 0-7, 8-15, 16-22.
# Collectors select ONE epoch across all 23 users after the source array ends.
# Run with bash to submit both arrays. DRY_RUN=1 previews without submitting.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../../../../.." && pwd)}}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"
SIC_DIR="$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv15"
RUNNER="$SIC_DIR/sic_v15_valence_sweep.py"
COLLECTOR="$SIC_DIR/collect_valence_sweep_shards.py"
CONFIGS="$SIC_DIR/sic_v15_valence_scale_configs.json"
STORAGE="$SIC_DIR/valence_sweep_storage.py"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
LOG_DIR="${LOG_DIR:-}"
METRICS_ONLY="${METRICS_ONLY:-1}"
[[ "$METRICS_ONLY" =~ ^[01]$ ]] || { echo "METRICS_ONLY must be 0 or 1." >&2; exit 1; }
if [[ "$METRICS_ONLY" == 1 ]]; then
    MIN_FREE_GIB="${MIN_FREE_GIB:-1}"
    RUN_CALIBRATION="${RUN_CALIBRATION:-0}"
else
    MIN_FREE_GIB="${MIN_FREE_GIB:-100}"
    RUN_CALIBRATION="${RUN_CALIBRATION:-1}"
fi
MIN_FREE_INODES="${MIN_FREE_INODES:-100000}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-48}"
SOURCE_BATCH_SIZE="${SOURCE_BATCH_SIZE:-64}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-1}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-64}"
TRAINING_SEED="${TRAINING_SEED:-42}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-82800}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
PHASE="${1:---task}"

[[ "$PHASE" == --task || "$PHASE" == --collect ]] || { echo "Use --task or --collect." >&2; exit 1; }
[[ "$RUN_CALIBRATION" =~ ^[01]$ ]] || { echo "RUN_CALIBRATION must be 0 or 1." >&2; exit 1; }
[[ "$METRICS_ONLY" != 1 || "$RUN_CALIBRATION" == 0 ]] || {
    echo "METRICS_ONLY=1 cannot calibrate unsaved models. Set RUN_CALIBRATION=0 or METRICS_ONLY=0." >&2
    exit 1
}
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ $# -gt 0 ]]; then
        echo "ERROR: --task/--collect require SLURM_ARRAY_TASK_ID." >&2
        exit 1
    fi
    [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid MAX_CONCURRENT." >&2; exit 1; }
    SUBMIT_PYTHON="${SUBMIT_PYTHON:-python3}"
    STORAGE_PATHS="$("$SUBMIT_PYTHON" "$STORAGE" paths --storage-root "${SWEEP_STORAGE_ROOT:-}" \
        --output-root "$OUTPUT_ROOT" --log-dir "$LOG_DIR")"
    { IFS= read -r OUTPUT_ROOT; IFS= read -r LOG_DIR; } <<< "$STORAGE_PATHS"
    "$SUBMIT_PYTHON" "$STORAGE" check --output-root "$OUTPUT_ROOT" --log-dir "$LOG_DIR" \
        --min-free-gib "$MIN_FREE_GIB" --min-free-inodes "$MIN_FREE_INODES" --show-quota
    COUNT="$("$SUBMIT_PYTHON" "$RUNNER" list --configs "$CONFIGS" --count)"
    SOURCE_TASKS=$((COUNT * 3))
    export PROJECT_DIR VENV_DIR EEG_PATH LABELS_PATH OUTPUT_ROOT
    export LOG_DIR MIN_FREE_GIB MIN_FREE_INODES
    export SOURCE_EPOCHS SOURCE_BATCH_SIZE CALIBRATION_EPOCHS CALIBRATION_BATCH_SIZE
    export METRICS_ONLY RUN_CALIBRATION TRAINING_SEED TIME_BUDGET_SECONDS
    # Slurm opens logs before the batch script runs: set absolute paths here.
    SOURCE_SUBMIT=(sbatch --parsable --array="0-$((SOURCE_TASKS-1))%${MAX_CONCURRENT}" --chdir="$PROJECT_DIR"
                   --export=ALL --output="$LOG_DIR/v15_valence_focal_scale_recon_%A_%a.out"
                   --error="$LOG_DIR/v15_valence_focal_scale_recon_%A_%a.err" "$SCRIPT_PATH" --task)
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        SOURCE_JOB_ID="SOURCE_ARRAY_ID"
    else
        SOURCE_JOB="$("${SOURCE_SUBMIT[@]}")"
        SOURCE_JOB_ID="${SOURCE_JOB%%;*}"
        echo "Submitted source array $SOURCE_JOB_ID"
    fi
    COLLECT_SUBMIT=(sbatch --parsable --dependency="afterany:${SOURCE_JOB_ID}" --array="0-$((COUNT-1))%${MAX_CONCURRENT}"
                    --job-name=v15_valence_collect --gres=gpu:2 --chdir="$PROJECT_DIR"
                    --output="$LOG_DIR/v15_valence_collect_%A_%a.out" --error="$LOG_DIR/v15_valence_collect_%A_%a.err"
                    --export="ALL,SWEEP_SUITE_ID=${SOURCE_JOB_ID}" "$SCRIPT_PATH" --collect)
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf 'Source submission: '; printf '%q ' "${SOURCE_SUBMIT[@]}"; printf '\n'
        printf 'Collect submission: '; printf '%q ' "${COLLECT_SUBMIT[@]}"; printf '\n'
        echo "$COUNT configurations; $SOURCE_TASKS source tasks (3 shards each); $COUNT collectors."
        echo "24:00:00 per task; at most $MAX_CONCURRENT concurrent tasks; source 4 GPUs/task, collector 2 GPUs/task."
        echo "Gamma 0.5; scale 8/16/32; reconstruction 0.4; MLDG episodes 8; subject loss 0.3"
        echo "Outer/source epochs: $SOURCE_EPOCHS; calibration enabled: $RUN_CALIBRATION"
        echo "Metrics only: $METRICS_ONLY (1: scores only, temporary data removed; 0: retain models and processed data)"
        echo "Outputs: $OUTPUT_ROOT; Slurm logs: $LOG_DIR"
        exit 0
    fi
    COLLECT_JOB="$("${COLLECT_SUBMIT[@]}")"
    echo "Submitted collector array ${COLLECT_JOB%%;*}, dependent on source array $SOURCE_JOB_ID"
    echo "Outputs: $OUTPUT_ROOT/suite_$SOURCE_JOB_ID; Slurm logs: $LOG_DIR"
    exit 0
fi

ARRAY_INDEX="$SLURM_ARRAY_TASK_ID"
[[ "$ARRAY_INDEX" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Invalid SLURM_ARRAY_TASK_ID." >&2; exit 1; }
if [[ "$PHASE" == --collect ]]; then
    [[ "$ARRAY_INDEX" -lt 3 ]] || { echo "Collector array index must be 0..2." >&2; exit 1; }
    [[ -n "${SWEEP_SUITE_ID:-}" ]] || { echo "Collectors require SWEEP_SUITE_ID from the source array." >&2; exit 1; }
    CONFIG_INDEX="$ARRAY_INDEX"
else
    [[ "$ARRAY_INDEX" -lt 9 ]] || { echo "Source array index must be 0..8." >&2; exit 1; }
    CONFIG_INDEX=$((ARRAY_INDEX / 3))
    SHARD_INDEX=$((ARRAY_INDEX % 3))
fi

[[ -n "$OUTPUT_ROOT" && -n "$LOG_DIR" ]] || {
    echo "Submit with bash (the wrapper), so storage is checked and Slurm logs are routed before submission." >&2
    exit 1
}
cd "$PROJECT_DIR"
module purge
module load python/3.12.4
if [[ "$PHASE" != --collect || "$METRICS_ONLY" == 0 ]]; then
    module load cuda/12.9
    module load cudnn/9.11.0
fi
[[ -x "$VENV_DIR/bin/python" ]] || { echo "Prepare the environment at $VENV_DIR first." >&2; exit 1; }
source "$VENV_DIR/bin/activate"
python "$STORAGE" check --output-root "$OUTPUT_ROOT" --log-dir "$LOG_DIR" \
    --min-free-gib "$MIN_FREE_GIB" --min-free-inodes "$MIN_FREE_INODES"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TF_GPU_ALLOCATOR=cuda_malloc_async
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TRAINING_SEED PYTHONHASHSEED="$TRAINING_SEED"
export TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8

# Metrics-only collection uses the saved epoch scores, without CUDA or models.
if [[ "$PHASE" == --collect && "$METRICS_ONLY" == 1 ]]; then
    exec python "$COLLECTOR" "$OUTPUT_ROOT/suite_${SWEEP_SUITE_ID}/configuration_${CONFIG_INDEX}"
fi

# Reuse the prepared CUDA environment; array tasks do not install packages.
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

SUITE_ID="${SWEEP_SUITE_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
OUTPUT_DIR="$OUTPUT_ROOT/suite_${SUITE_ID}/configuration_${CONFIG_INDEX}"
MODEL_CONFIG="$(python - "$CONFIGS" "$CONFIG_INDEX" "$TRAINING_SEED" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    document = json.load(handle)
entry = document["configurations"][int(sys.argv[2])]
config = {**document["base"], **entry["overrides"]}
config.update(mldg_seed=int(sys.argv[3]), mi_random_state=int(sys.argv[3]))
print(json.dumps({"name": entry["name"], "hyperparameters": config}, sort_keys=True))
PY
)"
echo "SICv15 focal/scale/reconstruction/episode sweep: suite $SUITE_ID configuration $CONFIG_INDEX phase $PHASE"
echo "Selected configuration: $MODEL_CONFIG"
echo "Outer/source epochs: $SOURCE_EPOCHS; MLDG 12/6 subjects x 2 trials; no validation split"
echo "Oracle: one common epoch maximizing mean balanced accuracy over all 23 held-out users at threshold 0.5"
echo "Metrics only: $METRICS_ONLY; calibration enabled: $RUN_CALIBRATION"
echo "Git commit: $(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "WARNING: tracked worktree changes are present; preserve this exact checkout with the results."
fi
python --version
nvidia-smi

if [[ "$PHASE" == --collect ]]; then
    python "$COLLECTOR" "$OUTPUT_DIR"
    if [[ "$RUN_CALIBRATION" == 1 ]]; then
        python "$RUNNER" calibrate "$OUTPUT_DIR" \
            --calibration-epochs "$CALIBRATION_EPOCHS" \
            --calibration-batch-size "$CALIBRATION_BATCH_SIZE"
    fi
    exit 0
fi

python - <<'PY_GPU'
import tensorflow as tf
count = len(tf.config.list_physical_devices("GPU"))
if count != 4:
    raise SystemExit(f"Each source task requires exactly four allocated GPUs; visible={count}.")
PY_GPU
if [[ "${GPU_PREFLIGHT:-1}" == 1 ]]; then
    EEGPROC_TEST_GPUS=1 python -m src.tests.test_sic_v15_multi_gpu
fi
TARGET_GROUPS=("0 1 2 3 4 5 6 7" "8 9 10 11 12 13 14 15" "16 17 18 19 20 21 22")
read -r -a TARGET_LIST <<< "${TARGET_GROUPS[$SHARD_INDEX]}"
echo "Source shard $SHARD_INDEX: held-out users ${TARGET_LIST[*]}; each uses all 22 other source users"
echo "Source soft budget: $TIME_BUDGET_SECONDS seconds; Slurm hard limit: 24 hours"
RUN_COMMAND=(python "$RUNNER" run)
if [[ "$METRICS_ONLY" == 1 ]]; then
    RUN_COMMAND+=(--metrics-only)
fi
exec "${RUN_COMMAND[@]}" \
    --configs "$CONFIGS" \
    --config-index "$CONFIG_INDEX" \
    --output-dir "$OUTPUT_DIR/shard_${SHARD_INDEX}" \
    --eeg "$EEG_PATH" --labels "$LABELS_PATH" \
    --source-epochs "$SOURCE_EPOCHS" \
    --source-batch-size "$SOURCE_BATCH_SIZE" \
    --seed "$TRAINING_SEED" \
    --deterministic-training \
    --time-budget-seconds "$TIME_BUDGET_SECONDS" \
    --target-subjects "${TARGET_LIST[@]}"
