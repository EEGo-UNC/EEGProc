#!/bin/bash

set -Eeuo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "${PROJECT_DIR:-$(dirname "$SCRIPT_PATH")/../../../../../..}" && pwd)"
SIC_DIR="$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv15"
RUNNER="$SIC_DIR/sic_v15_valence_sweep.py"
COLLECTOR="$SIC_DIR/collect_valence_sweep_shards.py"
CONFIGS="$SIC_DIR/sic_v15_dreamer_reconstruction_configs.json"
STORAGE="$SIC_DIR/valence_sweep_storage.py"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
TRAINING_SEED="${TRAINING_SEED:-42}"
SOURCE_BATCH_SIZE="${SOURCE_BATCH_SIZE:-64}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-82800}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"
MIN_FREE_INODES="${MIN_FREE_INODES:-1000}"
PILOT_SUBJECT="${PILOT_SUBJECT:-0}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

phase="${1:-pilot}"
case "$phase" in
    pilot|full) mode="$phase" ;;
    --train|--collect) mode="${2:-}" ;;
    *) die "Usage: bash $SCRIPT_PATH [pilot|full]" ;;
esac
case "$mode" in
    pilot)
        SHARD_COUNT=1
        GPUS=2
        SOURCE_EPOCHS="${SOURCE_EPOCHS:-2}"
        [[ "$PILOT_SUBJECT" =~ ^(0|[1-9][0-9]*)$ && "$PILOT_SUBJECT" -lt 23 ]] ||
            die "PILOT_SUBJECT must be 0–22."
        ;;
    full)
        SHARD_COUNT=3
        GPUS=4
        SOURCE_EPOCHS="${SOURCE_EPOCHS:-48}"
        ;;
    *) die "Invalid run mode: $mode" ;;
esac
for name in SOURCE_EPOCHS SOURCE_BATCH_SIZE TIME_BUDGET_SECONDS MAX_CONCURRENT; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer."
done

if [[ -z "${SUBMIT_PYTHON:-}" ]]; then
    SUBMIT_PYTHON="$VENV_DIR/bin/python"
    [[ -x "$SUBMIT_PYTHON" ]] || SUBMIT_PYTHON=python3
fi
"$SUBMIT_PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' ||
    die "Load Python 3.12 or set SUBMIT_PYTHON to a Python 3.10+ interpreter."
ARM_COUNT="$("$SUBMIT_PYTHON" "$RUNNER" list --configs "$CONFIGS" --count)"
[[ "$ARM_COUNT" == 4 ]] || die "The reconstruction ablation requires four configurations."
CELL_COUNT=$((2 * ARM_COUNT))
TASK_COUNT=$((CELL_COUNT * SHARD_COUNT))

if [[ "$phase" == pilot || "$phase" == full ]]; then
    [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die "Submit the wrapper with bash from the login node."
    account="${USER:-$(id -un)}"
    work_root="${SWEEP_STORAGE_ROOT:-/work/users/${account:0:1}/${account:1:1}/$account/EEGProc}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$work_root/runs/sweeps/sic_v15_dreamer_reconstruction}"
    LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
    paths="$("$SUBMIT_PYTHON" "$STORAGE" paths --output-root "$OUTPUT_ROOT" --log-dir "$LOG_DIR")"
    { IFS= read -r OUTPUT_ROOT; IFS= read -r LOG_DIR; } <<< "$paths"
    train=(sbatch --parsable --job-name=v15_dreamer_reconstruction --partition=l40-gpu --qos=gpu_access
           --cpus-per-task=8 --mem=128G --time=24:00:00
           --gres="gpu:$GPUS" --array="0-$((TASK_COUNT - 1))%$MAX_CONCURRENT"
           --chdir="$PROJECT_DIR" --export=ALL
           --output="$LOG_DIR/train_%A_%a.out" --error="$LOG_DIR/train_%A_%a.err"
           "$SCRIPT_PATH" --train "$mode")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        SOURCE_JOB_ID=SOURCE_ARRAY_ID
        printf 'Training submission:'; printf ' %q' "${train[@]}"; printf '\n'
    else
        [[ -r "$EEG_PATH" && -r "$LABELS_PATH" ]] || die "DREAMER arrays are missing or unreadable."
        [[ -x "$VENV_DIR/bin/python" ]] || die "Prepare the Python environment at $VENV_DIR."
        git -C "$PROJECT_DIR" diff --quiet HEAD -- src/eegproc ||
            die "Use a clean checkout of the training code."
        SIC_V15_SOURCE_REVISION="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
        "$SUBMIT_PYTHON" "$STORAGE" check --output-root "$OUTPUT_ROOT" --log-dir "$LOG_DIR" \
            --min-free-gib "$MIN_FREE_GIB" --min-free-inodes "$MIN_FREE_INODES" --show-quota
        export PROJECT_DIR OUTPUT_ROOT LOG_DIR VENV_DIR EEG_PATH LABELS_PATH
        export SOURCE_EPOCHS SOURCE_BATCH_SIZE TRAINING_SEED TIME_BUDGET_SECONDS PILOT_SUBJECT
        export SIC_V15_SOURCE_REVISION
        submission="$("${train[@]}")"
        SOURCE_JOB_ID="${submission%%;*}"
        [[ "$SOURCE_JOB_ID" =~ ^[0-9]+$ ]] || die "Could not read the source job ID: $submission"
        echo "Submitted training array $SOURCE_JOB_ID"
    fi
    if [[ "$mode" == full ]]; then
        collect=(sbatch --parsable --partition=general --job-name=v15_dreamer_collect
                 --cpus-per-task=1 --mem=4G --time=00:30:00
                 --dependency="afterany:$SOURCE_JOB_ID" --array="0-$((CELL_COUNT - 1))%$MAX_CONCURRENT"
                 --chdir="$PROJECT_DIR" --export=ALL
                 --output="$LOG_DIR/collect_%A_%a.out" --error="$LOG_DIR/collect_%A_%a.err"
                 "$SCRIPT_PATH" --collect full "$SOURCE_JOB_ID")
        if [[ "${DRY_RUN:-0}" == 1 ]]; then
            printf 'Collector submission:'; printf ' %q' "${collect[@]}"; printf '\n'
        else
            collector_job="$("${collect[@]}")" ||
                die "The training array $SOURCE_JOB_ID was submitted, but the collector submission failed."
            echo "Submitted collector array ${collector_job%%;*}"
        fi
    fi
    if [[ "$mode" == pilot ]]; then
        echo "8 model fits on subject $PILOT_SUBJECT in $TASK_COUNT training tasks."
    else
        echo "184 model fits in $TASK_COUNT training tasks (shards 0–7, 8–15, 16–22)."
    fi
    echo "Fixed final epoch: $SOURCE_EPOCHS; results: $OUTPUT_ROOT/suite_$SOURCE_JOB_ID"
    exit 0
fi

index="${SLURM_ARRAY_TASK_ID:-}"
[[ "$index" =~ ^(0|[1-9][0-9]*)$ ]] || die "This phase requires a Slurm array index."
if [[ "$phase" == --collect ]]; then
    [[ "$mode" == full && "$index" -lt "$CELL_COUNT" ]] || die "Invalid collector index."
    suite_id="${3:-}"
    cell="$index"
else
    [[ "$index" -lt "$TASK_COUNT" ]] || die "Invalid training index."
    suite_id="${SLURM_ARRAY_JOB_ID:-}"
    cell=$((index / SHARD_COUNT))
    shard=$((index % SHARD_COUNT))
fi
[[ "$suite_id" =~ ^[0-9]+$ && -n "${OUTPUT_ROOT:-}" ]] || die "Missing source suite ID or output root."
[[ -n "${SIC_V15_SOURCE_REVISION:-}" &&
    "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$SIC_V15_SOURCE_REVISION" ]] ||
    die "The training checkout changed after submission."
git -C "$PROJECT_DIR" diff --quiet HEAD -- src/eegproc || die "The training checkout has local edits."
targets=(valence arousal)
target="${targets[cell / ARM_COUNT]}"
arm=$((cell % ARM_COUNT))
output="$OUTPUT_ROOT/suite_$suite_id/$target/configuration_$arm"

module purge
module load python/3.12.4
python="$VENV_DIR/bin/python"
[[ -x "$python" ]] || die "Missing Python environment at $VENV_DIR."
if [[ "$phase" == --collect ]]; then
    exec "$python" "$COLLECTOR" "$output"
fi

module load cuda/12.9
module load cudnn/9.11.0
cuda_root="${EBROOTCUDA:-${CUDA_HOME:-}}"
if [[ -z "$cuda_root" ]] && command -v nvcc >/dev/null; then
    cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi
[[ -f "$cuda_root/nvvm/libdevice/libdevice.10.bc" ]] || die "Cannot find CUDA libdevice in $cuda_root."
export CUDA_HOME="$cuda_root" CUDA_PATH="$cuda_root"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$cuda_root"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED="$TRAINING_SEED"
export TF_GPU_ALLOCATOR=cuda_malloc_async TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ "$mode" == pilot ]]; then
    subject_args=("$PILOT_SUBJECT")
else
    # Match the existing sweep exactly, including the smaller last shard.
    subject_groups=("0 1 2 3 4 5 6 7" "8 9 10 11 12 13 14 15" "16 17 18 19 20 21 22")
    read -r -a subject_args <<< "${subject_groups[shard]}"
    output="$output/shard_$shard"
fi
echo "$target configuration $arm, held-out subjects: ${subject_args[*]}"
exec "$python" "$RUNNER" run --configs "$CONFIGS" --config-index "$arm" \
    --label-dimension "$target" --fixed-final-epoch --output-dir "$output" \
    --eeg "$EEG_PATH" --labels "$LABELS_PATH" --source-epochs "$SOURCE_EPOCHS" \
    --source-batch-size "$SOURCE_BATCH_SIZE" --seed "$TRAINING_SEED" --deterministic-training \
    --time-budget-seconds "$TIME_BUDGET_SECONDS" --target-subjects "${subject_args[@]}"
