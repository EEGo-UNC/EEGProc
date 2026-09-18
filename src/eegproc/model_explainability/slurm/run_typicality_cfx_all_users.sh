#!/bin/bash
#SBATCH --job-name=typicality_cfx
#SBATCH --output=typicality_outs/typicality_cfx_%A_%a.out
#SBATCH --error=typicality_outs/typicality_cfx_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-22%8

set -euo pipefail

# One array task is one DREAMER held-out subject (0--22). The typicality
# runner owns source-only typicality calibration, held-out R(Z0) VCSC
# calibration, and paired base/typicality objectives.
# Required exports: MODELS_JSON and TYPICALITY_WEIGHT.

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"

find_project_dir() {
    local start candidate
    for start in "${SLURM_SUBMIT_DIR:-}" "$SCRIPT_DIR"; do
        [[ -n "$start" ]] || continue
        candidate="$(cd -- "$start" 2>/dev/null && pwd)" || continue
        while [[ "$candidate" != "/" ]]; do
            if [[ -d "$candidate/src/eegproc" ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
            candidate="$(dirname -- "$candidate")"
        done
    done
    return 1
}

if [[ -n "${PROJECT_DIR:-}" ]]; then
    PROJECT_DIR="$(cd -- "$PROJECT_DIR" && pwd)"
elif ! PROJECT_DIR="$(find_project_dir)"; then
    echo "ERROR: could not locate the EEGProc repository. Submit from inside the repository or export PROJECT_DIR."
    exit 2
fi
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
MODELS_JSON="${MODELS_JSON:-}"
MODEL_DIR="${MODEL_DIR:-}"
TRIALS_NPZ="${TRIALS_NPZ:-}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
TASK="${TASK:-arousal}"
TYPICALITY_SEQUENCE="${TYPICALITY_SEQUENCE:-vc_window_embeddings}"
TYPICALITY_WEIGHT="${TYPICALITY_WEIGHT:-}"
TYPICALITY_QUANTILE="${TYPICALITY_QUANTILE:-0.95}"
VARIANCE_FLOOR="${VARIANCE_FLOOR:-1e-6}"
TARGET_PROBABILITY="${TARGET_PROBABILITY:-0.80}"
TARGET_LOSS_COMPONENT="${TARGET_LOSS_COMPONENT:-confidence}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
LEARNING_RATE_DECAY="${LEARNING_RATE_DECAY:-1.0}"
MAX_STEPS="${MAX_STEPS:-200}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
MIN_GRADIENT_NORM="${MIN_GRADIENT_NORM:-1e-6}"
LOW_GRADIENT_PATIENCE="${LOW_GRADIENT_PATIENCE:-5}"
TYPICALITY_IMPROVEMENT_PATIENCE="${TYPICALITY_IMPROVEMENT_PATIENCE:-10}"
TYPICALITY_MIN_DELTA="${TYPICALITY_MIN_DELTA:-1e-6}"
PHYSIOLOGICAL_TOLERANCE="${PHYSIOLOGICAL_TOLERANCE:-1e-8}"
TARGET_WEIGHT="${TARGET_WEIGHT:-1.0}"
LATENT_WEIGHT="${LATENT_WEIGHT:-0.1}"
DECODED_WEIGHT="${DECODED_WEIGHT:-0.1}"
PHYSIOLOGICAL_WEIGHT="${PHYSIOLOGICAL_WEIGHT:-1.0}"
PHYSIOLOGY_QUANTILE="${PHYSIOLOGY_QUANTILE:-0.95}"
PHYSIOLOGY_REQUIRED_FRACTION="${PHYSIOLOGY_REQUIRED_FRACTION:-0.95}"
DECODER_MODE="${DECODER_MODE:-joint}"
FS="${FS:-128}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/runs/typicality/${TASK}_full_${RUN_ID}}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: Python environment not found at $VENV_DIR."
    exit 2
fi
if [[ -z "$MODELS_JSON" || ! -s "$MODELS_JSON" ]]; then
    echo "ERROR: set MODELS_JSON to the complete LOSO manifest."
    exit 2
fi
if [[ ! "$TASK" =~ ^(valence|arousal)$ ]]; then
    echo "ERROR: TASK must be valence or arousal."
    exit 2
fi
if [[ -z "$TYPICALITY_WEIGHT" ]]; then
    echo "ERROR: set TYPICALITY_WEIGHT to the frozen source-only tuned weight."
    exit 2
fi
case "$STOP_ON_SUCCESS" in
    1|true|TRUE) STOP_ARGUMENTS=(--stop-on-success) ;;
    0|false|FALSE) STOP_ARGUMENTS=(--no-stop-on-success) ;;
    *)
        echo "ERROR: STOP_ON_SUCCESS must be 1/0 or true/false."
        exit 2
        ;;
esac
if [[ -n "$MODEL_DIR" && ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: MODEL_DIR does not exist: $MODEL_DIR"
    exit 2
fi
if [[ -n "$TRIALS_NPZ" && ! -s "$TRIALS_NPZ" ]]; then
    echo "ERROR: TRIALS_NPZ does not exist: $TRIALS_NPZ"
    exit 2
fi
if [[ -z "$TRIALS_NPZ" ]]; then
    for required_file in "$EEG_PATH" "$LABELS_PATH"; do
        if [[ ! -s "$required_file" ]]; then
            echo "ERROR: required raw-data file not found: $required_file"
            exit 2
        fi
    done
fi

SUBJECT_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$SUBJECT_ID" || ! "$SUBJECT_ID" =~ ^[0-9]+$ || "$SUBJECT_ID" -gt 22 ]]; then
    echo "ERROR: this script requires a Slurm array index in 0-22."
    exit 2
fi

cd "$PROJECT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Keras executes an XLA-compiled tf.sign operation while rebuilding the saved
# recurrent classifier. Longleaf's CUDA module is not rooted at
# /usr/local/cuda, so XLA must be pointed at the toolkit that owns libdevice.
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
    [[ -d "$VENV_DIR/lib/python3.12/site-packages/nvidia" ]] && roots+=("$VENV_DIR/lib/python3.12/site-packages/nvidia")
    [[ -d /usr/local/cuda ]] && roots+=("/usr/local/cuda")
    [[ -d /opt/cuda ]] && roots+=("/opt/cuda")
    [[ ${#roots[@]} -eq 0 ]] && return 0
    find "${roots[@]}" -type f -path "*/nvvm/libdevice/libdevice.10.bc" -print -quit 2>/dev/null || true
}

LIBDEVICE_PATH="$(find_libdevice)"
if [[ -z "$LIBDEVICE_PATH" || ! -f "$LIBDEVICE_PATH" ]]; then
    echo "ERROR: unable to locate CUDA libdevice.10.bc after loading cuda/12.9." >&2
    echo "EBROOTCUDA=${EBROOTCUDA:-unset} CUDA_HOME=${CUDA_HOME:-unset} nvcc=$(command -v nvcc || echo unset)" >&2
    exit 2
fi

CUDA_XLA_ROOT="${LIBDEVICE_PATH%/nvvm/libdevice/libdevice.10.bc}"
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_cuda_data_dir=${CUDA_XLA_ROOT}"
if [[ -n "$MODULE_CUDA_ROOT" ]]; then
    export CUDA_HOME="$MODULE_CUDA_ROOT"
    export CUDA_PATH="$MODULE_CUDA_ROOT"
fi

echo "CUDA XLA root: $CUDA_XLA_ROOT"
echo "CUDA libdevice: $LIBDEVICE_PATH"

# Exercise the exact GPU operation that failed during Keras deserialization,
# before the runner writes its large input archive.
"$VENV_DIR/bin/python" - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) != 1:
    raise RuntimeError(f"Expected exactly one visible GPU; TensorFlow sees {gpus}")
with tf.device("/GPU:0"):
    actual = tf.sign(tf.constant([-1.0, 0.0, 1.0])).numpy().tolist()
if actual != [-1.0, 0.0, 1.0]:
    raise RuntimeError(f"Unexpected tf.sign result on GPU 0: {actual}")
print("TensorFlow GPU libdevice preflight passed", flush=True)
PY

HELP="$("$VENV_DIR/bin/python" -m eegproc.model_explainability.typicality.runner --help)"
for required_option in --models-json --task --typicality-sequence --typicality-weight --subjects --out-dir; do
    if [[ "$HELP" != *"$required_option"* ]]; then
        echo "ERROR: typicality.runner lacks $required_option."
        exit 3
    fi
done

if [[ -n "$MODEL_DIR" ]]; then
    MODEL_ARGUMENTS=(--models-json "$MODELS_JSON" --model-dir "$MODEL_DIR")
else
    MODEL_ARGUMENTS=(--models-json "$MODELS_JSON")
fi
if [[ -n "$TRIALS_NPZ" ]]; then
    DATA_ARGUMENTS=(--trials-npz "$TRIALS_NPZ")
else
    DATA_CONFIG="${DATA_CONFIG:-}"
    if [[ -z "$DATA_CONFIG" ]]; then
        DATA_CONFIG="$(printf '{\"raw_eeg_npy\":\"%s\",\"raw_labels_npy\":\"%s\",\"dataset\":\"dreamer\",\"label_dimension\":\"%s\",\"fs\":%s,\"window_sec\":1,\"window_overlap\":0,\"window_normalization\":\"global_rms\",\"label_threshold_mode\":\"global\",\"median_label\":3}' "$EEG_PATH" "$LABELS_PATH" "$TASK" "$FS")"
    fi
    DATA_ARGUMENTS=(
        --data-loader eegproc.model_explainability.model_agnostic.sic_adapter:load_sic_raw_trials
        --data-config "$DATA_CONFIG"
    )
fi

OUT_DIR="$OUT_ROOT/subject_${SUBJECT_ID}"
if [[ -e "$OUT_DIR" && "${RESUME:-0}" != "1" ]]; then
    echo "ERROR: refusing to overwrite existing output: $OUT_DIR"
    exit 4
fi
RESUME_ARGUMENTS=()
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_ARGUMENTS+=(--resume)
fi

echo "Typicality CFX: task=$TASK subject=$SUBJECT_ID"
echo "Manifest: $MODELS_JSON"
echo "Sequence: $TYPICALITY_SEQUENCE | weight: $TYPICALITY_WEIGHT"
echo "Output: $OUT_DIR"

"$VENV_DIR/bin/python" -m eegproc.model_explainability.typicality.runner \
    "${MODEL_ARGUMENTS[@]}" \
    "${DATA_ARGUMENTS[@]}" \
    --model-module eegproc.deep_learning.joint_architectures.SICModelv15.sic_model \
    --task "$TASK" \
    --subjects "$SUBJECT_ID" \
    --typicality-sequence "$TYPICALITY_SEQUENCE" \
    --typicality-weight "$TYPICALITY_WEIGHT" \
    --typicality-quantile "$TYPICALITY_QUANTILE" \
    --variance-floor "$VARIANCE_FLOOR" \
    --decoder-mode "$DECODER_MODE" \
    --target-probability "$TARGET_PROBABILITY" \
    --target-loss-component "$TARGET_LOSS_COMPONENT" \
    --target-weight "$TARGET_WEIGHT" \
    --latent-weight "$LATENT_WEIGHT" \
    --decoded-weight "$DECODED_WEIGHT" \
    --physiological-weight "$PHYSIOLOGICAL_WEIGHT" \
    --learning-rate "$LEARNING_RATE" \
    --learning-rate-decay "$LEARNING_RATE_DECAY" \
    --max-steps "$MAX_STEPS" \
    "${STOP_ARGUMENTS[@]}" \
    --min-gradient-norm "$MIN_GRADIENT_NORM" \
    --low-gradient-patience "$LOW_GRADIENT_PATIENCE" \
    --typicality-improvement-patience "$TYPICALITY_IMPROVEMENT_PATIENCE" \
    --typicality-min-delta "$TYPICALITY_MIN_DELTA" \
    --physiological-tolerance "$PHYSIOLOGICAL_TOLERANCE" \
    --physiology-quantile "$PHYSIOLOGY_QUANTILE" \
    --physiology-required-fraction "$PHYSIOLOGY_REQUIRED_FRACTION" \
    --fs "$FS" \
    --seed "$SEED" \
    --log-every "${LOG_EVERY:-10}" \
    "${RESUME_ARGUMENTS[@]}" \
    --out-dir "$OUT_DIR"

if [[ ! -s "$OUT_DIR/study.json" || ! -s "$OUT_DIR/subject_${SUBJECT_ID}/fold.json" ]]; then
    echo "ERROR: typicality runner did not produce a complete fold manifest."
    exit 4
fi
echo "Completed typicality-aware CFX for $TASK subject $SUBJECT_ID."
