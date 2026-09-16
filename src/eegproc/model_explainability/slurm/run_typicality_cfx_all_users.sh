#!/bin/bash
#SBATCH --job-name=typicality_cfx
#SBATCH --output=typicality_cfx_%A_%a.out
#SBATCH --error=typicality_cfx_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-22%8

set -euo pipefail

# One array task is one DREAMER held-out subject (0--22). The typicality
# runner owns source-only calibration and paired base/typicality objectives.
# Required exports: MODELS_JSON and TYPICALITY_WEIGHT.

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
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
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-10}"
TARGET_WEIGHT="${TARGET_WEIGHT:-1.0}"
LATENT_WEIGHT="${LATENT_WEIGHT:-0.1}"
DECODED_WEIGHT="${DECODED_WEIGHT:-0.1}"
PHYSIOLOGICAL_WEIGHT="${PHYSIOLOGICAL_WEIGHT:-0.0}"
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
    --snapshot-every "$SNAPSHOT_EVERY" \
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
