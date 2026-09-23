#!/bin/bash
#SBATCH --job-name=typicality_cfx_smoke
#SBATCH --output=typicality_outs/typicality_cfx_smoke_%j.out
#SBATCH --error=typicality_outs/typicality_cfx_smoke_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

# One-user/one-trial smoke run. The runner retains the complete dataset for
# source-only typicality calibration, but --trial-ids restricts optimization
# to the selected held-out trial. Defaults target subject 0, trial 8, which is
# a true-class-0/predicted-class-0 trial for the saved full arousal checkpoint.
#
# The defaults use the 23-fold arousal scale-64 run. Optional overrides:
# MODELS_JSON, MODEL_DIR, TRIALS_NPZ, TASK, SMOKE_SUBJECT, SMOKE_TRIAL.
#
# Example:
#   sbatch src/eegproc/model_explainability/slurm/smoke_typicality_cfx_arousal_user0_trial8.sh

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
SAVED_CONFIG_DIR="$PROJECT_DIR/runs/full/sic_v15_arousal_scale64/DREAMER/arousal/suite_837240/full/full_run_v15_arousal_scale64_20260912_225458/configuration_0001"
MODELS_JSON="${MODELS_JSON:-$SAVED_CONFIG_DIR/loso_zero_shot_models.json}"
MODEL_DIR="${MODEL_DIR:-$(dirname "$MODELS_JSON")/loso_zero_shot_models}"
TRIALS_NPZ="${TRIALS_NPZ:-}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
TASK="${TASK:-arousal}"
SMOKE_SUBJECT="${SMOKE_SUBJECT:-0}"
SMOKE_TRIAL="${SMOKE_TRIAL:-8}"
TYPICALITY_REPRESENTATION="${TYPICALITY_REPRESENTATION:-${TYPICALITY_SEQUENCE:-vc_trial_embedding}}"
TYPICALITY_WEIGHT="${TYPICALITY_WEIGHT:-1.0}"
TYPICALITY_QUANTILE="${TYPICALITY_QUANTILE:-0.95}"
VARIANCE_FLOOR="${VARIANCE_FLOOR:-1e-6}"
TARGET_PROBABILITY="${TARGET_PROBABILITY:-0.80}"
TARGET_LOSS_COMPONENT="${TARGET_LOSS_COMPONENT:-confidence}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
LEARNING_RATE_DECAY="${LEARNING_RATE_DECAY:-1.0}"
MAX_STEPS="${MAX_STEPS:-5}"
TARGET_WEIGHT="${TARGET_WEIGHT:-1.0}"
LATENT_WEIGHT="${LATENT_WEIGHT:-0.1}"
DECODED_WEIGHT="${DECODED_WEIGHT:-0.1}"
PHYSIOLOGICAL_WEIGHT="${PHYSIOLOGICAL_WEIGHT:-0.0}"
PHYSIOLOGY_QUANTILE="${PHYSIOLOGY_QUANTILE:-0.95}"
PHYSIOLOGY_REQUIRED_FRACTION="${PHYSIOLOGY_REQUIRED_FRACTION:-0.95}"
DECODER_MODE="${DECODER_MODE:-joint}"
FS="${FS:-128}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/runs/typicality/smoke_${TASK}_${RUN_ID}}"

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
if [[ ! "$SMOKE_SUBJECT" =~ ^[0-9]+$ || ! "$SMOKE_TRIAL" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SMOKE_SUBJECT and SMOKE_TRIAL must be nonnegative integers."
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
if [[ -n "$MODEL_DIR" && ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: MODEL_DIR does not exist: $MODEL_DIR"
    exit 2
fi
if [[ -e "$OUT_DIR" && "${RESUME:-0}" != "1" ]]; then
    echo "ERROR: refusing to overwrite existing output: $OUT_DIR"
    exit 4
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
# before the runner starts optimization.
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
for required_option in --models-json --task --typicality-representation --typicality-weight --subjects --trial-ids --out-dir; do
    if [[ "$HELP" != *"$required_option"* ]]; then
        echo "ERROR: typicality.runner lacks $required_option."
        exit 3
    fi
done

MODEL_ARGUMENTS=(--models-json "$MODELS_JSON")
if [[ -n "$MODEL_DIR" ]]; then
    MODEL_ARGUMENTS+=(--model-dir "$MODEL_DIR")
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
RESUME_ARGUMENTS=()
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_ARGUMENTS+=(--resume)
fi

echo "Typicality smoke: task=$TASK subject=$SMOKE_SUBJECT trial=$SMOKE_TRIAL"
echo "Manifest: $MODELS_JSON"
echo "Representation: $TYPICALITY_REPRESENTATION | weight: $TYPICALITY_WEIGHT | steps: $MAX_STEPS"
echo "Output: $OUT_DIR"

"$VENV_DIR/bin/python" -m eegproc.model_explainability.typicality.runner \
    "${MODEL_ARGUMENTS[@]}" \
    "${DATA_ARGUMENTS[@]}" \
    --model-module eegproc.deep_learning.joint_architectures.SICModelv15.sic_model \
    --task "$TASK" \
    --subjects "$SMOKE_SUBJECT" \
    --trial-ids "$SMOKE_TRIAL" \
    --typicality-representation "$TYPICALITY_REPRESENTATION" \
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
    --physiology-quantile "$PHYSIOLOGY_QUANTILE" \
    --physiology-required-fraction "$PHYSIOLOGY_REQUIRED_FRACTION" \
    --fs "$FS" \
    --seed "$SEED" \
    --log-every "${LOG_EVERY:-1}" \
    "${RESUME_ARGUMENTS[@]}" \
    --out-dir "$OUT_DIR"

if [[ ! -s "$OUT_DIR/study.json" || ! -s "$OUT_DIR/subject_${SMOKE_SUBJECT}/fold.json" ]]; then
    echo "ERROR: typicality runner did not produce the expected smoke artifacts."
    exit 4
fi

"$VENV_DIR/bin/python" - "$OUT_DIR" "$SMOKE_SUBJECT" "$SMOKE_TRIAL" <<'PY'
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
subject = int(sys.argv[2])
trial = int(sys.argv[3])
fold = json.loads((out / f"subject_{subject}" / "fold.json").read_text())

if fold.get("status") != "completed":
    raise SystemExit("ERROR: smoke fold did not finish")
if trial not in fold.get("eligible_trial_ids", []):
    raise SystemExit(
        f"ERROR: subject {subject} trial {trial} was not eligible; "
        "no counterfactual smoke test was performed"
    )
if fold.get("n_optimization_errors"):
    raise SystemExit(
        f"ERROR: smoke run recorded {fold['n_optimization_errors']} optimization errors"
    )

trial_dir = out / f"subject_{subject}" / f"trial_{trial}"
for objective in ("base", "typicality"):
    completed = list((trial_dir / objective).glob("attempt_*/complete.json"))
    if not completed:
        raise SystemExit(f"ERROR: {objective} objective did not complete")
PY

echo "Smoke completed successfully: $OUT_DIR"
