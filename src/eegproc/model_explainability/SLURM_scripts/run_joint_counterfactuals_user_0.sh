#!/bin/bash
#SBATCH --job-name=xai_cf_u0
#SBATCH --output=xai_cf_u0_%A_%a.out
#SBATCH --error=xai_cf_u0_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-17

set -euo pipefail

# One Slurm array task processes one DREAMER trial for subject 0. The shared
# array job therefore creates exactly 18 tasks, with trial IDs 0 through 17.
# MODEL_PATH must point to subject 0's trained SICModelv15 LOSO checkpoint.

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="${PROJECT_DIR:-$HOME/EEGProc}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
MODEL_PATH="${MODEL_PATH:-}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
TRIALS_NPZ="${TRIALS_NPZ:-}"
TARGET_DIMENSION="${TARGET_DIMENSION:-valence}"
WINDOW_NORMALIZATION="${WINDOW_NORMALIZATION:-global_rms}"
TARGET_PROBABILITY="${TARGET_PROBABILITY:-0.8}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
MAX_STEPS="${MAX_STEPS:-200}"
TARGET_WEIGHT="${TARGET_WEIGHT:-1.0}"
LATENT_WEIGHT="${LATENT_WEIGHT:-0.1}"
DECODED_WEIGHT="${DECODED_WEIGHT:-0.1}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-5.0}"
LOG_EVERY="${LOG_EVERY:-10}"
SEED="${SEED:-42}"
SUBJECT_ID=0
EXPECTED_TRIALS=18
TRIAL_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-manual}}"
XAI_ROOT="${XAI_ROOT:-$PROJECT_DIR/XAI_runs}"
RUN_ROOT="$XAI_ROOT/subject_${SUBJECT_ID}_joint_counterfactuals_${RUN_ID}"
TASK_ROOT="$RUN_ROOT/trial_$(printf '%02d' "$TRIAL_ID")"

if (( TRIAL_ID < 0 || TRIAL_ID >= EXPECTED_TRIALS )); then
    echo "ERROR: array task $TRIAL_ID is outside trial range 0-17."
    exit 2
fi
if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: set MODEL_PATH to subject 0's trained SICModelv15 .keras checkpoint."
    exit 2
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: Python environment not found at $VENV_DIR."
    exit 2
fi
if [[ -e "$TASK_ROOT" ]]; then
    echo "ERROR: refusing to overwrite existing task output: $TASK_ROOT"
    exit 2
fi

mkdir -p "$RUN_ROOT"

cd "$PROJECT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$RUN_ROOT/.matplotlib/task_${TRIAL_ID}"
export XDG_CACHE_HOME="$RUN_ROOT/.cache/task_${TRIAL_ID}"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

aggregate_metrics() {
    "$VENV_DIR/bin/python" -m \
        eegproc.model_explainability.aggregate_counterfactual_metrics \
        "$RUN_ROOT" \
        --subject-id "$SUBJECT_ID" \
        --expected-trials "$EXPECTED_TRIALS"
}

finalize() {
    local task_status=$?
    local metrics_status=0
    trap - EXIT
    set +e
    aggregate_metrics
    metrics_status=$?
    set -e
    if (( task_status == 0 && metrics_status != 0 )); then
        task_status=$metrics_status
    fi
    exit "$task_status"
}
trap finalize EXIT

if [[ -n "$TRIALS_NPZ" ]]; then
    if [[ ! -f "$TRIALS_NPZ" ]]; then
        echo "ERROR: prepared trial file not found: $TRIALS_NPZ"
        exit 2
    fi
    DATA_ARGUMENTS=(--trials-npz "$TRIALS_NPZ")
else
    if [[ ! -f "$EEG_PATH" || ! -f "$LABELS_PATH" ]]; then
        echo "ERROR: raw DREAMER arrays were not found."
        exit 2
    fi
    DATA_ARGUMENTS=(
        --raw-eeg-npy "$EEG_PATH"
        --raw-labels-npy "$LABELS_PATH"
        --dataset dreamer
        --label-dimension "$TARGET_DIMENSION"
        --fs 128
        --window-sec 1
        --window-overlap 0
        --window-normalization "$WINDOW_NORMALIZATION"
        --label-threshold-mode global
        --median-label 3
    )
fi

echo "Array job: ${SLURM_ARRAY_JOB_ID:-manual}"
echo "Array task/trial: $TRIAL_ID"
echo "Subject: $SUBJECT_ID"
echo "Model: $MODEL_PATH"
echo "Output: $TASK_ROOT"

"$VENV_DIR/bin/python" -m eegproc.model_explainability.run_counterfactuals \
    --model "$MODEL_PATH" \
    --model-module eegproc.deep_learning.joint_architectures.SICModelv15.sic_model \
    "${DATA_ARGUMENTS[@]}" \
    --subject-id "$SUBJECT_ID" \
    --trial-id "$TRIAL_ID" \
    --decoder-mode joint \
    --target-probability "$TARGET_PROBABILITY" \
    --learning-rate "$LEARNING_RATE" \
    --max-steps "$MAX_STEPS" \
    --gradient-clip-norm "$GRADIENT_CLIP_NORM" \
    --target-weight "$TARGET_WEIGHT" \
    --latent-weight "$LATENT_WEIGHT" \
    --decoded-weight "$DECODED_WEIGHT" \
    --log-every "$LOG_EVERY" \
    --seed "$SEED" \
    --out-dir "$TASK_ROOT"

TRIAL_DIRECTORY="$TASK_ROOT/subject_${SUBJECT_ID}_trial_${TRIAL_ID}"
COUNTERFACTUAL_NPZ="$TRIAL_DIRECTORY/counterfactual.npz"
HISTORY_CSV="$TRIAL_DIRECTORY/history.csv"

"$VENV_DIR/bin/python" -m eegproc.model_explainability.counterfactual_heatmap \
    "$COUNTERFACTUAL_NPZ" \
    --branch joint \
    --sampling-rate 128 \
    --no-show

"$VENV_DIR/bin/python" -m eegproc.model_explainability.counterfactual_topography \
    "$COUNTERFACTUAL_NPZ" \
    --branch joint \
    --no-show

"$VENV_DIR/bin/python" -m \
    eegproc.model_explainability.counterfactual_training_monitor \
    "$HISTORY_CSV" \
    --no-show

echo "Completed subject $SUBJECT_ID trial $TRIAL_ID."
