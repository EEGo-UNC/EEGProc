#!/bin/bash
#SBATCH --job-name=cfo_population
#SBATCH --output=cfo_population_%A_%a.out
#SBATCH --error=cfo_population_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-45%8

set -euo pipefail

# Array indices 0-22 are valence subjects 0-22; 23-45 are arousal subjects
# 0-22. A task runs the base and constrained objectives sequentially so the
# pair shares its checkpoint, reference, eligible trials, GPU, and environment.

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

if [[ -z "${PROJECT_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]]; then
        PROJECT_DIR="$SLURM_SUBMIT_DIR"
    elif [[ -f "$HOME/EEGProc/pyproject.toml" ]]; then
        PROJECT_DIR="$HOME/EEGProc"
    else
        echo "ERROR: submit from the EEGProc root or set PROJECT_DIR."
        exit 2
    fi
fi
PROJECT_DIR="${PROJECT_DIR%/}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
CFO_EXPERIMENT_ID="${CFO_EXPERIMENT_ID:-}"
VALENCE_CONFIG_DIR="${VALENCE_CONFIG_DIR:-}"
AROUSAL_CONFIG_DIR="${AROUSAL_CONFIG_DIR:-}"

TARGET_PROBABILITY="${TARGET_PROBABILITY:-0.60}"
LEARNING_RATE="${LEARNING_RATE:-1.0}"
LEARNING_RATE_DECAY="${LEARNING_RATE_DECAY:-0.95}"
MAX_STEPS="${MAX_STEPS:-200}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-5.0}"
TARGET_WEIGHT="${TARGET_WEIGHT:-1.0}"
LATENT_WEIGHT="${LATENT_WEIGHT:-0.1}"
DECODED_WEIGHT="${DECODED_WEIGHT:-0.1}"
PHYSIOLOGICAL_WEIGHT="${PHYSIOLOGICAL_WEIGHT:-0.0}"
TYPICALITY_WEIGHT="${TYPICALITY_WEIGHT:-}"
LOG_EVERY="${LOG_EVERY:-10}"
SEED="${SEED:-42}"

if [[ -z "$CFO_EXPERIMENT_ID" ]]; then
    echo "ERROR: CFO_EXPERIMENT_ID is required and must match the reference job."
    exit 2
fi
if [[ ! "$CFO_EXPERIMENT_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: CFO_EXPERIMENT_ID may contain only letters, digits, dot, underscore, and dash."
    exit 2
fi
if [[ -z "$TYPICALITY_WEIGHT" ]]; then
    echo "ERROR: set TYPICALITY_WEIGHT to the source-only tuned constrained-objective weight."
    exit 2
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: Python environment not found at $VENV_DIR."
    exit 2
fi
for required_file in "$EEG_PATH" "$LABELS_PATH"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required data file not found: $required_file"
        exit 2
    fi
done

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$ARRAY_INDEX" || ! "$ARRAY_INDEX" =~ ^[0-9]+$ || "$ARRAY_INDEX" -gt 45 ]]; then
    echo "ERROR: this script requires a Slurm array index in 0-45."
    exit 2
fi
if (( ARRAY_INDEX < 23 )); then
    TASK=valence
    SUBJECT_ID="$ARRAY_INDEX"
    CONFIG_DIR="$VALENCE_CONFIG_DIR"
else
    TASK=arousal
    SUBJECT_ID="$((ARRAY_INDEX - 23))"
    CONFIG_DIR="$AROUSAL_CONFIG_DIR"
fi

if [[ -z "$CONFIG_DIR" || ! -d "$CONFIG_DIR" ]]; then
    echo "ERROR: configuration directory for $TASK is missing: $CONFIG_DIR"
    exit 2
fi
MANIFEST_PATH="$CONFIG_DIR/loso_zero_shot_models.json"
FOLD_NUMBER="$((SUBJECT_ID + 1))"
MODEL_PATH="$CONFIG_DIR/loso_zero_shot_models/loso_fold_$(printf '%04d' "$FOLD_NUMBER")_target_${SUBJECT_ID}_zero_shot.keras"
for required_file in "$MANIFEST_PATH" "$MODEL_PATH"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required model artifact not found: $required_file"
        exit 2
    fi
done

RUN_ROOT="$PROJECT_DIR/runs/paper_counterfactuals/$CFO_EXPERIMENT_ID"
REFERENCE_DIR="$RUN_ROOT/references/$TASK/fold_$(printf '%02d' "$SUBJECT_ID")"
REFERENCE_JSON="$REFERENCE_DIR/class_1_reference.json"
REFERENCE_NPZ="$REFERENCE_DIR/class_1_reference.npz"
for required_file in "$REFERENCE_JSON" "$REFERENCE_NPZ"; do
    if [[ ! -s "$required_file" ]]; then
        echo "ERROR: reference artifact not found; did the dependency job finish? $required_file"
        exit 2
    fi
done

FOLD_ROOT="$RUN_ROOT/counterfactuals/$TASK/fold_$(printf '%02d' "$SUBJECT_ID")"
ELIGIBLE_MANIFEST="$FOLD_ROOT/eligible_trials.csv"
BASE_DIR="$FOLD_ROOT/base"
TYPICALITY_DIR="$FOLD_ROOT/typicality"
if [[ -e "$FOLD_ROOT" ]]; then
    echo "ERROR: refusing to overwrite existing fold output: $FOLD_ROOT"
    exit 2
fi

cd "$PROJECT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Verify the extended paper-evaluation interface rather than silently running
# the current base-only CLI with scientific semantics it does not implement.
RUNNER_HELP="$("$VENV_DIR/bin/python" -m eegproc.model_explainability.run_counterfactuals --help)"
for required_option in \
    --model-manifest \
    --typicality-reference-json \
    --typicality-reference-npz \
    --typicality-weight \
    --eligible-true-class \
    --require-correct-original \
    --eligible-manifest \
    --write-eligible-manifest \
    --read-eligible-manifest \
    --include-target-probability-failures; do
    if [[ "$RUNNER_HELP" != *"$required_option"* ]]; then
        echo "ERROR: run_counterfactuals lacks $required_option. Complete the runbook readiness gate first."
        exit 3
    fi
done

mkdir -p "$FOLD_ROOT"

COMMON_ARGUMENTS=(
    --model "$MODEL_PATH"
    --model-manifest "$MANIFEST_PATH"
    --model-module eegproc.deep_learning.joint_architectures.SICModelv15.sic_model
    --raw-eeg-npy "$EEG_PATH"
    --raw-labels-npy "$LABELS_PATH"
    --dataset dreamer
    --label-dimension "$TASK"
    --fs 128
    --window-sec 1
    --window-overlap 0
    --window-normalization global_rms
    --label-threshold-mode global
    --median-label 3
    --subject-id "$SUBJECT_ID"
    --target-class 1
    --eligible-true-class 0
    --require-correct-original
    --decoder-mode joint
    --target-probability "$TARGET_PROBABILITY"
    --learning-rate "$LEARNING_RATE"
    --learning-rate-decay "$LEARNING_RATE_DECAY"
    --max-steps "$MAX_STEPS"
    --gradient-clip-norm "$GRADIENT_CLIP_NORM"
    --target-weight "$TARGET_WEIGHT"
    --latent-weight "$LATENT_WEIGHT"
    --decoded-weight "$DECODED_WEIGHT"
    --physiological-weight "$PHYSIOLOGICAL_WEIGHT"
    --typicality-reference-json "$REFERENCE_JSON"
    --typicality-reference-npz "$REFERENCE_NPZ"
    --eligible-manifest "$ELIGIBLE_MANIFEST"
    --include-target-probability-failures
    --log-every "$LOG_EVERY"
    --seed "$SEED"
)

echo "Task: $TASK"
echo "Held-out subject: $SUBJECT_ID"
echo "Model: $MODEL_PATH"
echo "Reference: $REFERENCE_JSON"
echo "Output: $FOLD_ROOT"
echo "Typicality weight: $TYPICALITY_WEIGHT"

# Base CFO still receives the reference so D_C1 and tau_C1 are evaluated and
# reported. A zero weight guarantees that they do not affect optimization.
"$VENV_DIR/bin/python" -m eegproc.model_explainability.run_counterfactuals \
    "${COMMON_ARGUMENTS[@]}" \
    --typicality-weight 0 \
    --write-eligible-manifest \
    --out-dir "$BASE_DIR"

if [[ ! -s "$ELIGIBLE_MANIFEST" ]]; then
    echo "ERROR: base CFO did not write a nonempty eligible-trial manifest."
    exit 4
fi

# The constrained run must consume, not regenerate, the base manifest. This
# makes all objective comparisons paired and keeps failed-p_target trials.
"$VENV_DIR/bin/python" -m eegproc.model_explainability.run_counterfactuals \
    "${COMMON_ARGUMENTS[@]}" \
    --typicality-weight "$TYPICALITY_WEIGHT" \
    --read-eligible-manifest \
    --out-dir "$TYPICALITY_DIR"

for objective_dir in "$BASE_DIR" "$TYPICALITY_DIR"; do
    if [[ ! -s "$objective_dir/settings.json" || ! -s "$objective_dir/results.json" || ! -s "$objective_dir/summary.json" ]]; then
        echo "ERROR: incomplete objective output: $objective_dir"
        exit 4
    fi
done

echo "Completed paired CFO objectives for $TASK subject $SUBJECT_ID."
