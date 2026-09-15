#!/bin/bash
#SBATCH --job-name=cfo_refs
#SBATCH --output=cfo_refs_%A_%a.out
#SBATCH --error=cfo_refs_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-45%8

set -euo pipefail

# Array indices 0-22 are valence subjects 0-22; 23-45 are arousal subjects
# 0-22. Each task estimates one source-only class-1 discrepancy threshold and
# writes held-out real discrepancies for the later subject-invariance audit.

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
TYPICALITY_QUANTILE="${TYPICALITY_QUANTILE:-0.95}"
SEED="${SEED:-42}"

if [[ -z "$CFO_EXPERIMENT_ID" ]]; then
    echo "ERROR: CFO_EXPERIMENT_ID is required and must be shared by both stages."
    exit 2
fi
if [[ ! "$CFO_EXPERIMENT_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: CFO_EXPERIMENT_ID may contain only letters, digits, dot, underscore, and dash."
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
OUTPUT_DIR="$RUN_ROOT/references/$TASK/fold_$(printf '%02d' "$SUBJECT_ID")"
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "ERROR: refusing to overwrite existing reference output: $OUTPUT_DIR"
    exit 2
fi

cd "$PROJECT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TF_GPU_ALLOCATOR=cuda_malloc_async

# This module is an explicit readiness gate documented in
# COUNTERFACTUAL_PAPER_EVALUATION.md. It is not present in the repository yet.
if ! "$VENV_DIR/bin/python" -m \
    eegproc.model_explainability.fit_class_typicality --help >/dev/null 2>&1; then
    echo "ERROR: fit_class_typicality is not implemented. Complete the runbook readiness gate first."
    exit 3
fi

mkdir -p "$RUN_ROOT/references/$TASK"

echo "Task: $TASK"
echo "Held-out subject: $SUBJECT_ID"
echo "Model: $MODEL_PATH"
echo "Source-only threshold quantile: $TYPICALITY_QUANTILE"
echo "Output: $OUTPUT_DIR"

"$VENV_DIR/bin/python" -m eegproc.model_explainability.fit_class_typicality \
    --model "$MODEL_PATH" \
    --model-manifest "$MANIFEST_PATH" \
    --model-module eegproc.deep_learning.joint_architectures.SICModelv15.sic_model \
    --raw-eeg-npy "$EEG_PATH" \
    --raw-labels-npy "$LABELS_PATH" \
    --dataset dreamer \
    --label-dimension "$TASK" \
    --fs 128 \
    --window-sec 1 \
    --window-overlap 0 \
    --window-normalization global_rms \
    --label-threshold-mode global \
    --median-label 3 \
    --held-out-subject "$SUBJECT_ID" \
    --target-class 1 \
    --threshold-quantile "$TYPICALITY_QUANTILE" \
    --discrepancy diagonal-gaussian-standardized-mse \
    --include-all-true-source-targets \
    --write-heldout-audit \
    --seed "$SEED" \
    --out-dir "$OUTPUT_DIR"

for expected_file in \
    class_1_reference.json \
    class_1_reference.npz \
    source_class_1_discrepancies.csv \
    heldout_real_discrepancies.csv; do
    if [[ ! -s "$OUTPUT_DIR/$expected_file" ]]; then
        echo "ERROR: expected nonempty output is missing: $OUTPUT_DIR/$expected_file"
        exit 4
    fi
done

echo "Completed $TASK reference for held-out subject $SUBJECT_ID."
