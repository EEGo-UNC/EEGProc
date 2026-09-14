#!/bin/bash
#SBATCH --job-name=vcsc_z0
#SBATCH --output=vcsc_z0_%A_%a.out
#SBATCH --error=vcsc_z0_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-15

set -euo pipefail

# ---------------------------------------------------------------------------
# VCSC Z0 sensitivity sweep, v15 arousal, joint decoder
# ---------------------------------------------------------------------------
# WHY. Across 20 counterfactuals on subjects 0-3, VCSC came back between
# 0.0000 and 0.0006 on every single one. That is not evidence the
# counterfactuals are perfectly plausible -- it means Z0=2.0 is too loose to
# discriminate at this scale, so the penalty sits at its floor and the term
# carries no information. VCSC only penalises a pair once its combined
# deviation exceeds Z0, so lowering Z0 is what gives the metric range.
#
# This sweeps Z0 over 2.0 / 1.5 / 1.0 / 0.5 for every subject, running all
# trials per subject. The useful output is the spread of vcsc_counterfactual
# within each Z0: the smallest Z0 that still separates trials is the one
# worth adopting.
#
# One array task = one (subject, Z0) pair. 4 subjects x 4 Z0 values = 16
# tasks, hence --array=0-15. Change the arrays below and the --array range
# together or the extra combinations silently never run.
#
# NOTE ON SUBJECTS. The temperature_64 smoke suite only contains LOSO folds
# 1-4, i.e. subjects 0-3. Running "all users" needs a full 23-fold suite;
# point MODEL_ROOT at it and set SUBJECTS_OVERRIDE="0 1 2 ... 22", and widen
# the --array range to match.
#
# Example:
#   sbatch run_vcsc_z0_sweep.sh
#   SUBJECTS_OVERRIDE="0 1" Z0_OVERRIDE="1.0 0.5" sbatch --array=0-3 run_vcsc_z0_sweep.sh
# ---------------------------------------------------------------------------

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="${PROJECT_DIR:-$HOME/EEGProc}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"

EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"

MODEL_ROOT="${MODEL_ROOT:-$PROJECT_DIR/runs/smoke/sic_v15_arousal_grid/DREAMER/arousal/suite_798030/temperature_64/users_0_3/smoke_arousal_0_3_grid_temperature_64_20260911_130508/configuration_0001/loso_zero_shot_models}"
MODEL_MODULE="${MODEL_MODULE:-eegproc.deep_learning.joint_architectures.SICModelv15.sic_model}"

# Sweep grid --------------------------------------------------------------
SUBJECTS=(${SUBJECTS_OVERRIDE:-0 1 2 3})
Z0_VALUES=(${Z0_OVERRIDE:-2.0 1.5 1.0 0.5})

# Optimization settings ---------------------------------------------------
TARGET_PROBABILITY="${TARGET_PROBABILITY:-0.6}"
LEARNING_RATE="${LEARNING_RATE:-0.5}"
MAX_STEPS="${MAX_STEPS:-200}"
TARGET_WEIGHT="${TARGET_WEIGHT:-1.0}"
LATENT_WEIGHT="${LATENT_WEIGHT:-0.1}"
DECODED_WEIGHT="${DECODED_WEIGHT:-0.1}"
PHYS_WEIGHT="${PHYS_WEIGHT:-0.1}"
DECODER_MODE="${DECODER_MODE:-joint}"

# VCSC knobs other than Z0 (Z_MAX must stay above Z0) ----------------------
VCSC_DISTANCE_CM="${VCSC_DISTANCE_CM:-12.0}"
VCSC_TAU_CM="${VCSC_TAU_CM:-4.0}"
VCSC_Z_MAX="${VCSC_Z_MAX:-20.0}"

# Preprocessing -- MUST match what the checkpoint was trained with ---------
# The shipped VCSC calibration is measured at n_windows=60. Changing FS or
# WINDOW_SEC changes the window count, which triggers a RuntimeWarning and
# produces biased z-scores until the calibration is regenerated.
FS="${FS:-128}"
WINDOW_SEC="${WINDOW_SEC:-1}"
WINDOW_OVERLAP="${WINDOW_OVERLAP:-0}"
WINDOW_NORMALIZATION="${WINDOW_NORMALIZATION:-global_rms}"
LABEL_DIMENSION="${LABEL_DIMENSION:-arousal}"
LABEL_THRESHOLD_MODE="${LABEL_THRESHOLD_MODE:-global}"
MEDIAN_LABEL="${MEDIAN_LABEL:-3}"

SUITE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/runs/counterfactuals/z0_sweep_${SUITE_ID}}"

# ---------------------------------------------------------------------------
# Decode this task's grid position
# ---------------------------------------------------------------------------
TASK="${SLURM_ARRAY_TASK_ID:-0}"
N_Z0=${#Z0_VALUES[@]}
TOTAL=$(( ${#SUBJECTS[@]} * N_Z0 ))

if (( TASK < 0 || TASK >= TOTAL )); then
    echo "ERROR: array task $TASK outside 0-$((TOTAL - 1))."
    echo "Grid is ${#SUBJECTS[@]} subjects x $N_Z0 Z0 values. Fix --array to match."
    exit 2
fi

SUBJECT=${SUBJECTS[$(( TASK / N_Z0 ))]}
VCSC_Z0=${Z0_VALUES[$(( TASK % N_Z0 ))]}

# LOSO checkpoint is derived, never hand-typed: fold NNNN holds out subject
# NNNN-1. A mismatched checkpoint trained on this subject would silently
# invalidate the zero-shot framing and nothing downstream can detect it.
MODEL_FILE=$(printf "loso_fold_%04d_target_%d_zero_shot.keras" $(( SUBJECT + 1 )) "$SUBJECT")
MODEL_PATH="$MODEL_ROOT/$MODEL_FILE"

for required in "$MODEL_PATH" "$EEG_PATH" "$LABELS_PATH"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: required file missing: $required"
        exit 2
    fi
done

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Shared venv (same bootstrap pattern as the training scripts)
# ---------------------------------------------------------------------------
if command -v flock >/dev/null 2>&1; then
    (
        flock -x 9
        if [[ ! -x "$VENV_DIR/bin/python" ]]; then
            python -m venv "$VENV_DIR"
            "$VENV_DIR/bin/python" -m pip install --upgrade pip
            "$VENV_DIR/bin/python" -m pip install -r requirements.txt
        elif [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
            "$VENV_DIR/bin/python" -m pip install -r requirements.txt
        fi
    ) 9>"$PROJECT_DIR/.venv312_install.lock"
else
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        python -m venv "$VENV_DIR"
        "$VENV_DIR/bin/python" -m pip install --upgrade pip
        "$VENV_DIR/bin/python" -m pip install -r requirements.txt
    fi
fi
source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

OUT_DIR="$OUT_ROOT/subject${SUBJECT}_z0_${VCSC_Z0}"

echo "=========================================================================="
echo "task $TASK/$((TOTAL - 1))  subject=$SUBJECT  vcsc_z0=$VCSC_Z0"
echo "checkpoint: $MODEL_FILE"
echo "decoder mode: $DECODER_MODE | label dimension: $LABEL_DIMENSION"
echo "out: $OUT_DIR"
echo "=========================================================================="

# --trial-id omitted on purpose: the runner defaults to every trial of the
# selected subject, in source order.
python -u -m eegproc.model_explainability.run_counterfactuals \
    --model "$MODEL_PATH" \
    --model-module "$MODEL_MODULE" \
    --decoder-mode "$DECODER_MODE" \
    --raw-eeg-npy "$EEG_PATH" \
    --raw-labels-npy "$LABELS_PATH" \
    --dataset dreamer \
    --label-dimension "$LABEL_DIMENSION" \
    --fs "$FS" \
    --window-sec "$WINDOW_SEC" \
    --window-overlap "$WINDOW_OVERLAP" \
    --window-normalization "$WINDOW_NORMALIZATION" \
    --label-threshold-mode "$LABEL_THRESHOLD_MODE" \
    --median-label "$MEDIAN_LABEL" \
    --subject-id "$SUBJECT" \
    --target-probability "$TARGET_PROBABILITY" \
    --learning-rate "$LEARNING_RATE" \
    --max-steps "$MAX_STEPS" \
    --target-weight "$TARGET_WEIGHT" \
    --latent-weight "$LATENT_WEIGHT" \
    --decoded-weight "$DECODED_WEIGHT" \
    --physiological-weight "$PHYS_WEIGHT" \
    --vcsc-distance-cm "$VCSC_DISTANCE_CM" \
    --vcsc-tau-cm "$VCSC_TAU_CM" \
    --vcsc-z0 "$VCSC_Z0" \
    --vcsc-z-max "$VCSC_Z_MAX" \
    --log-every 50 \
    --out-dir "$OUT_DIR"

echo "task $TASK complete: $OUT_DIR"
