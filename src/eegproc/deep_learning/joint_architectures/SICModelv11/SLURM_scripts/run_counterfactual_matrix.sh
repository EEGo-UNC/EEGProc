#!/bin/bash
#SBATCH --job-name=cf_matrix
#SBATCH --output=cf_matrix_%A_%a.out
#SBATCH --error=cf_matrix_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=0-53

set -euo pipefail

# ---------------------------------------------------------------------------
# Counterfactual experiment matrix (checklist item 5)
# ---------------------------------------------------------------------------
# One array task = one (subject, trial, learning_rate, target_probability)
# combination. Defaults give 3 subjects x 3 trials x 3 learning rates x
# 2 target probabilities = 54 tasks, hence --array=0-53 above. Change the
# arrays below and the --array range together, or extra combinations will
# silently never run.
#
# WHAT THIS MEASURES. Not classification success. This model is highly
# uncertain by design -- the maximum confidence across all 7866 predictions
# in the source training run is 0.5136 -- so any threshold above ~0.51 fails
# by construction and says nothing about the counterfactual. The quantity of
# interest is whether the GENERATED FEATURES are physiologically realistic,
# which is what VCSC measures. Target probabilities are set to 0.55/0.6:
# 0.55 is just reachable (0.5412 observed at lr=0.1 in 200 steps), 0.6 sits
# above anything the model produced in training, so it forces the search
# further out and gives VCSC something to push against.
#
# Learning rate is swept because at lr=0.1 the search stalls: the gradient
# collapsed to 1.8e-4 by step 90 and target_p froze, and VCSC fell to exactly
# 0 by step 70 (inside the plausibility envelope, where the penalty is flat
# and contributes no gradient). Larger steps should leave that envelope.
#
# Each run reports VCSC for the reconstructed original AND the counterfactual.
# The decoders reconstruct imperfectly, so the decoded original is already
# implausible (15.4 observed, against a real-trial median of 0.66); the delta
# is what the counterfactual search itself cost.
#
# Example:
#   sbatch run_counterfactual_matrix.sh                     # VCSC measured only
#   PHYS_WEIGHT=0.1 sbatch run_counterfactual_matrix.sh     # VCSC active
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

# Checkpoint suite. Change if you target a different training run.
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_DIR/runs/full/sic_trial_bigru_v11_mldg_brier_ablation/DREAMER/valence/suite_65452590/full/dreamer_valence_sic_trial_bigru_v11_mldg_full_full_20260827_213158/configuration_0001/loso_zero_shot_models}"

# Experiment grid ----------------------------------------------------------
SUBJECTS=(${SUBJECTS_OVERRIDE:-0 1 2})
TRIALS=(${TRIALS_OVERRIDE:-0 1 2})
TARGET_PROBS=(${TARGET_PROBS_OVERRIDE:-0.55 0.6})
LEARNING_RATES=(${LEARNING_RATES_OVERRIDE:-0.1 0.5 2.0})

# Optimization settings ----------------------------------------------------
MAX_STEPS="${MAX_STEPS:-200}"
TARGET_WEIGHT="${TARGET_WEIGHT:-1.0}"
LATENT_WEIGHT="${LATENT_WEIGHT:-0.1}"
DECODED_WEIGHT="${DECODED_WEIGHT:-0.1}"
PHYS_WEIGHT="${PHYS_WEIGHT:-0.0}"

# VCSC knobs (defaults match CounterfactualLoss) ----------------------------
VCSC_DISTANCE_CM="${VCSC_DISTANCE_CM:-12.0}"
VCSC_TAU_CM="${VCSC_TAU_CM:-4.0}"
VCSC_Z0="${VCSC_Z0:-2.0}"
VCSC_Z_MAX="${VCSC_Z_MAX:-20.0}"

# Preprocessing -- MUST match what the checkpoint was trained with ----------
FS="${FS:-128}"
WINDOW_SEC="${WINDOW_SEC:-1}"
WINDOW_OVERLAP="${WINDOW_OVERLAP:-0}"
WINDOW_NORMALIZATION="${WINDOW_NORMALIZATION:-global_rms}"
LABEL_DIMENSION="${LABEL_DIMENSION:-valence}"
LABEL_THRESHOLD_MODE="${LABEL_THRESHOLD_MODE:-global}"
MEDIAN_LABEL="${MEDIAN_LABEL:-3}"

SUITE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/runs/counterfactuals/matrix_${SUITE_ID}}"

# ---------------------------------------------------------------------------
# Decode this task's grid position
# ---------------------------------------------------------------------------
TASK="${SLURM_ARRAY_TASK_ID:-0}"
N_TRIALS=${#TRIALS[@]}
N_PROBS=${#TARGET_PROBS[@]}
N_LRS=${#LEARNING_RATES[@]}
PER_SUBJECT=$(( N_TRIALS * N_LRS * N_PROBS ))
TOTAL=$(( ${#SUBJECTS[@]} * PER_SUBJECT ))

if (( TASK < 0 || TASK >= TOTAL )); then
    echo "ERROR: array task $TASK outside 0-$((TOTAL - 1)). Grid is ${#SUBJECTS[@]} subjects x $N_TRIALS trials x $N_LRS learning rates x $N_PROBS probs."
    echo "Fix the --array range at the top of this script to match."
    exit 2
fi

SUBJECT=${SUBJECTS[$(( TASK / PER_SUBJECT ))]}
TRIAL=${TRIALS[$(( (TASK / (N_LRS * N_PROBS)) % N_TRIALS ))]}
LEARNING_RATE=${LEARNING_RATES[$(( (TASK / N_PROBS) % N_LRS ))]}
TARGET_PROB=${TARGET_PROBS[$(( TASK % N_PROBS ))]}

# ---------------------------------------------------------------------------
# LOSO checkpoint selection -- derived, never hand-typed
# ---------------------------------------------------------------------------
# loso_fold_NNNN_target_M holds out subject M, with fold NNNN = M + 1.
# Using a checkpoint that trained on this subject would silently invalidate
# the zero-shot framing, and nothing downstream can detect that, so it is
# computed here rather than passed in.
MODEL_FILE=$(printf "loso_fold_%04d_target_%d_zero_shot.keras" $(( SUBJECT + 1 )) "$SUBJECT")
MODEL_PATH="$MODEL_ROOT/$MODEL_FILE"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: checkpoint not found for subject $SUBJECT: $MODEL_PATH"
    exit 2
fi
for required in "$EEG_PATH" "$LABELS_PATH"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: required data file missing: $required"
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

OUT_DIR="$OUT_ROOT/subject${SUBJECT}_trial${TRIAL}_lr${LEARNING_RATE}_p${TARGET_PROB}_w${PHYS_WEIGHT}"

echo "=========================================================================="
echo "task $TASK/$((TOTAL - 1))  subject=$SUBJECT  trial=$TRIAL  lr=$LEARNING_RATE  target_p=$TARGET_PROB"
echo "physiological_weight=$PHYS_WEIGHT  ($( [[ "$PHYS_WEIGHT" == "0.0" || "$PHYS_WEIGHT" == "0" ]] && echo 'MEASUREMENT pass: VCSC logged, not optimized' || echo 'VCSC ACTIVE in objective' ))"
echo "checkpoint: $MODEL_FILE"
echo "out: $OUT_DIR"
echo "=========================================================================="

python -u -m eegproc.model_explainability.run_counterfactuals \
    --model "$MODEL_PATH" \
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
    --trial-id "$TRIAL" \
    --target-probability "$TARGET_PROB" \
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
    --log-every 10 \
    --out-dir "$OUT_DIR"

echo "task $TASK complete: $OUT_DIR"
