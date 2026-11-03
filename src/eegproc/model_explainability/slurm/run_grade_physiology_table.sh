#!/bin/bash
#SBATCH --job-name=grade_physiology
#SBATCH --output=grade_physiology_%A_%a.out
#SBATCH --error=grade_physiology_%A_%a.err
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-5

set -euo pipefail

# ---------------------------------------------------------------------------
# Graded physiology column for Table 2 (CPU only, no GPU, no model loading).
# ---------------------------------------------------------------------------
# Rescores every archived physiology_<which>.npz in the two CFO ablation
# suites against a PhysiologicalReference refit from source-subject EEG, and
# prints median [Q1, Q3] exceedance per objective (base, typicality,
# typicality_no_physiology, target_latent).
#
# Array layout: 2 tasks x 3 signals = 6 array tasks.
#   0 arousal counterfactual   <- Table 2 column
#   1 arousal reconstruction   <- decoder-only reference row
#   2 arousal original         <- real-EEG reference row
#   3 valence counterfactual   <- Table 2 column
#   4 valence reconstruction
#   5 valence original
# REPORT=graded runs worst-family and mean-family graded distances.
# REPORT=binary reads saved JSON and reports each family's pass percentage
# and in-range fraction distribution, without inferring graded distances.
#
# Correctness gate: grade_physiology_table.py refuses to count any trial whose
# refit bounds do not reproduce the archived fraction_in_range to 1e-9, and
# exits 1 if any trial fails. A failed gate means the reference is not the one
# the run used; do NOT report numbers from a run that printed GATE FAILED.
#
# SUBMIT from the valence-enhancement checkout after merging the grading code:
#   cd ~/EEGProc
#   sbatch src/eegproc/model_explainability/slurm/run_grade_physiology_table.sh
#
# Grading runs from this checkout by default. GRADE_REPO can still point to a
# separate checkout containing the grading modules.
#
# Overrides (export before sbatch): PROJECT_DIR, GRADE_REPO, VENV_DIR,
# AROUSAL_SUITE, VALENCE_SUITE, EEG_PATH, LABELS_PATH, OUT_DIR, TOLERANCE,
# REPORT (graded by default; binary works with summary-only JSON archives).
#
# COLLECT when all 6 tasks finish:
#   grep -H "LaTeX cell\|GATE FAILED\|found " runs/graded_physiology/*/*.txt
# ---------------------------------------------------------------------------

module purge
module load python/3.12.4

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/EEGProc}}"
PROJECT_DIR="$(cd -- "$PROJECT_DIR" && pwd)"
GRADE_REPO="${GRADE_REPO:-$PROJECT_DIR}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
PYTHON="$VENV_DIR/bin/python"

AROUSAL_SUITE="${AROUSAL_SUITE:-$PROJECT_DIR/runs/counterfactuals/arousal_suite_1599318_ablations_2355492}"
VALENCE_SUITE="${VALENCE_SUITE:-$PROJECT_DIR/runs/counterfactuals/valence_suite_65452590_ablations_2356028}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
TOLERANCE="${TOLERANCE:-1e-9}"
REPORT="${REPORT:-graded}"
if [[ "$REPORT" != graded && "$REPORT" != binary ]]; then
    echo "ERROR: REPORT must be graded or binary." >&2
    exit 2
fi

# These match the archived runs (fold_00 study.json). Changing any of them
# makes the refit reference differ from the one the runs used, and the gate
# will (correctly) fail.
FS=128
PHYSIOLOGY_QUANTILE=0.95
PHYSIOLOGY_REQUIRED_FRACTION=0.95

# Eligible trials per objective, from the ablation reports. Used only for a
# warning if the archive count looks off; the grader is the source of truth.
EXPECTED_AROUSAL=57
EXPECTED_VALENCE=92
N_OBJECTIVES=4

TASKS=(arousal arousal arousal valence valence valence)
WHICHS=(counterfactual reconstruction original counterfactual reconstruction original)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "$INDEX" =~ ^[0-5]$ ]]; then
    echo "ERROR: array index must be 0-5, got '$INDEX'." >&2
    exit 2
fi
TASK="${TASKS[$INDEX]}"
WHICH="${WHICHS[$INDEX]}"
if [[ "$TASK" == arousal ]]; then
    SUITE="$AROUSAL_SUITE"; EXPECTED="$EXPECTED_AROUSAL"
else
    SUITE="$VALENCE_SUITE"; EXPECTED="$EXPECTED_VALENCE"
fi

RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/runs/graded_physiology/$RUN_ID}"
mkdir -p "$OUT_DIR"

echo "array task $INDEX: task=$TASK which=$WHICH report=$REPORT"
echo "  suite      $SUITE"
echo "  grade repo $GRADE_REPO"
echo "  python     $PYTHON"
echo "  output     $OUT_DIR"

# ---- preflight -------------------------------------------------------------
fail() { echo "ERROR: $*" >&2; exit 2; }

[[ -x "$PYTHON" ]] || fail "no Python at $PYTHON (set VENV_DIR)."
GRADER="$GRADE_REPO/src/eegproc/model_explainability/typicality/grade_physiology_table.py"
[[ -f "$GRADER" ]] || fail "grader not found at $GRADER. Use the merged checkout or set GRADE_REPO."
[[ -d "$SUITE" ]] || fail "suite directory not found: $SUITE (set ${TASK^^}_SUITE)."

# Raw EEG is loaded by the grader only when component arrays need rescoring.
# Saved binary/graded JSON summaries do not require datasets or checkpoints.

"$PYTHON" - "$GRADE_REPO" <<'PY' || fail "GRADE_REPO lacks the merged physiological grading modules."
import sys
sys.path.insert(0, sys.argv[1] + "/src")
from eegproc.model_explainability.typicality.physiology import PhysiologicalReference
from eegproc.model_explainability.typicality.grading_data import load_grading_trials
assert hasattr(PhysiologicalReference, "deviation")
PY

N_FILES=$(find "$SUITE" -path "*/attempt_*/physiology_${WHICH}.npz" | wc -l)
N_JSON=$(find "$SUITE" -path "*/attempt_*/physiology.json" | wc -l)
echo "  archived physiology_${WHICH}.npz: $N_FILES   physiology.json: $N_JSON"
[[ "$N_JSON" -gt 0 ]] || fail "no attempt physiology.json summaries under $SUITE."
if [[ "$N_JSON" -ne $((EXPECTED * N_OBJECTIVES)) ]]; then
    echo "WARNING: expected $((EXPECTED * N_OBJECTIVES)) summaries ($EXPECTED trials x $N_OBJECTIVES objectives), found $N_JSON." >&2
    echo "         Check per-objective n in the output before using the numbers." >&2
fi

# ---- run -------------------------------------------------------------------
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
cd "$GRADE_REPO"

STATUS=0
STATISTICS=(worst mean)
[[ "$REPORT" == binary ]] && STATISTICS=(binary)
for STATISTIC in "${STATISTICS[@]}"; do
    STATISTIC_ARGS=(--statistic worst)
    [[ "$REPORT" == graded ]] && STATISTIC_ARGS=(--statistic "$STATISTIC")
    LOG="$OUT_DIR/${TASK}_${WHICH}_${STATISTIC}.txt"
    echo
    echo "=== $TASK / $WHICH / $STATISTIC -> $LOG"
    set +e
    "$PYTHON" "$GRADER" \
        --suite "$SUITE" \
        --task "$TASK" \
        --which "$WHICH" \
        --report "$REPORT" \
        "${STATISTIC_ARGS[@]}" \
        --repo "$GRADE_REPO" \
        --eeg "$EEG_PATH" \
        --labels "$LABELS_PATH" \
        --fs "$FS" \
        --quantile "$PHYSIOLOGY_QUANTILE" \
        --required-fraction "$PHYSIOLOGY_REQUIRED_FRACTION" \
        --tolerance "$TOLERANCE" \
        2>&1 | tee "$LOG"
    CODE=${PIPESTATUS[0]}
    set -e
    if [[ "$CODE" -ne 0 ]]; then
        echo "!!! $TASK/$WHICH/$STATISTIC exited $CODE (1 = bounds mismatch; 2 = unavailable input; see $LOG)" >&2
        STATUS=1
    fi
done

echo
echo "done: task=$TASK which=$WHICH status=$STATUS"
exit "$STATUS"
