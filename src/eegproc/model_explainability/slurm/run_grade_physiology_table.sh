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
# Each array task runs both per-trial statistics: worst-family (the proposed
# Table 2 statistic) and mean-family.
#
# Correctness gate: grade_physiology_table.py refuses to count any trial whose
# refit bounds do not reproduce the archived fraction_in_range to 1e-9, and
# exits 1 if any trial fails. A failed gate means the reference is not the one
# the run used; do NOT report numbers from a run that printed GATE FAILED.
#
# SETUP (once). The grading code lives on the graded-physiological-checks
# branch. A worktree keeps your main checkout on SIC-model untouched:
#   cd ~/EEGProc
#   git fetch origin graded-physiological-checks
#   git worktree add ../EEGProc-graded origin/graded-physiological-checks
#
# SUBMIT from ~/EEGProc (so PROJECT_DIR, venv, datasets and runs resolve);
# the script itself lives in the worktree:
#   cd ~/EEGProc
#   sbatch ../EEGProc-graded/src/eegproc/model_explainability/slurm/run_grade_physiology_table.sh
#
# Overrides (export before sbatch): PROJECT_DIR, GRADE_REPO, VENV_DIR,
# AROUSAL_SUITE, VALENCE_SUITE, EEG_PATH, LABELS_PATH, OUT_DIR, TOLERANCE.
#
# COLLECT when all 6 tasks finish:
#   grep -H "LaTeX cell\|GATE FAILED\|found " runs/graded_physiology/*/*.txt
# ---------------------------------------------------------------------------

module purge
module load python/3.12.4

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/EEGProc}}"
PROJECT_DIR="$(cd -- "$PROJECT_DIR" && pwd)"
GRADE_REPO="${GRADE_REPO:-$(dirname "$PROJECT_DIR")/EEGProc-graded}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
PYTHON="$VENV_DIR/bin/python"

AROUSAL_SUITE="${AROUSAL_SUITE:-$PROJECT_DIR/runs/counterfactuals/arousal_suite_1599318_ablations_2111899}"
VALENCE_SUITE="${VALENCE_SUITE:-$PROJECT_DIR/runs/counterfactuals/valence_suite_65452590_ablations_2112037}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
TOLERANCE="${TOLERANCE:-1e-9}"

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

echo "array task $INDEX: task=$TASK which=$WHICH"
echo "  suite      $SUITE"
echo "  grade repo $GRADE_REPO"
echo "  python     $PYTHON"
echo "  output     $OUT_DIR"

# ---- preflight -------------------------------------------------------------
fail() { echo "ERROR: $*" >&2; exit 2; }

[[ -x "$PYTHON" ]] || fail "no Python at $PYTHON (set VENV_DIR)."
GRADER="$GRADE_REPO/src/eegproc/model_explainability/typicality/grade_physiology_table.py"
[[ -f "$GRADER" ]] || fail "grader not found at $GRADER. Create the worktree (see SETUP) or set GRADE_REPO."
[[ -d "$SUITE" ]] || fail "suite directory not found: $SUITE (set ${TASK^^}_SUITE)."
for f in "$EEG_PATH" "$LABELS_PATH"; do
    [[ -s "$f" ]] || fail "dataset file missing: $f"
done

# cross_val.py does `from joblib.externals import cloudpickle`, which joblib
# >= 1.5 removed. The data loader imports it, so check before burning time.
"$PYTHON" - <<'PY' || fail "joblib.externals.cloudpickle unavailable; in this venv run: pip install 'joblib<1.5'"
from joblib.externals import cloudpickle  # noqa: F401
PY

"$PYTHON" - "$GRADE_REPO" <<'PY' || fail "GRADE_REPO lacks PhysiologicalReference.deviation(); is it on graded-physiological-checks?"
import sys, unittest.mock as m
sys.path.insert(0, sys.argv[1] + "/src")
for name in ("tensorflow", "tensorflow.keras", "keras"):
    sys.modules.setdefault(name, m.MagicMock())
from eegproc.model_explainability.typicality.physiology import PhysiologicalReference
assert hasattr(PhysiologicalReference, "deviation")
PY

N_FILES=$(find "$SUITE" -path "*/attempt_*/physiology_${WHICH}.npz" | wc -l)
N_JSON=$(find "$SUITE" -path "*/attempt_*/physiology.json" | wc -l)
echo "  archived physiology_${WHICH}.npz: $N_FILES   physiology.json: $N_JSON"
[[ "$N_FILES" -gt 0 ]] || fail "no physiology_${WHICH}.npz under $SUITE. Were state arrays archived for this suite?"
if [[ "$N_FILES" -ne $((EXPECTED * N_OBJECTIVES)) ]]; then
    echo "WARNING: expected $((EXPECTED * N_OBJECTIVES)) files ($EXPECTED trials x $N_OBJECTIVES objectives), found $N_FILES." >&2
    echo "         Check per-objective n in the output before using the numbers." >&2
fi

# ---- run -------------------------------------------------------------------
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
cd "$GRADE_REPO"

STATUS=0
for STATISTIC in worst mean; do
    LOG="$OUT_DIR/${TASK}_${WHICH}_${STATISTIC}.txt"
    echo
    echo "=== $TASK / $WHICH / $STATISTIC -> $LOG"
    set +e
    "$PYTHON" "$GRADER" \
        --suite "$SUITE" \
        --task "$TASK" \
        --which "$WHICH" \
        --statistic "$STATISTIC" \
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
        echo "!!! $TASK/$WHICH/$STATISTIC exited $CODE (1 = gate failure; see GATE FAILED list in $LOG)" >&2
        STATUS=1
    fi
done

echo
echo "done: task=$TASK which=$WHICH status=$STATUS"
exit "$STATUS"
