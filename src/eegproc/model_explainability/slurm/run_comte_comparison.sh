#!/usr/bin/env bash
#SBATCH --job-name=cfo-comte
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Activate the same Python environment used for the saved SIC study first.
# Required environment: PROJECT_ROOT, TASK, MODELS_JSON, TRIALS_NPZ, OUT_DIR.
# Optional MODEL_DIR relocates checkpoint filenames. Remaining CLI arguments
# override defaults and can select --subjects, budgets, and exact CFO settings.
set -euo pipefail
: "${PROJECT_ROOT:?Set PROJECT_ROOT to this repository checkout}"
: "${TASK:?Set TASK to valence or arousal}"
: "${MODELS_JSON:?Set MODELS_JSON to the frozen LOSO manifest}"
: "${TRIALS_NPZ:?Set TRIALS_NPZ to the matching prepared trials}"
: "${OUT_DIR:?Set OUT_DIR to a new comparison directory}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
relocation=()
if [[ -n "${MODEL_DIR:-}" ]]; then
  relocation=(--model-dir "$MODEL_DIR")
fi
python -m eegproc.model_explainability.comte.runner \
  --models-json "$MODELS_JSON" "${relocation[@]}" \
  --trials-npz "$TRIALS_NPZ" --task "$TASK" \
  --directions 0_to_1 1_to_0 --cfo-objective typicality \
  --out-dir "$OUT_DIR" "$@"
