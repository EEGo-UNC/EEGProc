#!/bin/bash
# Prepare/submit both matched 15-participant experiments from the reviewed snapshot.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
SMOKE="${SMOKE:-0}"
COMPARISON_ID="${COMPARISON_ID:-hidden_params_x2_$(date +%Y%m%d_%H%M%S)}"
while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --id) COMPARISON_ID="${2:?--id requires a value}"; shift 2 ;;
    *) echo "Usage: bash submit_comparison.sh [--dry-run] [--smoke] [--id NAME]" >&2; exit 2 ;;
  esac
done
if [[ ! "$COMPARISON_ID" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
  echo "Comparison ID must contain only letters, digits, dots, underscores, or hyphens" >&2
  exit 2
fi
if [[ "$DRY_RUN" == 0 ]]; then
  command -v sbatch >/dev/null || { echo "Run submission on the Slurm login node" >&2; exit 2; }
  source "$SCRIPT_DIR/environment.sh"
fi
ROOT="${ROOT:-/work/users/p/m/pmucharl/sic-v15-seed-iv}"
COMPARISON_DIR="$ROOT/runs/$COMPARISON_ID"
STAGE=full
if [[ "$SMOKE" == 1 ]]; then STAGE=smoke; fi
if [[ "$DRY_RUN" == 0 ]]; then
  mkdir "$COMPARISON_DIR"
  printf 'variant\tfirst_source\tremaining_source\tselection\tcalibration\n' > "$COMPARISON_DIR/submission.tsv"
fi
submit() {
  local label="$1"
  shift
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'sbatch --parsable ' >&2
    printf '%q ' "$@" >&2
    printf '\n' >&2
    printf 'DRY_%s\n' "$label"
  else
    local result
    result="$(sbatch --parsable "$@")"
    printf '%s\n' "${result%%;*}"
  fi
}
for VARIANT in baseline hidden_params_x2; do
  CONFIG="$SCRIPT_DIR/model_config.json"
  if [[ "$VARIANT" == hidden_params_x2 ]]; then CONFIG="$SCRIPT_DIR/model_config_hidden_params_x2.json"; fi
  test -f "$CONFIG"
  SUITE="$COMPARISON_DIR/$VARIANT/$STAGE"
  EXPORT="ALL,SUITE_DIR=$SUITE,MODEL_CONFIG=$CONFIG,SOURCE_EPOCHS=${SOURCE_EPOCHS:-12},SMOKE=$SMOKE"
  FIRST="$(submit "${VARIANT}_first" --array=0 --export="$EXPORT,PHASE=source" "$SCRIPT_DIR/train.slurm")"
  REMAINING="$(submit "${VARIANT}_remaining" --array=1-14%2 --dependency="afterok:$FIRST" --export="$EXPORT,PHASE=source" "$SCRIPT_DIR/train.slurm")"
  SELECT="$(submit "${VARIANT}_select" --dependency="afterok:$REMAINING" --export="$EXPORT" "$SCRIPT_DIR/select.slurm")"
  CALIBRATE="$(submit "${VARIANT}_calibrate" --array=0-14%2 --dependency="afterok:$SELECT" --export="$EXPORT,PHASE=calibrate" "$SCRIPT_DIR/train.slurm")"
  printf '%s: first=%s remaining=%s select=%s calibrate=%s suite=%s\n' "$VARIANT" "$FIRST" "$REMAINING" "$SELECT" "$CALIBRATE" "$SUITE"
  if [[ "$DRY_RUN" == 0 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$VARIANT" "$FIRST" "$REMAINING" "$SELECT" "$CALIBRATE" >> "$COMPARISON_DIR/submission.tsv"
  fi
done
