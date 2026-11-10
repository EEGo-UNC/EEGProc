#!/bin/bash
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export PERMUTATIONS="${PERMUTATIONS:-100}"
[[ "$PERMUTATIONS" =~ ^[1-9][0-9]*$ ]] || { echo "PERMUTATIONS must be positive" >&2; exit 1; }
export SIC_V16_WORK_ROOT="${SIC_V16_WORK_ROOT:-/work/users/${USER:0:1}/${USER:1:1}/$USER/sic-v16}"
export CONTROL_ROOT="$SIC_V16_WORK_ROOT/audit-permutation/$(date +%Y%m%d-%H%M%S)-$$"
launcher="$PROJECT_DIR/experiments/sic_v16/audit_permutation.slurm"
prepare=$(sbatch --parsable --export=ALL,CONTROL_MODE=prepare "$launcher")
prepare="${prepare%%;*}"
runs=$(sbatch --parsable --dependency="afterok:$prepare" --array="0-${PERMUTATIONS}%4" --export=ALL,CONTROL_MODE=run "$launcher")
runs="${runs%%;*}"
summary=$(sbatch --parsable --dependency="afterok:$runs" --export=ALL,CONTROL_MODE=summarize "$launcher")
echo "Audit job: $prepare; baseline/permutation array: $runs; summary job: $summary"
echo "Results: $CONTROL_ROOT"
