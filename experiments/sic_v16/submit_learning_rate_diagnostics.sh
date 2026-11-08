#!/bin/bash
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export SIC_V16_WORK_ROOT="${SIC_V16_WORK_ROOT:-/work/users/${USER:0:1}/${USER:1:1}/$USER/sic-v16}"
export SWEEP_ROOT="$SIC_V16_WORK_ROOT/learning-rate-diagnostics/$(date +%Y%m%d-%H%M%S)-$$"
launcher="$PROJECT_DIR/experiments/sic_v16/learning_rate_diagnostics.slurm"
prep=$(sbatch --parsable --export=ALL,SWEEP_MODE=prepare "$launcher"); prep="${prep%%;*}"
runs=$(sbatch --parsable --dependency="afterok:$prep" --array=0-1%2 --export=ALL,SWEEP_MODE=diagnose "$launcher"); runs="${runs%%;*}"
summary=$(sbatch --parsable --dependency="afterok:$runs" --export=ALL,SWEEP_MODE=diagnostic_summary "$launcher")
echo "Preparation: $prep; diagnostic array: $runs; summary: $summary"
echo "Diagnostic directory: $SWEEP_ROOT"
