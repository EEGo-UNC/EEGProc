#!/bin/bash
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export SIC_V16_WORK_ROOT="${SIC_V16_WORK_ROOT:-/work/users/${USER:0:1}/${USER:1:1}/$USER/sic-v16}"
export SWEEP_ROOT="$SIC_V16_WORK_ROOT/rebuilt-four-model/$(date +%Y%m%d-%H%M%S)-$$"
launcher="$PROJECT_DIR/experiments/sic_v16/rebuilt_model_sweep.slurm"
prep=$(sbatch --parsable --export=ALL,SWEEP_MODE=prepare "$launcher"); prep="${prep%%;*}"
checks=$(sbatch --parsable --dependency="afterok:$prep" --array=0-3%4 --export=ALL,SWEEP_MODE=check "$launcher"); checks="${checks%%;*}"
gate_report=$(sbatch --parsable --dependency="afterany:$checks" --export=ALL,SWEEP_MODE=collect "$launcher")
full=$(sbatch --parsable --dependency="afterok:$checks" --array=0-91%4 --export=ALL,SWEEP_MODE=train "$launcher"); full="${full%%;*}"
summary=$(sbatch --parsable --dependency="afterok:$full" --export=ALL,SWEEP_MODE=summarize "$launcher")
echo "Preparation: $prep; learning checks: $checks; check archive: $gate_report"
echo "Full array: $full; summary: $summary"
echo "Sweep directory: $SWEEP_ROOT"
echo "Full training runs only if all four memorization checks pass."
