#!/bin/bash

set -euo pipefail

# SLURM resolves relative --output/--error paths before the job script starts,
# so the job itself cannot safely create its log directory. This wrapper does
# that first and passes absolute log paths to sbatch.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$HOME/EEGProc}"
LOG_DIR="${SLURM_LOG_DIR:-$PROJECT_DIR/runs/counterfactuals/z0_sweep_slurm_logs}"
RUN_SCRIPT="$SCRIPT_DIR/run_vcsc_z0_sweep.sh"

mkdir -p "$LOG_DIR"

exec sbatch \
    --output="$LOG_DIR/vcsc_z0_%A_%a.out" \
    --error="$LOG_DIR/vcsc_z0_%A_%a.err" \
    "$@" \
    "$RUN_SCRIPT"
