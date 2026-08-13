#!/bin/bash

set -euo pipefail

# ---------------------------------------------------------------------------
# Organized submission launcher for the SIC Slurm arrays
# ---------------------------------------------------------------------------
# Usage:
#   ./submit_sic_slurm_suite.sh smoke [MAX_CONCURRENT]
#   ./submit_sic_slurm_suite.sh full [MAX_CONCURRENT]
#   ./submit_sic_slurm_suite.sh full-after-smoke [MAX_CONCURRENT]
#
# Examples:
#   ./submit_sic_slurm_suite.sh smoke 2
#   ./submit_sic_slurm_suite.sh full 2
#   ./submit_sic_slurm_suite.sh full-after-smoke 2
#
# Environment overrides are forwarded to every array task:
#   SOURCE_EPOCHS=150 MLDG_STEPS_PER_EPOCH=20 ./submit_sic_slurm_suite.sh full 2
#   TARGET_DIMENSION=arousal PREDICTION_DIAGNOSTICS_METRIC=ece \
#       ./submit_sic_slurm_suite.sh full-after-smoke 2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_SCRIPT="$SCRIPT_DIR/smoke_test_sic_mldg_brier.sh"
FULL_SCRIPT="$SCRIPT_DIR/full_run_sic_mldg_brier_ablation_valence.sh"

MODE="${1:-}"
MAX_CONCURRENT="${2:-2}"

if [[ ! "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_CONCURRENT must be a positive integer; got '$MAX_CONCURRENT'."
    exit 2
fi

for required_script in "$SMOKE_SCRIPT" "$FULL_SCRIPT"; do
    if [[ ! -f "$required_script" ]]; then
        echo "ERROR: required worker script not found: $required_script"
        exit 2
    fi
done

submit_smoke() {
    sbatch --parsable \
        --array="0-1%${MAX_CONCURRENT}" \
        "$SMOKE_SCRIPT"
}

submit_full() {
    local dependency="${1:-}"
    local command=(
        sbatch
        --parsable
        --array="0-3%${MAX_CONCURRENT}"
    )
    if [[ -n "$dependency" ]]; then
        command+=(--dependency="afterok:${dependency}")
    fi
    command+=("$FULL_SCRIPT")
    "${command[@]}"
}

case "$MODE" in
    smoke)
        smoke_job_id="$(submit_smoke)"
        echo "Submitted smoke array: $smoke_job_id"
        echo "Profiles: 0=full, 1=no_bilstm"
        ;;
    full)
        full_job_id="$(submit_full)"
        echo "Submitted full array: $full_job_id"
        echo "Profiles: 0=full, 1=remove_median, 2=no_gcn_gru, 3=no_bilstm"
        ;;
    full-after-smoke)
        smoke_job_id="$(submit_smoke)"
        full_job_id="$(submit_full "$smoke_job_id")"
        echo "Submitted smoke array: $smoke_job_id"
        echo "Submitted dependent full array: $full_job_id"
        echo "The full array will start only if every smoke task succeeds."
        echo "Full profiles: 0=full, 1=remove_median, 2=no_gcn_gru, 3=no_bilstm"
        ;;
    *)
        echo "Usage: $0 {smoke|full|full-after-smoke} [MAX_CONCURRENT]"
        exit 2
        ;;
esac
