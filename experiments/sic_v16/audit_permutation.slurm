#!/bin/bash
#SBATCH --job-name=sicv16-audit-permutation
#SBATCH --output=sicv16-audit-permutation-%j.out
#SBATCH --error=sicv16-audit-permutation-%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source "$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv16/SLURM_scripts/environment.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
case "${CONTROL_MODE:?}" in
prepare)
    exec python experiments/sic_v16/permutation_audit.py prepare --data-dir "${DATA_DIR:-$PROJECT_DIR/datasets}" --output-dir "$CONTROL_ROOT"
    ;;
run)
    exec python experiments/sic_v16/nested_logpower.py --data-dir "${DATA_DIR:-$PROJECT_DIR/datasets}" --feature-cache "$CONTROL_ROOT/prepared" --permutation-index "$SLURM_ARRAY_TASK_ID" --output-dir "$CONTROL_ROOT/run-$SLURM_ARRAY_TASK_ID"
    ;;
summarize)
    exec python experiments/sic_v16/permutation_audit.py summarize --output-dir "$CONTROL_ROOT" --permutations "$PERMUTATIONS"
    ;;
*) exit 2 ;;
esac
