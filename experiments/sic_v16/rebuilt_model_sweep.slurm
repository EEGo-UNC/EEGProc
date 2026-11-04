#!/bin/bash
#SBATCH --job-name=sicv16-rebuilt-four-model
#SBATCH --output=sicv16-rebuilt-four-model-%j.out
#SBATCH --error=sicv16-rebuilt-four-model-%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source "$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv16/SLURM_scripts/environment.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
extra_args=()
if [[ -n "${REBUILD_DIR:-}" ]]; then extra_args+=(--rebuild-dir "$REBUILD_DIR"); fi
exec python "$PROJECT_DIR/experiments/sic_v16/rebuilt_model_sweep.py" "${SWEEP_MODE:?}" \
    --root "${SWEEP_ROOT:?}" --index "${SLURM_ARRAY_TASK_ID:-0}" \
    --epochs "${SWEEP_EPOCHS:-500}" --check-epochs "${CHECK_EPOCHS:-1000}" "${extra_args[@]}"
