#!/bin/bash
#SBATCH --job-name=sicv16-overfit-36
#SBATCH --output=sicv16-overfit-36-%j.out
#SBATCH --error=sicv16-overfit-36-%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source "$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv16/SLURM_scripts/environment.sh"
exec python "$PROJECT_DIR/experiments/sic_v16/overfit_36.py" \
    --data-dir "${DATA_DIR:-$PROJECT_DIR/datasets}" \
    --output-dir "$SIC_V16_WORK_ROOT/overfit-36/job-${SLURM_JOB_ID}" \
    --epochs "${OVERFIT_EPOCHS:-2000}" \
    --learning-rate "${OVERFIT_LEARNING_RATE:-0.001}"
