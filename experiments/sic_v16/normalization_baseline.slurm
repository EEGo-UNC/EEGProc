#!/bin/bash
#SBATCH --job-name=sicv16-normalization-baseline
#SBATCH --output=sicv16-normalization-baseline-%j.out
#SBATCH --error=sicv16-normalization-baseline-%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source "$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv16/SLURM_scripts/environment.sh"
exec python "$PROJECT_DIR/experiments/sic_v16/normalization_baseline.py" \
    --data-dir "${DATA_DIR:-$PROJECT_DIR/datasets}" \
    --output-dir "$SIC_V16_WORK_ROOT/normalization-baseline/job-${SLURM_JOB_ID}"
