#!/bin/bash
#SBATCH --job-name=sicv16_smoke
#SBATCH --output=sicv16_smoke_%A_%a.out
#SBATCH --error=sicv16_smoke_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --array=0-3%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
source "$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv16/SLURM_scripts/environment.sh"
export SOURCE_EPOCHS="${SOURCE_EPOCHS:-2}"
export TARGET_SUBJECTS="${TARGET_SUBJECTS:-0 1}"
export TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-6600}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs/sic_v16/smoke}"
bash "$SIC16/SLURM_scripts/task.sh"
