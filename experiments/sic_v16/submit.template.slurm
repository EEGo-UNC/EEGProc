#!/bin/bash
#SBATCH --job-name={{RUN_ID}}
#SBATCH --output={{EXPERIMENT_DIR}}/runs/{{RUN_ID}}/slurm-%j.out
#SBATCH --error={{EXPERIMENT_DIR}}/runs/{{RUN_ID}}/slurm-%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time={{TIME_LIMIT}}
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
source "$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv16/SLURM_scripts/environment.sh"
exec python "$PROJECT_DIR/{{TRAINING_SCRIPT}}" \
    --config "$PROJECT_DIR/{{EXPERIMENT_DIR}}/runs/{{RUN_ID}}/config.json" \
    --output-dir "$PROJECT_DIR/{{EXPERIMENT_DIR}}/runs/{{RUN_ID}}" \
    --data-dir "${DATA_DIR:-$PROJECT_DIR/datasets}"
