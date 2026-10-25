#!/bin/bash
#SBATCH --job-name=sicv16_preflight
#SBATCH --output=sicv16_preflight_%j.out
#SBATCH --error=sicv16_preflight_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
source "$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv16/SLURM_scripts/environment.sh"
nvidia-smi
python -c 'import tensorflow as tf; assert len(tf.config.list_physical_devices("GPU")) == 1, "Expected one GPU"'
python -m unittest discover -s src/tests -p test_sic_v16.py -v
