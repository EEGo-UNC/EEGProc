#!/usr/bin/env bash
#SBATCH --job-name=gcn_softmax_arousal
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/gcn_softmax_%j.out
#SBATCH --error=slurm_logs/gcn_softmax_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs runs/gcn_softmax

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TF_NUM_INTEROP_THREADS=2

python -m eegproc.deep_learning.gcn_softmax.train \
  --dataset dreamer \
  --label-dimension arousal \
  --window-sec 4.0 \
  --window-overlap 0.0 \
  --n-channels 14 \
  --n-bands 4 \
  --epochs 300 \
  --batch-size 8 \
  --validation-subjects 4 \
  --validation-seed 42 \
  --selection-level trial \
  --selection-metric f1 \
  --early-stopping-monitor val_trial_f1 \
  --early-stopping-mode max \
  --early-stopping-patience 30 \
  --early-stopping-min-delta 0.005 \
  --hyperparameters-file src/eegproc/deep_learning/gcn_softmax/configs/gcn_medium.json \
  --out-dir runs/gcn_softmax \
  --run-name dreamer_arousal_gcn_softmax \
  --n-jobs 1 \
  --cpus-per-worker "${SLURM_CPUS_PER_TASK:-8}" \
  --verbose 2
