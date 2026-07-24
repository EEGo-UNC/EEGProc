#!/usr/bin/env bash
#SBATCH --job-name=gcn_softmax_smoke
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/gcn_softmax_smoke_%j.out
#SBATCH --error=slurm_logs/gcn_softmax_smoke_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs runs/gcn_softmax_smoke

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=1

# First confirm that the architecture imports and performs one training step.
python -m eegproc.deep_learning.gcn_softmax.smoke_test

# Then run two short LOSO folds through the real dataset and evaluator.
python -m eegproc.deep_learning.gcn_softmax.train \
  --dataset dreamer \
  --label-dimension arousal \
  --window-sec 4.0 \
  --window-overlap 0.0 \
  --n-channels 14 \
  --n-bands 4 \
  --epochs 3 \
  --batch-size 8 \
  --validation-subjects 4 \
  --validation-seed 42 \
  --selection-level trial \
  --selection-metric f1 \
  --early-stopping-monitor val_trial_f1 \
  --early-stopping-mode max \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.005 \
  --max-folds 2 \
  --hyperparameters-json '{
    "epochs": [3],
    "batch_size": [8],
    "learning_rate": [0.0001],
    "gcn_units": [[32, 16]],
    "temporal_pool_sizes": [[2]],
    "t_down": [2],
    "emb_dim": [32],
    "dropout": [0.3],
    "activation": ["relu"],
    "use_batch_norm": [false],
    "temporal_readout": ["mean_max"],
    "classifier_units": [64],
    "classifier_dropout": [0.5],
    "l2_weight": [0.0001],
    "clipnorm": [1.0]
  }' \
  --out-dir runs/gcn_softmax_smoke \
  --run-name smoke \
  --n-jobs 1 \
  --cpus-per-worker "${SLURM_CPUS_PER_TASK:-4}" \
  --no-final-model \
  --verbose 2
