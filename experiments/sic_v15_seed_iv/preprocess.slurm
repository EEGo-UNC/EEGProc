#!/bin/bash
#SBATCH --job-name=sicv15-seed-preprocess
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=1-15%3
#SBATCH --output=/work/users/p/m/pmucharl/sic-v15-seed-iv/logs/preprocess-%A_%a.out
#SBATCH --error=/work/users/p/m/pmucharl/sic-v15-seed-iv/logs/preprocess-%A_%a.err
set -euo pipefail
source /work/users/p/m/pmucharl/sic-v15-seed-iv/source/experiments/sic_v15_seed_iv/environment.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$PYTHON" experiments/sic_v15_seed_iv/preprocess.py \
  --raw-dir /work/users/p/m/pmucharl/data/SEED_IV/eeg_raw_data \
  --out-dir "$ROOT/processed" --subject "$SLURM_ARRAY_TASK_ID"
