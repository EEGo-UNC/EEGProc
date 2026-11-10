#!/bin/bash
#SBATCH --job-name=sicv15-seed-train
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-14%2
#SBATCH --output=/work/users/p/m/pmucharl/sic-v15-seed-iv/logs/train-%A_%a.out
#SBATCH --error=/work/users/p/m/pmucharl/sic-v15-seed-iv/logs/train-%A_%a.err
set -euo pipefail
source /work/users/p/m/pmucharl/sic-v15-seed-iv/source/experiments/sic_v15_seed_iv/environment.sh
module load cuda/12.9 cudnn/9.11.0
export PYTHONHASHSEED=42 TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TF_GPU_ALLOCATOR=cuda_malloc_async
CUDA_ROOT="${CUDA_HOME:-${EBROOTCUDA:-}}"
if [[ -z "$CUDA_ROOT" ]]; then
  CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi
test -f "$CUDA_ROOT/nvvm/libdevice/libdevice.10.bc"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_ROOT"
export CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT"
nvidia-smi
ARGS=()
STAGE=full
if [[ "${SMOKE:-0}" == 1 ]]; then
  ARGS+=(--smoke)
  STAGE=smoke
fi
"$PYTHON" experiments/sic_v15_seed_iv/train.py \
  --data-dir "$ROOT/processed" \
  --out-dir "$ROOT/runs/$STAGE/subject_${SLURM_ARRAY_TASK_ID}" \
  --target "$SLURM_ARRAY_TASK_ID" "${ARGS[@]}"
