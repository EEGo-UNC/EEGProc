#!/bin/bash
#SBATCH --job-name=stsnet_dreamer_arousal
#SBATCH --output=stsnet_dreamer_arousal_%j.out
#SBATCH --error=stsnet_dreamer_arousal_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

module purge
module load python
module load cuda/12.9
module load cudnn/9.11.0

cd ~/stsnet
source venv/bin/activate

python -m pip install --no-cache-dir -r requirements.txt

cd SMC_pub
mkdir -p outputs

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working dir: $(pwd)"

export PYTHONUNBUFFERED=1

# Resolve CUDA toolkit root so TensorFlow/XLA can find nvvm/libdevice
CUDA_TOOLKIT_ROOT=""

if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}/nvvm/libdevice" ]; then
    CUDA_TOOLKIT_ROOT="${CUDA_HOME}"
elif [ -n "${CUDA_DIR:-}" ] && [ -d "${CUDA_DIR}/nvvm/libdevice" ]; then
    CUDA_TOOLKIT_ROOT="${CUDA_DIR}"
elif command -v nvcc >/dev/null 2>&1; then
    CUDA_TOOLKIT_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi

# Fallback search in case the module does not export CUDA_HOME/CUDA_DIR cleanly
if [ -z "${CUDA_TOOLKIT_ROOT}" ] || [ ! -d "${CUDA_TOOLKIT_ROOT}/nvvm/libdevice" ]; then
    LIBDEVICE_PATH="$(find /nas/longleaf/apps -path '*/nvvm/libdevice/libdevice*.bc' 2>/dev/null | head -n 1 || true)"
    if [ -n "${LIBDEVICE_PATH}" ]; then
        CUDA_TOOLKIT_ROOT="$(dirname "$(dirname "$(dirname "${LIBDEVICE_PATH}")")")"
    fi
fi

if [ -z "${CUDA_TOOLKIT_ROOT}" ] || [ ! -d "${CUDA_TOOLKIT_ROOT}/nvvm/libdevice" ]; then
    echo "ERROR: Could not find CUDA toolkit root with nvvm/libdevice"
    echo "CUDA_HOME=${CUDA_HOME:-unset}"
    echo "CUDA_DIR=${CUDA_DIR:-unset}"
    which nvcc || true
    exit 1
fi

export CUDA_HOME="${CUDA_TOOLKIT_ROOT}"
export CUDA_DIR="${CUDA_TOOLKIT_ROOT}"
export PATH="${CUDA_TOOLKIT_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_TOOLKIT_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_TOOLKIT_ROOT}"

echo "==== CUDA CHECK ===="
echo "CUDA_TOOLKIT_ROOT=${CUDA_TOOLKIT_ROOT}"
which nvcc || true
nvidia-smi
ls -l "${CUDA_TOOLKIT_ROOT}/nvvm/libdevice" || true

python - <<'PY'
import tensorflow as tf

print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Small preflight XLA test to catch libdevice problems early
@tf.function(jit_compile=True)
def test_fn(x):
    return tf.sign(x)

x = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32)
y = test_fn(x)
print("XLA test passed:", y.numpy())
PY

python -u train_eval.py \
    --dataset dreamer \
    --dimension arousal \
    --eeg_path data/dreamer_eeg.npy \
    --label_path data/dreamer_labels.npy \
    --results_dir outputs/dreamer_arousal
