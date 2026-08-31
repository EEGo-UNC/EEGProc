#!/bin/bash
#SBATCH --job-name=sic-calibration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --partition=a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --output=sic-calibration_%j.out
#SBATCH --error=sic-calibration_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the EEGProc repository root}"

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

EEGPROC_VENV="${EEGPROC_VENV:-${SLURM_SUBMIT_DIR}/venv312}"
if [[ ! -f "${EEGPROC_VENV}/bin/activate" ]]; then
  echo "Python environment not found: ${EEGPROC_VENV}" >&2
  echo "Set EEGPROC_VENV to the environment containing EEGProc's dependencies." >&2
  exit 2
fi
source "${EEGPROC_VENV}/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=1
export OMP_NUM_THREADS=4

# TensorFlow's GPU kernel generator needs CUDA's libdevice bitcode. Longleaf's
# CUDA module is not installed at /usr/local/cuda, so point XLA at the toolkit
# root selected by the loaded module.
EEGPROC_NVCC="$(command -v nvcc || true)"
if [[ -z "${EEGPROC_NVCC}" ]]; then
  echo "nvcc was not found after loading cuda/12.9." >&2
  exit 2
fi
EEGPROC_CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "${EEGPROC_NVCC}")")")"
EEGPROC_LIBDEVICE="${EEGPROC_CUDA_ROOT}/nvvm/libdevice/libdevice.10.bc"
if [[ ! -f "${EEGPROC_LIBDEVICE}" ]]; then
  echo "CUDA libdevice was not found: ${EEGPROC_LIBDEVICE}" >&2
  exit 2
fi
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_cuda_data_dir=${EEGPROC_CUDA_ROOT}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "CUDA toolkit root: ${EEGPROC_CUDA_ROOT}"
echo "CUDA libdevice: ${EEGPROC_LIBDEVICE}"
nvidia-smi -L

# Fail before the full grid search if TensorFlow still cannot compile the same
# GPU operation that Keras uses while rebuilding the saved recurrent model.
srun python - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) != 2:
    raise RuntimeError(f"Expected exactly 2 visible GPUs; TensorFlow sees {gpus}")

for index in range(2):
    with tf.device(f"/GPU:{index}"):
        actual = tf.sign(tf.constant([-1.0, 0.0, 1.0])).numpy().tolist()
    if actual != [-1.0, 0.0, 1.0]:
        raise RuntimeError(f"Unexpected tf.sign result on GPU {index}: {actual}")
    print(f"TensorFlow GPU {index} libdevice preflight passed", flush=True)
PY

srun python -u -m \
  eegproc.deep_learning.joint_architectures.SICModelv11.sic_calibration_only_train \
  --models-config "src/eegproc/deep_learning/joint_architectures/SICModelv11/calibration_hps/sic_calibration_model.json" \
  --calibration-hyperparameters-file "src/eegproc/deep_learning/joint_architectures/SICModelv11/calibration_hps/sic_calibration_full_grid.json" \
  --raw-eeg-npy "datasets/dreamer_eeg.npy" \
  --raw-labels-npy "datasets/dreamer_labels.npy" \
  --classification-level trial \
  --calibration-level 6 3 \
  --calibration-selection-shots 6 \
  --selection-metric balanced_accuracy \
  --calibration-verbose \
  --calibration-print-every-n-epochs 1 \
  --n-jobs 1 \
  --gpu-ids 0 \
  --cpus-per-worker 2
