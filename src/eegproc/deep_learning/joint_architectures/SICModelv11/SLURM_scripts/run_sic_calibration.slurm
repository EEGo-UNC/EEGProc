#!/bin/bash
#SBATCH --job-name=sic-calibration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=l40-gpu
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

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES:-not-set}"
nvidia-smi -L

srun python -u -m \
  eegproc.deep_learning.joint_architectures.SICModelv11.sic_calibration_only_train \
  --models-config "src/eegproc/deep_learning/joint_architectures/SICModelv11/calibration_hps/sic_calibration_model.json" \
  --calibration-hyperparameters-file "src/eegproc/deep_learning/joint_architectures/SICModelv11/calibration_hps/sic_calibration_full_grid.json" \
  --raw-eeg-npy "datasets/dreamer_eeg.npy" \
  --raw-labels-npy "datasets/dreamer_labels.npy" \
  --classification-level trial \
  --calibration-level 3 6 \
  --calibration-level 6 3 \
  --calibration-level 9 2 \
  --calibration-level 12 3 \
  --calibration-selection-shots 9 \
  --selection-metric balanced_accuracy \
  --calibration-verbose \
  --calibration-print-every-n-epochs 5 \
  --n-jobs 2 \
  --gpu-ids 0 1 \
  --cpus-per-worker 4
