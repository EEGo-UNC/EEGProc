#!/bin/bash
#SBATCH --job-name=smoke_joint_v2_dreamer_arousal
#SBATCH --output=smoke_joint_v2_dreamer_arousal_%j.out
#SBATCH --error=smoke_joint_v2_dreamer_arousal_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

cd "$HOME/EEGProc"
source "$HOME/EEGProc/venv/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1


echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
python --version

nvidia-smi

python - <<'PY'
import site
import sys
import tensorflow as tf

print("Python executable:", sys.executable)
print("Virtual environment:", sys.prefix)
print("User site enabled:", site.ENABLE_USER_SITE)
print("TensorFlow version:", tf.__version__)
print("TensorFlow location:", tf.__file__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))
PY


python -m src.eegproc.deep_learning.joint_architectures.joint_v2_autoencoder_vc_train \
  --raw-eeg-npy src/eegproc/deep_learning/supervised/stsnet/data/dreamer_eeg.npy \
  --raw-labels-npy src/eegproc/deep_learning/supervised/stsnet/data/dreamer_labels.npy \
  --label-dimension arousal \
  --outer-subjects 1 \
  --inner-subjects 1 \
  --hyperparameters-json '{"epochs":[3,5],"batch_size":[16],"learning_rate":[0.001],"ae_loss_weight":[0.3],"vc_loss_weight":[0.7],"emb_dim":[16],"dropout":[0.2]}'
  