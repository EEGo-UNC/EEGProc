#!/bin/bash
#SBATCH --job-name=joint_v2_amigos_arousal
#SBATCH --output=joint_v2_amigos_arousal_%j.out
#SBATCH --error=joint_v2_amigos_arousal_%j.err
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

cd ~/EEGProc
source venv/bin/activate

python -m pip install --no-cache-dir -r requirements.txt

export PYTHONUNBUFFERED=1

python -m src.eegproc.deep_learning.joint_architectures.joint_v2_autoencoder_vc_train \
  --raw-eeg-npy src/eegproc/deep_learning/joint_architectures/data/amigos_eeg.npy \
  --raw-labels-npy src/eegproc/deep_learning/joint_architectures/data/amigos_labels.npy \
  --label-dimension arousal \
  --outer-subjects 2 \
  --inner-subjects 1 \
  --hyperparameters-json '{"epochs":[3,5],"batch_size":[16],"learning_rate":[0.001],"ae_loss_weight":[0.3],"vc_loss_weight":[0.7],"emb_dim":[16],"dropout":[0.2]}'