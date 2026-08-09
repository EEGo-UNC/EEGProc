#!/bin/bash
#SBATCH --job-name=joint_v5_dreamer_valence
#SBATCH --output=joint_v5_dreamer_valence_%j.out
#SBATCH --error=joint_v5_dreamer_valence_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="$HOME/EEGProc"
VENV_DIR="$PROJECT_DIR/venv312"
cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

MODULE_CUDA_ROOT="${EBROOTCUDA:-${CUDA_HOME:-}}"
if [[ -n "$MODULE_CUDA_ROOT" ]]; then
    export CUDA_HOME="$MODULE_CUDA_ROOT"
    export CUDA_PATH="$MODULE_CUDA_ROOT"
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${MODULE_CUDA_ROOT}"
fi

python --version
nvidia-smi

python -m src.eegproc.deep_learning.joint_architectures.joint_v5_sts.joint_sts_model_train \
    --classification-level window \
    --validation-subjects 4 \
    --raw-eeg-npy datasets/remove_gamma/dreamer_eeg.npy \
    --raw-labels-npy datasets/remove_gamma/dreamer_labels.npy \
    --label-dimension valence \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir runs/joint_v5_sts/DREAMER/valence/window \
    --run-name dreamer_valence_joint_v5_window \
    --n-jobs 4 \
    --cpus-per-worker 3 \
    --outer-verbose 0 \
    --final-verbose 2 \
    --seed 42 \
    --label-threshold-mode global \
    --median-label 3 \
    --window-sec 1.0 \
    --window-overlap 0.0 \
    --window-normalization global_rms \
    --no-class-weight \
    --selection-level trial \
    --selection-metric accuracy \
    --decision-thresholds 0.5 \
    --threshold-selection-level trial \
    --threshold-selection-metric accuracy \
    --early-stopping-patience 30 \
    --early-stopping-min-delta 0.002 \
    --early-stopping-monitor val_accuracy \
    --early-stopping-mode max \
    --final-epoch-strategy median \
    --batch-size 4 \
    --mi-max-observations 50000 \
    --hyperparameters-json '{
        "epochs": [150],
        "optimizer": ["adamw"],
        "classification_learning_rate": [0.0001],
        "weight_decay": [0.00005],
        "gcn_units": [[32]],
        "gcn_dropout": [0.1],
        "gcn_activation": ["relu"],
        "gcn_use_batch_norm": [false],
        "spectral_gru_units": [384],
        "spectral_gru_dropout": [0.0],
        "mi_n_neighbors": [3],
        "mi_band_reduction": ["mean"],
        "mi_max_observations": [50000],
        "classification_hidden_units": [128],
        "classification_dropout": [0.3],
        "activation": ["relu"],
        "focal_gamma": [0.0],
        "focal_alpha": null,
        "use_class_weight": [false]
    }'
