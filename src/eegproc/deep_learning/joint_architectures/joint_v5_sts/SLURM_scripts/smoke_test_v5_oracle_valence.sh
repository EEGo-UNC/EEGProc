#!/bin/bash
#SBATCH --job-name=v5_oracle_smoke
#SBATCH --output=v5_oracle_smoke_%j.out
#SBATCH --error=v5_oracle_smoke_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

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
    --oracle-test-epoch-selection \
    --oracle-metric accuracy \
    --oracle-every 1 \
    --classification-level window \
    --raw-eeg-npy datasets/remove_gamma/dreamer_eeg.npy \
    --raw-labels-npy datasets/remove_gamma/dreamer_labels.npy \
    --label-dimension valence \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir runs/smoke/joint_v5_sts/DREAMER/valence/oracle \
    --run-name dreamer_valence_joint_v5_oracle_smoke \
    --max-folds 2 \
    --n-jobs 1 \
    --outer-verbose 0 \
    --final-verbose 0 \
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
    --batch-size 4 \
    --mi-max-observations 15000 \
    --hyperparameters-json '{
        "epochs": [20],
        "optimizer": ["adamw"],
        "classification_learning_rate": [0.0001],
        "weight_decay": [0.00005],
        "gcn_units": [[32]],
        "gcn_dropout": [0.1],
        "gcn_use_batch_norm": [false],
        "spectral_gru_units": [384],
        "spectral_gru_dropout": [0.2],
        "mi_n_neighbors": [3],
        "mi_band_reduction": ["mean"],
        "mi_max_observations": [15000],
        "classification_hidden_units": [128],
        "classification_dropout": [0.2],
        "activation": ["relu"],
        "focal_gamma": [1.0],
        "focal_alpha": null,
        "use_class_weight": [false]
    }'
