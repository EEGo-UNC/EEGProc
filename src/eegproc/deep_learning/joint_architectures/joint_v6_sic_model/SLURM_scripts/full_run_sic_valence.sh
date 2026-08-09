#!/bin/bash
#SBATCH --job-name=sic_valence
#SBATCH --output=sic_valence_%j.out
#SBATCH --error=sic_valence_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="$HOME/EEGProc"
VENV_DIR="$PROJECT_DIR/venv312"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-60}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-25}"
USE_SUBJECT_ADV_UPPERBOUND="${USE_SUBJECT_ADV_UPPERBOUND:-false}"
SUBJECT_ADV_UPPERBOUND="${SUBJECT_ADV_UPPERBOUND:-2.8}"
SUBJECT_ADV_RECOVERY_MAX_STEPS="${SUBJECT_ADV_RECOVERY_MAX_STEPS:-10}"
CALIBRATION_UNFREEZE_LAYERS="${CALIBRATION_UNFREEZE_LAYERS:-2}"
PREDICTION_LATENT_SAMPLES="${PREDICTION_LATENT_SAMPLES:-20}"

cd "$PROJECT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if command -v flock >/dev/null 2>&1; then
    (
        flock -x 9
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
    ) 9>"$PROJECT_DIR/.venv312_install.lock"
else
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi

MODULE_CUDA_ROOT=""
if [[ -n "${EBROOTCUDA:-}" && -d "${EBROOTCUDA}" ]]; then
    MODULE_CUDA_ROOT="$EBROOTCUDA"
elif [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
    MODULE_CUDA_ROOT="$CUDA_HOME"
elif command -v nvcc >/dev/null 2>&1; then
    MODULE_CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi
find_libdevice() {
    local roots=()
    [[ -n "$MODULE_CUDA_ROOT" && -d "$MODULE_CUDA_ROOT" ]] && roots+=("$MODULE_CUDA_ROOT")
    [[ -d "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" ]] && roots+=("$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia")
    [[ -d /usr/local/cuda ]] && roots+=("/usr/local/cuda")
    [[ -d /opt/cuda ]] && roots+=("/opt/cuda")
    [[ ${#roots[@]} -eq 0 ]] && return 0
    find "${roots[@]}" -type f -path "*/nvvm/libdevice/libdevice.10.bc" -print -quit 2>/dev/null || true
}
LIBDEVICE_PATH="$(find_libdevice)"
if [[ -z "$LIBDEVICE_PATH" ]]; then
    if command -v flock >/dev/null 2>&1; then
        (
            flock -x 9
            python -m pip install --upgrade nvidia-cuda-nvcc-cu12
        ) 9>"$PROJECT_DIR/.venv312_install.lock"
    else
        python -m pip install --upgrade nvidia-cuda-nvcc-cu12
    fi
    LIBDEVICE_PATH="$(find_libdevice)"
fi
if [[ -z "$LIBDEVICE_PATH" || ! -f "$LIBDEVICE_PATH" ]]; then
    echo "ERROR: Unable to locate libdevice.10.bc."
    exit 1
fi
CUDA_XLA_ROOT="${LIBDEVICE_PATH%/nvvm/libdevice/libdevice.10.bc}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_XLA_ROOT}"
if [[ -n "$MODULE_CUDA_ROOT" ]]; then
    export CUDA_HOME="$MODULE_CUDA_ROOT"
    export CUDA_PATH="$MODULE_CUDA_ROOT"
fi

MODEL_CONFIG="$(python - <<PY
import json
print(json.dumps({
    "optimizer_name": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 5e-5,
    "t_down": 2,
    "temporal_pool_sizes": [2],
    "gcn_units": [32],
    "gcn_dropout": 0.10,
    "gcn_activation": "relu",
    "gcn_use_batch_norm": False,
    "spectral_gru_units": 384,
    "spectral_gru_dropout": 0.20,
    "mi_n_neighbors": 3,
    "mi_random_state": 42,
    "mi_zero_diagonal": False,
    "mi_band_reduction": "mean",
    "mi_max_observations": 15000,
    "bilstm_units": 128,
    "n_bilstm_layers": 1,
    "bilstm_dropout": 0.30,
    "z_dim": 64,
    "classification_hidden_units": [128, 64],
    "classification_dropout": 0.20,
    "vc_loss_weight": 1.0,
    "vc_alpha": 1.0,
    "vc_beta": 0.5,
    "vc_gamma": 0.0,
    "vc_lambda": 0.0,
    "update_vc_discriminator": False,
    "vae_loss_weight": 0.10,
    "vae_beta": 0.05,
    "decoder_dropout": 0.10,
    "use_subject_adversarial": True,
    "subject_adversarial_weight": 0.8,
    "subject_loss_weight": 1.0,
    "subject_hidden_units": 64,
    "subject_dropout": 0.0,
    "use_subject_adversarial_upperbound": str("${USE_SUBJECT_ADV_UPPERBOUND}").lower() == "true",
    "subject_adversarial_loss_upperbound": float("${SUBJECT_ADV_UPPERBOUND}"),
    "subject_adversarial_recovery_max_steps": int("${SUBJECT_ADV_RECOVERY_MAX_STEPS}"),
    "calibration_unfreeze_layers": int("${CALIBRATION_UNFREEZE_LAYERS}"),
    "calibration_use_vc_target": True,
    "use_class_weight": False
}))
PY
)"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Model: SIC (Subject Invariant Calibrator)"
echo "Protocol: source-independent pretraining + 3-fold six-trial calibration"
echo "Source epochs: $SOURCE_EPOCHS"
echo "Calibration epochs: $CALIBRATION_EPOCHS"
echo "Calibration unfreeze layers: $CALIBRATION_UNFREEZE_LAYERS"
echo "Adversarial upper bound enabled: $USE_SUBJECT_ADV_UPPERBOUND"
echo "Adversarial upper bound: $SUBJECT_ADV_UPPERBOUND"
echo "Prediction latent samples: $PREDICTION_LATENT_SAMPLES"
python --version
nvidia-smi

python - <<'PY'
import inspect
import tensorflow as tf
from src.eegproc.deep_learning.joint_architectures.sic import sic_model
from src.eegproc.deep_learning import cross_val

print("TensorFlow:", tf.__version__)
print("SIC module:", sic_model.__file__)
print("SIC builder API:", sic_model.SIC_BUILDER_API_VERSION)
print("SIC builder:", inspect.signature(sic_model.build_sic_model))
print("Calibration CV:", inspect.signature(cross_val.subject_calibration_cv))
if sic_model.SIC_BUILDER_API_VERSION < 2:
    raise RuntimeError("Stale SIC model; expected builder API >= 2.")
gpus = tf.config.list_physical_devices("GPU")
print("Available GPUs:", gpus)
if len(gpus) < 4:
    raise RuntimeError(f"Expected 4 allocated GPUs, saw {len(gpus)}.")
PY

python -m src.eegproc.deep_learning.joint_architectures.sic.sic_model_train \
    --raw-eeg-npy datasets/remove_gamma/dreamer_eeg.npy \
    --raw-labels-npy datasets/remove_gamma/dreamer_labels.npy \
    --label-dimension valence \
    --classification-level trial \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir runs/sic/DREAMER/valence \
    --run-name dreamer_valence_sic \
    --source-epochs "$SOURCE_EPOCHS" \
    --source-batch-size 8 \
    --calibration-epochs "$CALIBRATION_EPOCHS" \
    --calibration-batch-size 6 \
    --calibration-trials 6 \
    --calibration-folds 3 \
    --calibration-learning-rate 0.0001 \
    --calibration-optimizer adamw \
    --calibration-weight-decay 0.0 \
    --calibration-seed 42 \
    --decision-threshold 0.5 \
    --prediction-latent-samples "$PREDICTION_LATENT_SAMPLES" \
    --latent-sampling-seed 42 \
    --n-jobs 4 \
    --gpu-ids 0 1 2 3 \
    --cpus-per-worker 2 \
    --verbose 2 \
    --seed 42 \
    --label-threshold-mode global \
    --median-label 3 \
    --window-sec 4.0 \
    --window-overlap 0.0 \
    --window-normalization global_rms \
    --hyperparameters-json "$MODEL_CONFIG"
