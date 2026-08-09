#!/bin/bash
#SBATCH --job-name=smoke_sic_valence
#SBATCH --output=smoke_sic_valence_%j.out
#SBATCH --error=smoke_sic_valence_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00

set -euo pipefail

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

PROJECT_DIR="$HOME/EEGProc"
VENV_DIR="$PROJECT_DIR/venv312"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-100}"
CALIBRATION_EPOCHS="${CALIBRATION_EPOCHS:-15}"
USE_SUBJECT_ADV_UPPERBOUND="${USE_SUBJECT_ADV_UPPERBOUND:-true}"
SUBJECT_ADV_UPPERBOUND="${SUBJECT_ADV_UPPERBOUND:-0.15}"
SUBJECT_ADV_RECOVERY_MAX_STEPS="${SUBJECT_ADV_RECOVERY_MAX_STEPS:-5}"
CALIBRATION_UNFREEZE_LAYERS="${CALIBRATION_UNFREEZE_LAYERS:-2}"

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
    "gcn_dropout": 0.2,
    "gcn_activation": "relu",
    "gcn_use_batch_norm": False,
    "spectral_gru_units": 384,
    "spectral_gru_dropout": 0.2,
    "mi_n_neighbors": 3,
    "mi_random_state": 42,
    "mi_zero_diagonal": False,
    "mi_band_reduction": "mean",
    "mi_max_observations": 15000,
    "bilstm_units": 128,
    "n_bilstm_layers": 1,
    "bilstm_dropout": 0.4,
    "z_dim": 128,
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
echo "Source epochs: $SOURCE_EPOCHS"
echo "Calibration epochs: $CALIBRATION_EPOCHS"
echo "Adversarial upper bound enabled: $USE_SUBJECT_ADV_UPPERBOUND"
echo "Adversarial upper bound: $SUBJECT_ADV_UPPERBOUND"
echo "Calibration unfreeze layers: $CALIBRATION_UNFREEZE_LAYERS"
python --version
nvidia-smi

# Runtime architecture/training preflight. This is intentionally stronger than
# an import test: it verifies the old GCNMTLDecoder, subject-adversary takeover,
# and that calibration does not modify the frozen representation.
python - <<'PY'
import numpy as np
import tensorflow as tf

from src.eegproc.deep_learning.joint_architectures.sic.sic_model import (
    SIC_BUILDER_API_VERSION,
    SICModel,
    build_sic_model,
)
from src.eegproc.deep_learning.unsupervised.GNN.GCNMTL import GCNMTLDecoder

print("TensorFlow:", tf.__version__)
print("SIC builder API:", SIC_BUILDER_API_VERSION)
if SIC_BUILDER_API_VERSION < 2:
    raise RuntimeError("Stale SIC model; expected builder API >= 2.")

trial_shape = (3, 8, 42)
rng = np.random.default_rng(42)
x = rng.normal(size=(4, *trial_shape)).astype(np.float32)
y = np.asarray([0, 1, 0, 1], dtype=np.int32)
subject_ids = np.asarray([10, 11, 10, 11], dtype=np.int32)

model = build_sic_model(
    input_shape=trial_shape,
    training_features=x,
    training_labels=y,
    training_subject_ids=subject_ids,
    training_trial_ids=np.arange(4),
    classification_level="trial",
    n_classes=2,
    n_channels=14,
    n_bands=3,
    t_down=2,
    temporal_pool_sizes=(2,),
    gcn_units=(4,),
    spectral_gru_units=8,
    spectral_gru_dropout=0.0,
    mi_max_observations=64,
    bilstm_units=4,
    z_dim=4,
    classification_hidden_units=(4, 2),
    classification_dropout=0.0,
    vc_loss_weight=1.0,
    vc_alpha=1.0,
    vc_beta=0.1,
    vae_loss_weight=0.1,
    vae_beta=0.01,
    use_subject_adversarial=True,
    subject_adversarial_weight=0.5,
    subject_loss_weight=1.0,
    subject_hidden_units=4,
    use_subject_adversarial_upperbound=True,
    subject_adversarial_loss_upperbound=0.1,
    subject_adversarial_recovery_max_steps=2,
    calibration_unfreeze_layers=2,
)
if not isinstance(model, SICModel):
    raise RuntimeError(type(model))
if not isinstance(model.decoder, GCNMTLDecoder):
    raise RuntimeError(
        f"SIC must use old GCNMTLDecoder; got {type(model.decoder).__name__}."
    )

prepared = model.prepare_fit_inputs(x, subject_ids)
metrics = model.train_on_batch(prepared, y, return_dict=True)
print("Source train preflight metrics:", metrics)
if float(metrics.get("subject_takeover_fraction", 0.0)) <= 0.0:
    raise RuntimeError("Forced adversarial takeover did not activate.")
if float(metrics.get("subject_recovery_steps", 0.0)) <= 0.0:
    raise RuntimeError("Adversarial recovery loop did not run.")

# Restore normal source-evaluation semantics before calibration.
model.prepare_for_zero_shot_evaluation()
backbone_before = [w.numpy().copy() for w in model.graph_encoder.weights]
model.prepare_for_subject_calibration(
    learning_rate=1e-3,
    optimizer_name="adamw",
    weight_decay=0.0,
    unfreeze_layers=2,
)
trainable_names = [v.name for v in model.trainable_variables]
print("Calibration trainable variables:", trainable_names)
if any("mtl_" in name and "classifier" not in name for name in trainable_names):
    raise RuntimeError("MTL representation unexpectedly trainable in calibration.")
model.train_on_batch(x, y, return_dict=True)
backbone_after = [w.numpy().copy() for w in model.graph_encoder.weights]
if any(not np.array_equal(a, b) for a, b in zip(backbone_before, backbone_after)):
    raise RuntimeError("Frozen graph encoder changed during calibration.")

latent = model.get_latent_distribution(x)
recon = model.decode_latent(latent["z_mean"])
print("z_mean shape:", latent["z_mean"].shape)
print("reconstruction shape:", recon.shape)
print("SIC runtime preflight: PASS")
PY

python -m src.eegproc.deep_learning.joint_architectures.sic.sic_model_train \
    --raw-eeg-npy datasets/remove_gamma/dreamer_eeg.npy \
    --raw-labels-npy datasets/remove_gamma/dreamer_labels.npy \
    --label-dimension valence \
    --classification-level trial \
    --n-channels 14 \
    --n-bands 3 \
    --out-dir runs/smoke/sic/DREAMER/valence \
    --run-name dreamer_valence_sic_smoke \
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
    --prediction-latent-samples 3 \
    --latent-sampling-seed 42 \
    --max-subjects 2 \
    --n-jobs 2 \
    --gpu-ids 0 1 \
    --cpus-per-worker 2 \
    --verbose 2 \
    --seed 42 \
    --label-threshold-mode global \
    --median-label 3 \
    --window-sec 2.0 \
    --window-overlap 0.0 \
    --window-normalization global_rms \
    --hyperparameters-json "$MODEL_CONFIG"
