"""Scheduler adapters; training and scheduler dependencies stay outside the core."""
from typing import Any
from pathlib import Path


def slurm_script(run_id: str, campaign: dict[str, Any]) -> str:
    """Render the legacy Longleaf adapter or an explicitly supplied template."""
    launcher = campaign.get("launcher", {"type": "longleaf"})
    kind = launcher.get("type", "longleaf")
    if kind == "longleaf":
        return longleaf_script(run_id, campaign)
    if kind != "template":
        raise ValueError(f"Unknown launcher type {kind!r}.")
    path = Path(launcher["template"])
    if not path.is_absolute():
        path = Path(campaign.get("_campaign_dir", ".")) / path
    text = path.read_text(encoding="utf-8")
    for token, value in {
        "RUN_ID": run_id,
        "EXPERIMENT_DIR": campaign.get("experiment_dir", "."),
        "TRAINING_SCRIPT": campaign.get("training_script", "train.py"),
    }.items():
        text = text.replace("{{" + token + "}}", str(value))
    return text

def longleaf_script(run_id: str, campaign: dict[str, Any]) -> str:
    cluster = campaign["longleaf"]
    experiment_dir = campaign.get("experiment_dir", "experiments/dreamer_3dcnn")
    training_script = campaign.get(
        "training_script", "experiments/dreamer_3dcnn/train.py"
    )
    return f'''#!/bin/bash
#SBATCH --job-name={run_id[:40]}
#SBATCH --output={experiment_dir}/runs/{run_id}/slurm-%j.out
#SBATCH --error={experiment_dir}/runs/{run_id}/slurm-%j.err
#SBATCH --partition={cluster["partition"]}
#SBATCH --qos={cluster["qos"]}
#SBATCH --gres=gpu:{cluster["gpus"]}
#SBATCH --cpus-per-task={cluster["cpus"]}
#SBATCH --mem={cluster["memory"]}
#SBATCH --time={cluster["time"]}

set -euo pipefail
module purge
module load {cluster["python_module"]}
module load {cluster["cuda_module"]}
module load {cluster["cudnn_module"]}

PROJECT_DIR="${{PROJECT_DIR:-$HOME/EEGProc}}"
VENV_DIR="${{VENV_DIR:-$PROJECT_DIR/venv312}}"
DATA_DIR="${{DATA_DIR:-$PROJECT_DIR/datasets}}"
RUN_DIR="$PROJECT_DIR/{experiment_dir}/runs/{run_id}"

cd "$PROJECT_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Missing virtual environment: $VENV_DIR" >&2
    echo "Create it and install this checkout with the deep-learning extra." >&2
    exit 2
fi
if [[ ! -f "$DATA_DIR/dreamer_eeg.npy" || ! -f "$DATA_DIR/dreamer_labels.npy" ]]; then
    echo "Missing DREAMER arrays in $DATA_DIR" >&2
    exit 2
fi

source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR=cuda_malloc_async
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"

# TensorFlow/XLA does not automatically discover the libdevice shipped by
# Longleaf's CUDA module. Resolve it from the active module (or the Python CUDA
# packages) and pass its CUDA root explicitly before TensorFlow starts.
MODULE_CUDA_ROOT=""
if [[ -n "${{EBROOTCUDA:-}}" && -d "$EBROOTCUDA" ]]; then
    MODULE_CUDA_ROOT="$EBROOTCUDA"
elif [[ -n "${{CUDA_HOME:-}}" && -d "$CUDA_HOME" ]]; then
    MODULE_CUDA_ROOT="$CUDA_HOME"
elif command -v nvcc >/dev/null 2>&1; then
    MODULE_CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi

find_libdevice() {{
    local roots=()
    [[ -n "$MODULE_CUDA_ROOT" && -d "$MODULE_CUDA_ROOT" ]] && roots+=("$MODULE_CUDA_ROOT")
    [[ -d "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" ]] && roots+=("$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia")
    [[ -d /usr/local/cuda ]] && roots+=("/usr/local/cuda")
    [[ -d /opt/cuda ]] && roots+=("/opt/cuda")
    [[ ${{#roots[@]}} -eq 0 ]] && return 0
    find "${{roots[@]}}" -type f -path "*/nvvm/libdevice/libdevice.10.bc" -print -quit 2>/dev/null || true
}}

LIBDEVICE_PATH="$(find_libdevice)"
if [[ -z "$LIBDEVICE_PATH" || ! -f "$LIBDEVICE_PATH" ]]; then
    echo "Unable to locate nvvm/libdevice/libdevice.10.bc." >&2
    echo "Loaded CUDA root: ${{MODULE_CUDA_ROOT:-<none>}}" >&2
    echo "Try: python -m pip install nvidia-cuda-nvcc-cu12" >&2
    exit 2
fi
CUDA_XLA_ROOT="${{LIBDEVICE_PATH%/nvvm/libdevice/libdevice.10.bc}}"
export XLA_FLAGS="${{XLA_FLAGS:+$XLA_FLAGS }}--xla_gpu_cuda_data_dir=$CUDA_XLA_ROOT"
[[ -n "$MODULE_CUDA_ROOT" ]] && export CUDA_HOME="$MODULE_CUDA_ROOT" CUDA_PATH="$MODULE_CUDA_ROOT"

echo "CUDA XLA root: $CUDA_XLA_ROOT"
echo "libdevice: $LIBDEVICE_PATH"

python {training_script} \\
    --config "$RUN_DIR/config.json" \\
    --data-dir "$DATA_DIR" \\
    --output-dir "$RUN_DIR"
'''


