#!/bin/bash
#SBATCH --job-name=valence-cfx
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-22%4
#SBATCH --output=valence-cfx-%A_%a.out
set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: sbatch [--array=SUBJECT] $0 MODELS_JSON RESULTS_ROOT [runner options]" >&2
    exit 2
fi
models_json="$1"
results_root="$2"
shift 2

subject="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
[[ "$subject" =~ ^([0-9]|1[0-9]|2[0-2])$ ]] || { echo "Subject must be 0–22." >&2; exit 2; }
[[ -r src/eegproc/model_explainability/typicality/runner.py ]] || {
    echo "Submit from the root of the EEGProc checkout containing the counterfactual runner." >&2
    exit 2
}
[[ -s "$models_json" ]] || { echo "Missing model manifest: $models_json" >&2; exit 2; }

module purge
module load python/3.12.4 cuda/12.9 cudnn/9.11.0
venv_dir="${VENV_DIR:-$PWD/venv312}"
python="$venv_dir/bin/python"
[[ -x "$python" ]] || { echo "Set VENV_DIR to the EEGProc Python environment." >&2; exit 2; }
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg TF_GPU_ALLOCATOR=cuda_malloc_async
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

cuda_root="${EBROOTCUDA:-${CUDA_HOME:-}}"
if [[ -z "$cuda_root" ]] && command -v nvcc >/dev/null; then
    cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi
[[ -n "$cuda_root" && -f "$cuda_root/nvvm/libdevice/libdevice.10.bc" ]] || {
    echo "CUDA libdevice.10.bc is missing after loading cuda/12.9." >&2
    exit 2
}
export CUDA_HOME="$cuda_root" CUDA_PATH="$cuda_root"
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_cuda_data_dir=$cuda_root"

data_config="$("$python" - "$models_json" "$subject" <<'PY'
import json
from pathlib import Path
import sys

manifest, subject = Path(sys.argv[1]), int(sys.argv[2])
models = json.loads(manifest.read_text())["models"]
matches = [entry for entry in models if entry["target_subject"] == subject]
if len(matches) != 1:
    raise SystemExit(f"Expected one checkpoint for subject {subject}")
checkpoint = Path(matches[0]["path"])
if not checkpoint.is_file():
    raise SystemExit(f"Missing checkpoint: {checkpoint}")
task = json.loads((checkpoint.parents[1] / "task.json").read_text())
for key in ("eeg", "labels"):
    if not Path(task[key]).is_file():
        raise SystemExit(f"Missing original training data: {task[key]}")
print(json.dumps(dict(raw_eeg_npy=task["eeg"], raw_labels_npy=task["labels"],
                      dataset="dreamer", label_dimension="valence", fs=128,
                      window_sec=1, window_overlap=0, window_normalization="global_rms",
                      label_threshold_mode="global", median_label=3)))
PY
)"
printf -v fold 'fold_%02d' "$subject"
run_dir="$results_root/run_${SLURM_ARRAY_JOB_ID:?Missing Slurm array job ID}"
exec "$python" -m eegproc.model_explainability.typicality.runner \
    --task valence --models-json "$models_json" --subjects "$subject" \
    --directions 0_to_1 1_to_0 --target-probability "${VALENCE_TARGET_PROBABILITY:-0.50}" \
    --data-loader eegproc.model_explainability.model_agnostic.sic_adapter:load_sic_raw_trials \
    --data-config "$data_config" --out-dir "$run_dir/valence/$fold" "$@"
