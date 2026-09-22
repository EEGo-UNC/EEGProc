#!/bin/bash
#SBATCH --job-name=cfo_ablations_1599318
#SBATCH --output=cfo_ablations_1599318_%A_%a.out
#SBATCH --error=cfo_ablations_1599318_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0

set -euo pipefail

# One GPU per subject; four matched arms run sequentially in one study:
#   target_latent: target + latent
#   base:          target + latent + decoded + physiology
#   typicality:    target + latent + decoded + physiology + typicality
#   typicality_no_physiology: target + latent + decoded + typicality
# All arms receive the same diagnostics, references, and eligible trials.
# Default: subject 0, all eligible trials. After validating that user:
#   sbatch --array=0-22%4 src/eegproc/model_explainability/slurm/run_cfo_ablations_arousal_1599318.sh
# Optional: TRIAL_IDS="8", MAX_STEPS=5, OUT_ROOT, RESUME=1.
# Local command preview (no GPU, module loads, or result writes):
#   DRY_RUN=1 VENV_DIR=venv bash src/eegproc/model_explainability/slurm/run_cfo_ablations_arousal_1599318.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
find_project_dir() {
    local start candidate
    for start in "${SLURM_SUBMIT_DIR:-}" "$SCRIPT_DIR"; do
        [[ -n "$start" ]] || continue
        candidate="$(cd -- "$start" 2>/dev/null && pwd)" || continue
        while [[ "$candidate" != / ]]; do
            if [[ -d "$candidate/src/eegproc" ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
            candidate="$(dirname -- "$candidate")"
        done
    done
    return 1
}
if [[ -n "${PROJECT_DIR:-}" ]]; then
    PROJECT_DIR="$(cd -- "$PROJECT_DIR" && pwd)"
elif ! PROJECT_DIR="$(find_project_dir)"; then
    echo "ERROR: submit from inside EEGProc or set PROJECT_DIR." >&2
    exit 2
fi
cd "$PROJECT_DIR"

VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_DIR/runs/full/sic_v15_arousal_scale64/DREAMER/arousal/suite_1599318.before-pull/all_subjects/full_run_v15_arousal_scale64_20260918_155429/configuration_0001}"
MODELS_JSON="${MODELS_JSON:-$CONFIG_DIR/loso_zero_shot_models.json}"
MODEL_DIR="${MODEL_DIR:-$(dirname -- "$MODELS_JSON")/loso_zero_shot_models}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
TRIALS_NPZ="${TRIALS_NPZ:-}"
SUBJECT_ID="${SLURM_ARRAY_TASK_ID:-${SUBJECT_ID:-0}}"
RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/runs/counterfactuals/arousal_suite_1599318_ablations_${RUN_ID}}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"

if [[ ! "$SUBJECT_ID" =~ ^([0-9]|1[0-9]|2[0-2])$ ]]; then
    echo "ERROR: subject/array index must be 0-22." >&2
    exit 2
fi
for boolean in "$DRY_RUN" "$RESUME"; do
    if [[ "$boolean" != 0 && "$boolean" != 1 ]]; then
        echo "ERROR: DRY_RUN and RESUME must be 0 or 1." >&2
        exit 2
    fi
done
if [[ "$DRY_RUN" != 1 ]]; then
    module purge
    module load python/3.12.4
    module load cuda/12.9
    module load cudnn/9.11.0
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: Python environment not found at $VENV_DIR." >&2
    exit 2
fi
OUT_DIR="$OUT_ROOT/fold_$(printf '%02d' "$SUBJECT_ID")"
if [[ -e "$OUT_DIR" && "$RESUME" != 1 ]]; then
    echo "ERROR: output exists; use a new OUT_ROOT or RESUME=1: $OUT_DIR" >&2
    exit 2
fi

# Resolve the held-out model from the actual manifest, not a guessed fold ID.
"$VENV_DIR/bin/python" - "$MODELS_JSON" "$MODEL_DIR" "$SUBJECT_ID" <<'PY'
import json
from pathlib import Path
import sys

manifest, directory, subject = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
payload = json.loads(manifest.read_text())
entries = payload["models"] if isinstance(payload, dict) else payload
matches = [entry for entry in entries if int(entry["target_subject"]) == subject]
if len(matches) != 1 or matches[0].get("stage") != "zero_shot_source_model":
    raise SystemExit(f"ERROR: expected one zero-shot checkpoint for subject {subject}")
entry = matches[0]
checkpoint = directory / Path(entry.get("filename", entry["path"])).name
if not checkpoint.is_file() or checkpoint.suffix != ".keras":
    raise SystemExit(f"ERROR: missing checkpoint: {checkpoint}")
if subject in entry.get("source_subject_ids", []):
    raise SystemExit("ERROR: held-out subject appears in checkpoint source subjects")
print(f"Checkpoint: {checkpoint}")
PY

if [[ -n "$TRIALS_NPZ" ]]; then
    if [[ ! -s "$TRIALS_NPZ" ]]; then
        echo "ERROR: prepared dataset missing: $TRIALS_NPZ" >&2
        exit 2
    fi
    DATA_ARGUMENTS=(--trials-npz "$TRIALS_NPZ")
else
    for required_file in "$EEG_PATH" "$LABELS_PATH"; do
        if [[ ! -s "$required_file" ]]; then
            echo "ERROR: raw data missing: $required_file" >&2
            exit 2
        fi
    done
    DATA_CONFIG="$("$VENV_DIR/bin/python" - "$EEG_PATH" "$LABELS_PATH" <<'PY'
import json
import sys
print(json.dumps(dict(raw_eeg_npy=sys.argv[1], raw_labels_npy=sys.argv[2],
    dataset="dreamer", label_dimension="arousal", fs=128, window_sec=1,
    window_overlap=0, window_normalization="global_rms",
    label_threshold_mode="global", median_label=3)))
PY
)"
    DATA_ARGUMENTS=(--data-loader eegproc.model_explainability.model_agnostic.sic_adapter:load_sic_raw_trials --data-config "$DATA_CONFIG")
fi

COMMAND=("$VENV_DIR/bin/python" -m eegproc.model_explainability.typicality.runner
    --artifact-mode "${ARTIFACT_MODE:-paper}"
    --models-json "$MODELS_JSON" --model-dir "$MODEL_DIR"
    --model-module eegproc.deep_learning.joint_architectures.SICModelv15.sic_model
    "${DATA_ARGUMENTS[@]}" --task arousal --subjects "$SUBJECT_ID"
    --include-target-latent --include-typicality-no-physiology --decoder-mode joint
    --typicality-representation vc_trial_embedding
    --target-loss-component "${TARGET_LOSS_COMPONENT:-confidence}"
    --target-probability "${TARGET_PROBABILITY:-0.80}"
    --target-weight "${TARGET_WEIGHT:-1.0}" --latent-weight "${LATENT_WEIGHT:-0.1}"
    --decoded-weight "${DECODED_WEIGHT:-0.1}" --physiological-weight "${PHYSIOLOGICAL_WEIGHT:-1.0}"
    --typicality-weight "${TYPICALITY_WEIGHT:-1.0}"
    --typicality-quantile "${TYPICALITY_QUANTILE:-0.95}" --variance-floor "${VARIANCE_FLOOR:-1e-6}"
    --learning-rate "${LEARNING_RATE:-0.01}" --learning-rate-decay "${LEARNING_RATE_DECAY:-1.0}"
    --max-steps "${MAX_STEPS:-200}" --gradient-clip-norm "${GRADIENT_CLIP_NORM:-5.0}"
    --min-gradient-norm "${MIN_GRADIENT_NORM:-1e-6}" --low-gradient-patience "${LOW_GRADIENT_PATIENCE:-5}"
    --typicality-improvement-patience "${TYPICALITY_IMPROVEMENT_PATIENCE:-10}"
    --typicality-min-delta "${TYPICALITY_MIN_DELTA:-1e-6}"
    --physiological-tolerance "${PHYSIOLOGICAL_TOLERANCE:-1e-8}"
    --physiology-quantile "${PHYSIOLOGY_QUANTILE:-0.95}"
    --physiology-required-fraction "${PHYSIOLOGY_REQUIRED_FRACTION:-0.95}"
    --fs 128 --seed "${SEED:-42}" --log-every "${LOG_EVERY:-10}" --out-dir "$OUT_DIR")
case "${STOP_ON_SUCCESS:-1}" in
    1) COMMAND+=(--stop-on-success) ;;
    0) COMMAND+=(--no-stop-on-success) ;;
    *) echo "ERROR: STOP_ON_SUCCESS must be 0 or 1." >&2; exit 2 ;;
esac
if [[ -n "${TRIAL_IDS:-}" ]]; then
    read -r -a SELECTED_TRIALS <<< "$TRIAL_IDS"
    for trial in "${SELECTED_TRIALS[@]}"; do
        if [[ ! "$trial" =~ ^[0-9]+$ ]]; then
            echo "ERROR: TRIAL_IDS must be a space-separated list of nonnegative integers." >&2
            exit 2
        fi
    done
    COMMAND+=(--trial-ids "${SELECTED_TRIALS[@]}")
fi
if [[ "$RESUME" == 1 ]]; then COMMAND+=(--resume); fi

echo "Suite 1599318 / configuration 0001 / arousal / subject $SUBJECT_ID"
echo "Arms: target_latent, base (full without typicality), typicality (full), typicality_no_physiology"
echo "Output: $OUT_DIR"
echo "Artifacts: ${ARTIFACT_MODE:-paper} (paper omits EEG, full latents, PSDs and optimizer snapshots)"
if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
    exit 0
fi

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Saved recurrent models require CUDA libdevice during Keras deserialization.
CUDA_ROOT="${EBROOTCUDA:-${CUDA_HOME:-}}"
if [[ -z "$CUDA_ROOT" ]] && command -v nvcc >/dev/null 2>&1; then
    CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi
LIBDEVICE_PATH=""
for cuda_candidate in "$CUDA_ROOT" "$VENV_DIR/lib/python3.12/site-packages/nvidia" /usr/local/cuda /opt/cuda; do
    [[ -d "$cuda_candidate" ]] || continue
    LIBDEVICE_PATH="$(find "$cuda_candidate" -type f -path '*/nvvm/libdevice/libdevice.10.bc' -print -quit 2>/dev/null || true)"
    [[ -z "$LIBDEVICE_PATH" ]] || break
done
if [[ -z "$LIBDEVICE_PATH" ]]; then
    echo "ERROR: CUDA libdevice.10.bc is missing after loading cuda/12.9." >&2
    exit 2
fi
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_cuda_data_dir=${LIBDEVICE_PATH%/nvvm/libdevice/libdevice.10.bc}"

"$VENV_DIR/bin/python" - <<'PY'
import tensorflow as tf
from eegproc.model_explainability.typicality.runner import build_parser
required = {"include_target_latent", "include_typicality_no_physiology", "artifact_mode"}
if not required.issubset({action.dest for action in build_parser()._actions}):
    raise SystemExit("ERROR: sync the paper-artifact runner updates before submitting")
if len(tf.config.list_physical_devices("GPU")) != 1:
    raise SystemExit("ERROR: expected exactly one visible GPU")
with tf.device("/GPU:0"):
    assert tf.sign(tf.constant([-1.0, 0.0, 1.0])).numpy().tolist() == [-1.0, 0.0, 1.0]
print("GPU and four-arm runner preflight passed", flush=True)
PY

"${COMMAND[@]}"

# Completion is about saved attempts, not whether a counterfactual succeeded.
"$VENV_DIR/bin/python" - "$OUT_DIR" "$SUBJECT_ID" <<'PY'
import json
from pathlib import Path
import sys
from eegproc.model_explainability.typicality.artifacts import completed_attempt
root, subject = Path(sys.argv[1]), int(sys.argv[2])
study = json.loads((root / "study.json").read_text())
fold = json.loads((root / f"subject_{subject}/fold.json").read_text())
if study["objectives"] != ["target_latent", "base", "typicality", "typicality_no_physiology"]:
    raise SystemExit("ERROR: expected all four ablation arms")
if fold["status"] != "completed" or fold.get("n_optimization_errors"):
    raise SystemExit("ERROR: fold has incomplete or failed optimization attempts")
for trial in fold["eligible_trial_ids"]:
    for objective in study["objectives"]:
        if not completed_attempt(root / f"subject_{subject}/trial_{trial}" / objective):
            raise SystemExit(f"ERROR: missing completed trial {trial}, arm {objective}")
print(f"Completed {len(fold['eligible_trial_ids'])} eligible trials in each of four arms")
if not fold["eligible_trial_ids"]:
    print("No eligible trials: no counterfactual optimization was performed for this subject")
print(f"Comparison tables: {root / 'report'}")
PY
