#!/bin/bash
#SBATCH --job-name=decoder_recovery
#SBATCH --output=decoder_recovery_%A_%a.out
#SBATCH --error=decoder_recovery_%A_%a.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0

set -euo pipefail

# One held-out DREAMER subject per job; both ideas and matched controls.
# Start with --array=0. Expand to --array=0-22%2 after checking the first fold.
# TASK=valence (default) or arousal. DRY_RUN=1 only prints commands.
# Run from a checkout containing experiments/decoder_recovery.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"
cd "$PROJECT_DIR"
if [[ ! -d experiments/decoder_recovery || ! -d src/eegproc ]]; then
    echo "ERROR: PROJECT_DIR must be the EEGProc checkout containing these experiments." >&2
    exit 2
fi

TASK="${TASK:-valence}"
SUBJECT_ID="${SLURM_ARRAY_TASK_ID:-${SUBJECT_ID:-0}}"
N_SUBJECTS="${N_SUBJECTS:-23}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_MODULES="${SKIP_MODULES:-0}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
PYTHON="$VENV_DIR/bin/python"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
MODEL_CONFIG="${MODEL_CONFIG:-$PROJECT_DIR/experiments/decoder_recovery/model_config.json}"
RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/runs/decoder_recovery/$TASK/suite_$RUN_ID}"
FOLD_DIR="$OUT_ROOT/fold_$(printf '%02d' "$SUBJECT_ID")"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-10}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-10}"
IDENTIFIER_EPOCHS="${IDENTIFIER_EPOCHS:-20}"
DECODER_EPOCHS="${DECODER_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ISA_WEIGHT="${ISA_WEIGHT:-0.6}"
IDENTITY_WEIGHT="${IDENTITY_WEIGHT:-0.1}"
IDENTITY_LOSS="${IDENTITY_LOSS:-cross_entropy}"
CF_MAX_STEPS="${CF_MAX_STEPS:-100}"
CF_LIMIT="${CF_LIMIT:-all}"
RUN_COUNTERFACTUALS="${RUN_COUNTERFACTUALS:-1}"
PHYSIOLOGICAL_WEIGHT="${PHYSIOLOGICAL_WEIGHT:-0.0}"
TARGET_PROBABILITY="${TARGET_PROBABILITY:-0.8}"

if [[ "$TASK" != valence && "$TASK" != arousal ]]; then
    echo "ERROR: TASK must be valence or arousal." >&2; exit 2
fi
if [[ ! "$SUBJECT_ID" =~ ^[0-9]+$ || ! "$N_SUBJECTS" =~ ^[0-9]+$ ]] ||
   (( N_SUBJECTS < 7 || SUBJECT_ID >= N_SUBJECTS )); then
    echo "ERROR: require N_SUBJECTS >= 7 and a valid held-out subject index." >&2; exit 2
fi
for flag in "$DRY_RUN" "$SKIP_MODULES" "$REQUIRE_GPU" "$RUN_COUNTERFACTUALS"; do
    if [[ "$flag" != 0 && "$flag" != 1 ]]; then
        echo "ERROR: boolean options must be 0 or 1." >&2; exit 2
    fi
done
if [[ "$IDENTITY_LOSS" != cross_entropy && "$IDENTITY_LOSS" != kl ]]; then
    echo "ERROR: IDENTITY_LOSS must be cross_entropy or kl." >&2; exit 2
fi
if [[ "$DRY_RUN" != 1 && -e "$FOLD_DIR" ]]; then
    echo "ERROR: use a new OUT_ROOT/RUN_ID; fold output already exists: $FOLD_DIR" >&2; exit 2
fi
if [[ "$DRY_RUN" != 1 && "$SKIP_MODULES" != 1 ]]; then
    module purge
    module load python/3.12.4
    module load cuda/12.9
    module load cudnn/9.11.0
fi
if [[ "$DRY_RUN" != 1 ]]; then
    for path in "$PYTHON" "$EEG_PATH" "$LABELS_PATH" "$MODEL_CONFIG"; do
        if [[ ! -f "$path" ]]; then echo "ERROR: missing $path" >&2; exit 2; fi
    done
fi

# Compute four source-validation subjects reproducibly without reading EEG or
# labels. An explicit VALIDATION_SUBJECTS="1 2 3 4" overrides this selection.
META_PYTHON="$PYTHON"
if [[ "$DRY_RUN" == 1 && ! -x "$META_PYTHON" ]]; then META_PYTHON=python3; fi
if [[ -z "${VALIDATION_SUBJECTS:-}" ]]; then
    VALIDATION_SUBJECTS="$("$META_PYTHON" - "$SUBJECT_ID" "$N_SUBJECTS" "$SEED" <<'PY'
import random
import sys
subject, count, seed = map(int, sys.argv[1:])
print(*sorted(random.Random(seed + subject).sample([s for s in range(count) if s != subject], 4)))
PY
)"
fi
read -r -a VALIDATION_IDS <<< "$VALIDATION_SUBJECTS"

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED="$SEED"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-6}" TF_NUM_INTEROP_THREADS=2
if [[ -n "${EBROOTCUDA:-}" ]]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=$EBROOTCUDA"
fi

run() {
    printf 'COMMAND:'
    printf ' %q' "$@"
    printf '\n'
    if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi
}
if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p "$FOLD_DIR"
    export MPLCONFIGDIR="$FOLD_DIR/cache/matplotlib" XDG_CACHE_HOME="$FOLD_DIR/cache"
    "$PYTHON" - "$FOLD_DIR" "$EEG_PATH" "$LABELS_PATH" "$TASK" <<'PY'
import json
from pathlib import Path
import sys
output, eeg, labels, task = sys.argv[1:]
Path(output, "loader.json").write_text(json.dumps(dict(
    eeg_path=eeg, labels_path=labels, dataset="dreamer", label_dimension=task,
    window_size_sec=1.0, fs=128.0, overlap=0.0, median_label=3.0,
    window_normalization="global_rms", label_threshold_mode="global"), indent=2) + "\n")
PY
    if [[ "$REQUIRE_GPU" == 1 ]]; then
        "$PYTHON" -c 'import tensorflow as tf; assert tf.config.list_physical_devices("GPU"), "No TensorFlow GPU visible"'
    fi
    git rev-parse HEAD > "$FOLD_DIR/code_revision.txt"
    cp "$MODEL_CONFIG" "$FOLD_DIR/model_config.json"
    MODEL_CONFIG="$FOLD_DIR/model_config.json"
fi

CLI=("$PYTHON" -m experiments.decoder_recovery)
run "${CLI[@]}" prepare-fold --loader-config "$FOLD_DIR/loader.json" \
    --test-subjects "$SUBJECT_ID" --validation-subjects "${VALIDATION_IDS[@]}" \
    --identity-validation-trials 1 --seed "$SEED" --output "$FOLD_DIR/data"
TRAIN="$FOLD_DIR/data/train.npz"
ID_VALIDATION="$FOLD_DIR/data/identity_validation.npz"
SOURCE_VALIDATION="$FOLD_DIR/data/validation.npz"
TEST="$FOLD_DIR/data/test.npz"

# Phase 1: task + reconstruction, no ISA gradient. Phase 2 starts from exactly
# those weights and enables ISA. Both frozen checkpoints are retained.
SOURCE_TRAIN=("${CLI[@]}" latent-coords --train "$TRAIN" --validation "$SOURCE_VALIDATION" \
    --model-config "$MODEL_CONFIG" --phase1-epochs "$PHASE1_EPOCHS" \
    --phase2-epochs "$PHASE2_EPOCHS" --isa-weight "$ISA_WEIGHT" \
    --batch-size "$BATCH_SIZE" --seed "$SEED" --output "$FOLD_DIR/latent_coords")
if [[ -n "${RECONSTRUCTION_SEAM_WEIGHT:-}" ]]; then
    SOURCE_TRAIN+=(--reconstruction-seam-weight "$RECONSTRUCTION_SEAM_WEIGHT")
fi
if [[ -n "${RECONSTRUCTION_DERIVATIVE_WEIGHT:-}" ]]; then
    SOURCE_TRAIN+=(--reconstruction-derivative-weight "$RECONSTRUCTION_DERIVATIVE_WEIGHT")
fi
run "${SOURCE_TRAIN[@]}"
SOURCE_MODEL="$FOLD_DIR/latent_coords/post_isa.keras"
REFERENCE_MODEL="$FOLD_DIR/latent_coords/pre_isa.keras"

run "${CLI[@]}" pretrain-identifier --train "$TRAIN" --validation "$ID_VALIDATION" \
    --epochs "$IDENTIFIER_EPOCHS" --batch-size "$BATCH_SIZE" --seed "$SEED" \
    --output "$FOLD_DIR/identifier"
# A second real-EEG probe never supplies a decoder gradient.
run "${CLI[@]}" pretrain-identifier --train "$TRAIN" --validation "$ID_VALIDATION" \
    --epochs "$IDENTIFIER_EPOCHS" --batch-size "$BATCH_SIZE" --seed "$((SEED + 1))" \
    --output "$FOLD_DIR/independent_probe"

for ARM in reconstruction_only idea1; do
    WEIGHT=0
    if [[ "$ARM" == idea1 ]]; then WEIGHT="$IDENTITY_WEIGHT"; fi
    DECODER_TRAIN=("${CLI[@]}" idea1 --model "$SOURCE_MODEL" \
        --identifier "$FOLD_DIR/identifier/identifier.keras" \
        --identifier-metadata "$FOLD_DIR/identifier/identifier.json" \
        --train "$TRAIN" --validation "$ID_VALIDATION" \
        --identity-weight "$WEIGHT" --identity-loss "$IDENTITY_LOSS" \
        --epochs "$DECODER_EPOCHS" --batch-size "$BATCH_SIZE" --seed "$SEED" \
        --output "$FOLD_DIR/$ARM")
    if [[ -n "${DECODER_RECONSTRUCTION_SEAM_WEIGHT:-}" ]]; then
        DECODER_TRAIN+=(--reconstruction-seam-weight "$DECODER_RECONSTRUCTION_SEAM_WEIGHT")
    fi
    if [[ -n "${DECODER_RECONSTRUCTION_DERIVATIVE_WEIGHT:-}" ]]; then
        DECODER_TRAIN+=(--reconstruction-derivative-weight "$DECODER_RECONSTRUCTION_DERIVATIVE_WEIGHT")
    fi
    run "${DECODER_TRAIN[@]}"
done

for ARM in baseline reconstruction_only idea1 latent_coords; do
    MODEL="$SOURCE_MODEL"
    if [[ "$ARM" == reconstruction_only || "$ARM" == idea1 ]]; then
        MODEL="$FOLD_DIR/$ARM/decoder_identity.keras"
    fi
    # Identify known source subjects on withheld complete trials; report
    # held-out-target emotion fidelity separately (target ID is unknown).
    for SPLIT in identity_validation test; do
        EVALUATE=("${CLI[@]}" evaluate --model "$MODEL" \
            --data "$FOLD_DIR/data/$SPLIT.npz" \
            --identifier "$FOLD_DIR/independent_probe/identifier.keras" \
            --identifier-metadata "$FOLD_DIR/independent_probe/identifier.json" \
            --batch-size "$BATCH_SIZE" --output "$FOLD_DIR/evaluation/$ARM/$SPLIT")
        if [[ "$ARM" == latent_coords ]]; then
            EVALUATE+=(--reference-model "$REFERENCE_MODEL")
        fi
        run "${EVALUATE[@]}"
    done
    if [[ "$RUN_COUNTERFACTUALS" == 1 ]]; then
        COUNTERFACTUAL=("${CLI[@]}" counterfactual --model "$MODEL" \
            --data "$TEST" --decoder-mode joint --max-steps "$CF_MAX_STEPS" \
            --target-probability "$TARGET_PROBABILITY" \
            --physiological-weight "$PHYSIOLOGICAL_WEIGHT" \
            --output "$FOLD_DIR/counterfactuals/$ARM")
        if [[ "$CF_LIMIT" != all ]]; then COUNTERFACTUAL+=(--limit "$CF_LIMIT"); fi
        if [[ "$ARM" == latent_coords ]]; then
            COUNTERFACTUAL+=(--reference-model "$REFERENCE_MODEL")
        fi
        run "${COUNTERFACTUAL[@]}"
    fi
done

if [[ "$DRY_RUN" != 1 ]]; then
    "$PYTHON" - "$FOLD_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
reports = {str(p.parent.relative_to(root / "evaluation")): json.loads(p.read_text())
           for p in sorted((root / "evaluation").glob("*/*/report.json"))}
(root / "comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
PY
fi
echo "Decoder recovery suite output: $FOLD_DIR"
