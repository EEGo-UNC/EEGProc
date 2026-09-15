#!/bin/bash
#SBATCH --job-name=cf_u0_t0_2
#SBATCH --output=cf_u0_t0_2_%j.out
#SBATCH --error=cf_u0_t0_2_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:30:00

set -euo pipefail

# Aggressive target-only SICModelv15 counterfactuals for subject 0, trials 0-2.
# Each trial is an independent sequential optimization assigned to one GPU, so
# all three trials run concurrently without sharing TensorFlow state.

module purge
module load python/3.12.4
module load cuda/12.9
module load cudnn/9.11.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv312}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/runs/smoke/sic_v15_arousal_grid/DREAMER/arousal/suite_330197/users_0_3/smoke_arousal_0_3_grid_20260908_200512/configuration_0006/loso_zero_shot_models/loso_fold_0001_target_0_zero_shot.keras}"
EEG_PATH="${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}"
LABELS_PATH="${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}"
XAI_ROOT="${XAI_ROOT:-$PROJECT_DIR/XAI_runs}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}}"
RUN_ROOT="$XAI_ROOT/subject_0_aggressive_counterfactuals_$RUN_ID"

SUBJECT_ID=0
TRIAL_IDS=(0 1 2)
EXPECTED_TRIALS=3
TARGET_PROBABILITY="${TARGET_PROBABILITY:-0.99}"
LEARNING_RATE="${LEARNING_RATE:-1.0}"
LEARNING_RATE_DECAY="${LEARNING_RATE_DECAY:-1.0}"
MAX_STEPS="${MAX_STEPS:-200}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-5.0}"
LOG_EVERY="${LOG_EVERY:-1}"
SEED="${SEED:-42}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: Python environment not found at $VENV_DIR."
    exit 2
fi
for required_file in "$MODEL_PATH" "$EEG_PATH" "$LABELS_PATH"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file not found: $required_file"
        exit 2
    fi
done
if [[ -e "$RUN_ROOT" ]]; then
    echo "ERROR: refusing to overwrite existing output: $RUN_ROOT"
    exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: Slurm did not expose the requested GPUs."
    exit 2
fi

IFS=',' read -r -a GPU_TOKENS <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#GPU_TOKENS[@]} < EXPECTED_TRIALS )); then
    echo "ERROR: expected at least $EXPECTED_TRIALS GPUs; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    exit 2
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/.matplotlib" "$RUN_ROOT/.cache"
cd "$PROJECT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TF_FORCE_GPU_ALLOW_GROWTH=true

run_trial() {
    local trial_id="$1"
    local gpu_token="$2"
    local task_root="$RUN_ROOT/trial_$(printf '%02d' "$trial_id")"
    local trial_directory="$task_root/subject_${SUBJECT_ID}_trial_${trial_id}"
    local log_path="$RUN_ROOT/logs/trial_$(printf '%02d' "$trial_id").log"

    (
        export CUDA_VISIBLE_DEVICES="$gpu_token"
        export MPLCONFIGDIR="$RUN_ROOT/.matplotlib/trial_$trial_id"
        export XDG_CACHE_HOME="$RUN_ROOT/.cache/trial_$trial_id"
        export OMP_NUM_THREADS=4
        export TF_NUM_INTRAOP_THREADS=4
        export TF_NUM_INTEROP_THREADS=2
        mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

        echo "Starting subject $SUBJECT_ID trial $trial_id on GPU $CUDA_VISIBLE_DEVICES"
        "$VENV_DIR/bin/python" -m eegproc.model_explainability.counterfactuals.runner \
            --model "$MODEL_PATH" \
            --model-module eegproc.deep_learning.joint_architectures.SICModelv15.sic_model \
            --raw-eeg-npy "$EEG_PATH" \
            --raw-labels-npy "$LABELS_PATH" \
            --dataset dreamer \
            --label-dimension arousal \
            --fs 128 \
            --window-sec 1 \
            --window-overlap 0 \
            --window-normalization global_rms \
            --label-threshold-mode global \
            --median-label 3 \
            --subject-id "$SUBJECT_ID" \
            --trial-id "$trial_id" \
            --decoder-mode joint \
            --target-loss-component confidence \
            --target-probability "$TARGET_PROBABILITY" \
            --target-weight 1 \
            --latent-weight 0 \
            --decoded-weight 0 \
            --physiological-weight 0 \
            --learning-rate "$LEARNING_RATE" \
            --learning-rate-decay "$LEARNING_RATE_DECAY" \
            --max-steps "$MAX_STEPS" \
            --gradient-clip-norm "$GRADIENT_CLIP_NORM" \
            --stop-on-success \
            --log-every "$LOG_EVERY" \
            --seed "$SEED" \
            --out-dir "$task_root"

        "$VENV_DIR/bin/python" -m eegproc.model_explainability.counterfactuals.heatmap \
            "$trial_directory/counterfactual.npz" \
            --branch joint \
            --sampling-rate 128 \
            --no-show

        "$VENV_DIR/bin/python" -m eegproc.model_explainability.counterfactuals.topography \
            "$trial_directory/counterfactual.npz" \
            --branch joint \
            --no-show

        "$VENV_DIR/bin/python" -m eegproc.model_explainability.counterfactuals.training_monitor \
            "$trial_directory/history.csv" \
            --no-show

        echo "Completed subject $SUBJECT_ID trial $trial_id on GPU $CUDA_VISIBLE_DEVICES"
    ) >"$log_path" 2>&1
}

echo "Aggressive target-only counterfactual run"
echo "Subject: $SUBJECT_ID"
echo "Trials: ${TRIAL_IDS[*]}"
echo "Target probability: $TARGET_PROBABILITY"
echo "Learning rate: $LEARNING_RATE (decay $LEARNING_RATE_DECAY)"
echo "Maximum steps: $MAX_STEPS"
echo "Model: $MODEL_PATH"
echo "Output: $RUN_ROOT"
echo "GPU assignments: ${GPU_TOKENS[0]}, ${GPU_TOKENS[1]}, ${GPU_TOKENS[2]}"

pids=()
for index in "${!TRIAL_IDS[@]}"; do
    run_trial "${TRIAL_IDS[$index]}" "${GPU_TOKENS[$index]}" &
    pids+=("$!")
done

worker_status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        worker_status=1
    fi
done

if (( worker_status == 0 )); then
    "$VENV_DIR/bin/python" -m eegproc.model_explainability.counterfactuals.aggregate_metrics \
        "$RUN_ROOT" \
        --subject-id "$SUBJECT_ID" \
        --expected-trials "$EXPECTED_TRIALS" \
        --require-complete
    echo "Completed all three counterfactual trials."
else
    "$VENV_DIR/bin/python" -m eegproc.model_explainability.counterfactuals.aggregate_metrics \
        "$RUN_ROOT" \
        --subject-id "$SUBJECT_ID" \
        --expected-trials "$EXPECTED_TRIALS" || true
    echo "ERROR: at least one trial failed; inspect $RUN_ROOT/logs."
    exit 1
fi
