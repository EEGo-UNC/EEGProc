#!/bin/bash
# Internal task entry; sourced environment is inherited from a Slurm launcher.
set -euo pipefail
SIC16="$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv16"
TARGET_ARGS=()
if [[ -n "${TARGET_SUBJECTS:-}" ]]; then
    read -r -a TARGET_LIST <<< "$TARGET_SUBJECTS"
    TARGET_ARGS=(--target-subjects "${TARGET_LIST[@]}")
fi
SUITE="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?Run through Slurm}}"
nvidia-smi
exec python "$SIC16/run_experiments.py" run \
    --configs "${CONFIGS:-$SIC16/configurations.json}" \
    --config-index "${SLURM_ARRAY_TASK_ID:?Expected an array task}" \
    --output-dir "$OUTPUT_ROOT/suite_${SUITE}/configuration_${SLURM_ARRAY_TASK_ID}" \
    --eeg "${EEG_PATH:-$PROJECT_DIR/datasets/dreamer_eeg.npy}" \
    --labels "${LABELS_PATH:-$PROJECT_DIR/datasets/dreamer_labels.npy}" \
    --source-epochs "$SOURCE_EPOCHS" --time-budget-seconds "$TIME_BUDGET_SECONDS" \
    --batch-size "${BATCH_SIZE:-1}" --validation-subjects "${VALIDATION_SUBJECTS:-2}" \
    --seed "${SEED:-42}" "${TARGET_ARGS[@]}"
