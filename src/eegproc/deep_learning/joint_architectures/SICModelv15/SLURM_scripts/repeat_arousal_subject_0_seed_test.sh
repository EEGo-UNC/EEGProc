#!/bin/bash
#SBATCH --job-name=v15_arousal_s0_seed_test
#SBATCH --output=repeat_arousal_subject_0_seed_test_%j.out
#SBATCH --error=repeat_arousal_subject_0_seed_test_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

# Run the exact arousal smoke winner twice for target subject 0 in independent
# spawned workers. Both repetitions use the same base seed and deterministic
# TensorFlow settings and fixed-order two-GPU execution from
# full_run_v15_arousal.sh. Afterward, compare the
# resolved model config, saved model tensors, trial predictions, calibration
# rows, and path-independent subject summary. Any mismatch fails the Slurm job.

PROJECT_DIR="${PROJECT_DIR:-$HOME/EEGProc}"
FULL_RUN_SCRIPT="${FULL_RUN_SCRIPT:-$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv15/SLURM_scripts/full_run_v15_arousal.sh}"
TRAINING_SEED="${TRAINING_SEED:-42}"
SUITE_ID="${SLURM_JOB_ID:-manual}"
BASE_OUTPUT_DIR="${REPRO_OUTPUT_DIR:-$PROJECT_DIR/runs/reproducibility/sic_v15_arousal_cfg6/suite_${SUITE_ID}/subject_0}"
REPRO_SOURCE_EPOCHS="${REPRO_SOURCE_EPOCHS:-4}"
REPRO_CALIBRATION_EPOCHS="${REPRO_CALIBRATION_EPOCHS:-10}"

if [[ ! -f "$FULL_RUN_SCRIPT" ]]; then
    echo "ERROR: full-run launcher not found: $FULL_RUN_SCRIPT"
    exit 1
fi

echo "SIC v15 deterministic repeatability test"
echo "Subject: 0"
echo "Repetitions: 2"
echo "Base seed: $TRAINING_SEED (effective subject seed is also $TRAINING_SEED)"
echo "Fixed configuration: focal_gamma=1.0 vc_alpha=2.0 reconstruction=0.6 subject_loss=0.2"
echo "Repeatability budget: $REPRO_SOURCE_EPOCHS source epochs, $REPRO_CALIBRATION_EPOCHS calibration epochs"
echo "Output root: $BASE_OUTPUT_DIR"

for repetition in 1 2; do
    echo
    echo "Starting independent repetition $repetition of 2"
    if [[ "$repetition" == "1" ]]; then
        RUN_PREFLIGHT=1
    else
        RUN_PREFLIGHT=0
    fi

    TRAINING_SEED="$TRAINING_SEED" \
    SIC_RUN_NAME="repeat_arousal_subject_0_seed_${TRAINING_SEED}_run_${repetition}" \
    SIC_OUTPUT_DIR="$BASE_OUTPUT_DIR/repeat_${repetition}" \
    SIC_TARGET_SUBJECTS="0" \
    SIC_EXPECTED_GPUS="2" \
    SIC_N_JOBS="1" \
    SIC_GPU_IDS="0 1" \
    SIC_RUN_PREFLIGHT="$RUN_PREFLIGHT" \
    SOURCE_EPOCHS="$REPRO_SOURCE_EPOCHS" \
    CALIBRATION_EPOCHS="$REPRO_CALIBRATION_EPOCHS" \
        bash "$FULL_RUN_SCRIPT"
done

CONFIG_ONE="$(find "$BASE_OUTPUT_DIR/repeat_1" -type d -name configuration_0001 -print -quit)"
CONFIG_TWO="$(find "$BASE_OUTPUT_DIR/repeat_2" -type d -name configuration_0001 -print -quit)"

if [[ -z "$CONFIG_ONE" || -z "$CONFIG_TWO" ]]; then
    echo "ERROR: could not locate both configuration_0001 output directories."
    exit 1
fi

export CONFIG_ONE CONFIG_TWO BASE_OUTPUT_DIR TRAINING_SEED

cd "$PROJECT_DIR"
echo
echo "Comparing deterministic artifacts"
echo "Repetition 1: $CONFIG_ONE"
echo "Repetition 2: $CONFIG_TWO"
# Keep a comparison traceback in the main Slurm output so a failed seed test
# can be diagnosed from the same log as the two training repetitions.
python - 2>&1 <<'PY_COMPARE'
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

# Import the registered SIC classes before loading the native Keras archives.
from src.eegproc.deep_learning.joint_architectures.SICModelv15 import sic_model  # noqa: F401


left = Path(os.environ["CONFIG_ONE"])
right = Path(os.environ["CONFIG_TWO"])
report_path = Path(os.environ["BASE_OUTPUT_DIR"]) / "reproducibility_report.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def only_model(config_dir: Path) -> Path:
    models = sorted((config_dir / "loso_zero_shot_models").glob("*.keras"))
    if len(models) != 1:
        raise AssertionError(
            f"Expected one saved subject-0 model in {config_dir}, found {models}"
        )
    return models[0]


def normalized_csv(path: Path, ignored_columns: set[str] | None = None):
    ignored_columns = ignored_columns or set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {key: value for key, value in row.items() if key not in ignored_columns}
        for row in rows
    ]


with (left / "model_config.json").open(encoding="utf-8") as handle:
    left_config = json.load(handle)
with (right / "model_config.json").open(encoding="utf-8") as handle:
    right_config = json.load(handle)
if left_config != right_config:
    raise AssertionError("Resolved model_config.json differs between repetitions.")

expected = {
    "focal_gamma": 1.0,
    "vc_alpha": 2.0,
    "reconstruction_loss_weight": 0.6,
    "subject_loss_weight": 0.2,
    "mldg_seed": int(os.environ["TRAINING_SEED"]),
}
for key, expected_value in expected.items():
    actual = left_config.get(key)
    if actual != expected_value:
        raise AssertionError(
            f"Unexpected winning-config value for {key}: {actual!r} != {expected_value!r}"
        )

left_model_path = only_model(left)
right_model_path = only_model(right)
left_model = tf.keras.models.load_model(left_model_path, compile=False)
right_model = tf.keras.models.load_model(right_model_path, compile=False)

if len(left_model.weights) != len(right_model.weights):
    raise AssertionError("Saved models contain different numbers of tensors.")

tensor_differences = []
tensor_hashes = []
for index, (left_weight, right_weight) in enumerate(
    zip(left_model.weights, right_model.weights)
):
    left_value = np.asarray(left_weight)
    right_value = np.asarray(right_weight)
    equal = np.array_equal(left_value, right_value, equal_nan=True)
    max_abs_difference = (
        0.0
        if equal
        else float(np.nanmax(np.abs(left_value - right_value)))
    )
    tensor_hashes.append(
        hashlib.sha256(left_value.tobytes(order="C")).hexdigest()
    )
    if not equal:
        tensor_differences.append(
            {
                "index": index,
                "name": left_weight.name,
                "max_abs_difference": max_abs_difference,
            }
        )

if tensor_differences:
    raise AssertionError(
        "Saved model tensors are not bitwise identical: "
        + json.dumps(tensor_differences[:10], indent=2)
    )

exact_csv_files = (
    "sic_trial_predictions.csv",
    "sic_calibration_folds.csv",
)
csv_hashes = {}
for filename in exact_csv_files:
    left_path = left / filename
    right_path = right / filename
    if left_path.read_bytes() != right_path.read_bytes():
        raise AssertionError(f"{filename} differs between repetitions.")
    csv_hashes[filename] = file_sha256(left_path)

ignored_summary_columns = {
    "zero_shot_model_path",
}
left_summary = normalized_csv(
    left / "sic_subject_summary.csv", ignored_summary_columns
)
right_summary = normalized_csv(
    right / "sic_subject_summary.csv", ignored_summary_columns
)
if left_summary != right_summary:
    raise AssertionError(
        "sic_subject_summary.csv differs after ignoring output-specific paths."
    )

effective_seed = int(left_summary[0]["training_seed"])
if effective_seed != int(os.environ["TRAINING_SEED"]):
    raise AssertionError(
        f"Subject 0 effective seed is {effective_seed}, expected {os.environ['TRAINING_SEED']}."
    )

report = {
    "status": "PASS",
    "subject": 0,
    "base_seed": int(os.environ["TRAINING_SEED"]),
    "effective_subject_seed": effective_seed,
    "deterministic_training": True,
    "resolved_hyperparameters": expected,
    "configuration_directories": [str(left), str(right)],
    "saved_models": [str(left_model_path), str(right_model_path)],
    "model_tensor_count": len(tensor_hashes),
    "model_tensor_hashes": tensor_hashes,
    "identical_csv_sha256": csv_hashes,
}
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

print()
print("PASS: both subject-0 repetitions are bitwise reproducible.")
print(f"Verified {len(tensor_hashes)} saved model tensors.")
print(f"Report: {report_path}")
PY_COMPARE
