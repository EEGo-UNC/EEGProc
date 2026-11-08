#!/usr/bin/env bash
#SBATCH --job-name=cfx-coverage
#SBATCH --array=0-45
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=cfx-coverage-%A_%a.out
set -euo pipefail

# Submit from the repository root, with frozen per-task checkpoint/data settings.
# Tasks 0..22 are valence subjects 0..22; tasks 23..45 are arousal subjects 0..22.
index="${SLURM_ARRAY_TASK_ID:?Submit as a 0-45 Slurm array}"
if (( index < 0 || index > 45 )); then
    echo 'Array task must be in 0..45' >&2
    exit 2
fi
if (( index < 23 )); then
    task=valence
    subject="$index"
    models="${VALENCE_MODELS_JSON:?Set VALENCE_MODELS_JSON}"
    model_dir="${VALENCE_MODEL_DIR:-}"
    prepared="${VALENCE_TRIALS_NPZ:-}"
    data_config="${VALENCE_DATA_CONFIG:-}"
    confidence="${VALENCE_TARGET_PROBABILITY:?Set the frozen valence target probability}"
    model_module="${VALENCE_MODEL_MODULE:-eegproc.deep_learning.joint_architectures.SICModelv15.sic_model}"
    fixed_alpha="${VALENCE_FIXED_JOINT_ALPHA:-}"
else
    task=arousal
    subject=$((index - 23))
    models="${AROUSAL_MODELS_JSON:?Set AROUSAL_MODELS_JSON}"
    model_dir="${AROUSAL_MODEL_DIR:-}"
    prepared="${AROUSAL_TRIALS_NPZ:-}"
    data_config="${AROUSAL_DATA_CONFIG:-}"
    confidence="${AROUSAL_TARGET_PROBABILITY:?Set the frozen arousal target probability}"
    model_module="${AROUSAL_MODEL_MODULE:-eegproc.deep_learning.joint_architectures.SICModelv15.sic_model}"
    fixed_alpha="${AROUSAL_FIXED_JOINT_ALPHA:-}"
fi
if [[ -n "$prepared" && -z "$data_config" ]]; then
    data_args=(--trials-npz "$prepared")
elif [[ -z "$prepared" && -n "$data_config" ]]; then
    data_args=(--data-loader eegproc.model_explainability.model_agnostic.sic_adapter:load_sic_raw_trials --data-config "$data_config")
else
    echo "For $task, set exactly one prepared-trial or raw-data configuration path" >&2
    exit 2
fi
extra=()
[[ -z "$model_dir" ]] || extra+=(--model-dir "$model_dir")
[[ -z "$fixed_alpha" ]] || extra+=(--fixed-joint-alpha "$fixed_alpha")
[[ "${RESUME:-0}" != 1 ]] || extra+=(--resume)
printf -v fold 'fold_%02d' "$subject"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python}" -m eegproc.model_explainability.typicality.runner \
    --task "$task" --models-json "$models" --model-module "$model_module" \
    --subjects "$subject" --directions 0_to_1 1_to_0 \
    --typicality-quantile 0.10 --target-probability "$confidence" \
    --include-target-latent --include-typicality-no-physiology \
    --out-dir "${CFX_OUT_DIR:?Set a new immutable suite output directory}/$task/$fold" \
    "${data_args[@]}" "${extra[@]}" "$@"
