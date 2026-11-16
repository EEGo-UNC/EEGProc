#!/bin/bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: bash $0 COMPLETED_VALENCE_CONFIGURATION_DIR" >&2
    exit 2
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SI_CONFIG="$(cd -- "$1" && pwd -P)"
SI_OUT="$SI_CONFIG/decoder_si"
venv_dir="${VENV_DIR:-$project_dir/venv312}"
[[ -x "$venv_dir/bin/python" ]] || {
    echo "Set VENV_DIR to the EEGProc Python environment (missing $venv_dir/bin/python)." >&2
    exit 2
}
VENV_DIR="$(cd -- "$venv_dir" && pwd -P)"
export SI_CONFIG SI_OUT VENV_DIR
cd "$project_dir"

for shard in "$SI_CONFIG"/shard_{0,1,2}; do
    [[ -r "$shard/task.json" ]] || { echo "Missing shard metadata: $shard/task.json" >&2; exit 1; }
done
for subject in {0..22}; do
    printf -v subject_dir 'subject_%02d' "$subject"
    checkpoint="$SI_CONFIG/shard_$((subject / 8))/$subject_dir/epoch_048.keras"
    [[ -s "$checkpoint" && -r "$checkpoint" ]] || { echo "Missing or unreadable checkpoint: $checkpoint" >&2; exit 1; }
done
mkdir -p "$SI_OUT/logs"

submit=(sbatch --parsable --job-name=valence-si-cfx --partition=a100-gpu
        --qos=gpu_access --gres=gpu:1 --ntasks=1 --cpus-per-task=6
        --mem=64G --time=24:00:00 --array=0-22%2 --export=ALL --chdir="$PWD"
        --output="$SI_OUT/logs/%A_%a.out")

"${submit[@]}" <<'SLURM'
#!/bin/bash
set -euo pipefail
module purge
module load python/3.12.4 cuda/12.9 cudnn/9.11.0

export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 TF_GPU_ALLOCATOR=cuda_malloc_async
if [[ -n "${EBROOTCUDA:-}" ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_cuda_data_dir=$EBROOTCUDA"
fi
py="$VENV_DIR/bin/python"
subject="$SLURM_ARRAY_TASK_ID"
fold="$SI_OUT/run_$SLURM_ARRAY_JOB_ID/fold_$(printf '%02d' "$subject")"
mkdir -p "$(dirname "$fold")"
mkdir "$fold"
scratch=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/decoder-si.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT

validation=$("$py" - "$SI_CONFIG" "$subject" "$fold" <<'PY'
import json, random, sys
from pathlib import Path
configuration, subject, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
shard = configuration / f"shard_{subject // 8}"
checkpoint = shard / f"subject_{subject:02d}" / "epoch_048.keras"
spec = (shard / "task.json").read_text()
task = json.loads(spec)
if task["label_dimension"] != "valence":
    raise SystemExit(f"Expected valence shard metadata: {shard / 'task.json'}")
if subject not in task["target_subjects"] or not checkpoint.is_file():
    raise SystemExit(f"Missing epoch-48 checkpoint for subject {subject}: {checkpoint}")
(out / "task.json").write_text(spec)
(out / "loader.json").write_text(json.dumps(dict(
    eeg_path=task["eeg"], labels_path=task["labels"], dataset="dreamer",
    label_dimension="valence", window_size_sec=1, fs=128, overlap=0,
    median_label=3, window_normalization="global_rms", label_threshold_mode="global")))
adapted = dict(target_subject=subject, stage="zero_shot_source_model",
               path=str(out / "idea1/decoder_identity.keras"),
               source_checkpoint=str(checkpoint))
(out / "models.json").write_text(json.dumps({"models": [adapted]}))
(out / "source_checkpoint.txt").write_text(str(checkpoint) + "\n")
print(*sorted(random.Random(42 + subject).sample([s for s in range(23) if s != subject], 4)))
PY
)
read -r -a val_ids <<< "$validation"
cli=("$py" -m experiments.decoder_recovery)
"${cli[@]}" prepare-fold --loader-config "$fold/loader.json" --test-subjects "$subject" \
    --validation-subjects "${val_ids[@]}" --output "$scratch/data"
cp "$scratch/data/fold.json" "$fold/fold.json"

split=(--train "$scratch/data/train.npz" --validation "$scratch/data/identity_validation.npz")
"${cli[@]}" pretrain-identifier "${split[@]}" --epochs 20 --batch-size 4 --output "$fold/identifier"
"${cli[@]}" idea1 "${split[@]}" --epochs 10 --batch-size 4 --identity-weight 0.1 \
    --model "$(cat "$fold/source_checkpoint.txt")" \
    --identifier "$fold/identifier/identifier.keras" \
    --identifier-metadata "$fold/identifier/identifier.json" --output "$fold/idea1"

bash src/eegproc/model_explainability/slurm/run_valence_cfx_slurm.sh \
    "$fold/models.json" "$SI_OUT/counterfactuals"
SLURM
