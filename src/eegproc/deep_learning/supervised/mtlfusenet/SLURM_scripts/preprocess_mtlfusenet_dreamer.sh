#!/bin/bash
#SBATCH --job-name=mtlfuse_preprocess
#SBATCH --output=mtlfuse_preprocess_%j.out
#SBATCH --error=mtlfuse_preprocess_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

module purge
module load python/3.12.4

PROJECT_DIR="$HOME/EEGProc"
VENV_DIR="$PROJECT_DIR/venv312"
CSV_PATH="$PROJECT_DIR/datasets/dreamer_joined.csv"
OUTPUT_DIR="$PROJECT_DIR/processed_trials"

cd "$PROJECT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Create it and install EEGProc before submitting this job."
    exit 1
fi

source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Ensure the local package is importable from this environment.
# Reuse the lock pattern from training scripts so concurrent jobs do not race.
if command -v flock >/dev/null 2>&1; then
    (
        flock -x 9
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
        python -m pip install -e .
    ) 9>"$PROJECT_DIR/.venv312_install.lock"
else
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .
fi

if [[ ! -f "$CSV_PATH" ]]; then
    echo "ERROR: DREAMER CSV not found at $CSV_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
echo "Input CSV: $CSV_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-1}"
echo "Start time: $(date)"

python - <<'PY'
import eegproc
print("Imported eegproc from:", eegproc.__file__)
PY

python -m eegproc.deep_learning.supervised.mtlfusenet.mtl_preprocess \
    --csv "$CSV_PATH" \
    --out "$OUTPUT_DIR" \
    --mi-max-samples 5000

echo "Cached trial count:"
find "$OUTPUT_DIR" -maxdepth 1 -name 'subj*_trial*.pkl' | wc -l

echo "End time: $(date)"
