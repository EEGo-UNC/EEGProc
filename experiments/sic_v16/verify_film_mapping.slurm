#!/bin/bash
#SBATCH --job-name=sicv16-film-mapping
#SBATCH --output=sicv16-film-mapping-%j.out
#SBATCH --error=sicv16-film-mapping-%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source "$PROJECT_DIR/src/eegproc/deep_learning/joint_architectures/SICModelv16/SLURM_scripts/environment.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
extra_args=()
if [[ -n "${FILM_MAP:-}" ]]; then extra_args+=(--reference-map "$FILM_MAP"); fi
exec python "$PROJECT_DIR/experiments/sic_v16/verify_film_mapping.py" \
    --output-dir "$SIC_V16_WORK_ROOT/film-mapping/job-${SLURM_JOB_ID}" \
    --source-csv "${SOURCE_CSV:-$PROJECT_DIR/datasets/dreamer_joined.csv}" "${extra_args[@]}"
