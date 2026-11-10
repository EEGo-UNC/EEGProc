#!/bin/bash
# Read the existing environment; never install into it or modify it.
module purge
module load python/3.12.4
ROOT=/work/users/p/m/pmucharl/sic-v15-seed-iv
PROJECT_DIR="$ROOT/source"
PYTHON="$HOME/EEGProc/venv312/bin/python"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$PROJECT_DIR"
