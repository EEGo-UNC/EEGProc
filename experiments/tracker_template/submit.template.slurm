#!/bin/bash
#SBATCH --job-name={{RUN_ID}}
# Add your site's resource directives and environment setup here.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/EEGProc}"
cd "$PROJECT_DIR"
RUN_DIR="$PROJECT_DIR/{{EXPERIMENT_DIR}}/runs/{{RUN_ID}}"
# Your training adapter owns its arguments, dependencies, data checks and status.
python "{{TRAINING_SCRIPT}}" --config "$RUN_DIR/config.json" --output-dir "$RUN_DIR"
