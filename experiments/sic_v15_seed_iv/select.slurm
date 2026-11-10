#!/bin/bash
#SBATCH --job-name=sicv15-seed-select
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/work/users/p/m/pmucharl/sic-v15-seed-iv/logs/select-%j.out
#SBATCH --error=/work/users/p/m/pmucharl/sic-v15-seed-iv/logs/select-%j.err
set -euo pipefail
source /work/users/p/m/pmucharl/sic-v15-seed-iv/source/experiments/sic_v15_seed_iv/environment.sh
ARGS=()
STAGE=full
if [[ "${SMOKE:-0}" == 1 ]]; then
  ARGS+=(--smoke)
  STAGE=smoke
fi
RUN_ID="${RUN_ID:-weighted_accuracy_oracle_v1}"
SUITE_DIR="${SUITE_DIR:-$ROOT/runs/$RUN_ID/$STAGE}"
MODEL_CONFIG="${MODEL_CONFIG:-$PROJECT_DIR/experiments/sic_v15_seed_iv/model_config.json}"
"$PYTHON" experiments/sic_v15_seed_iv/train.py --phase select \
  --suite-dir "$SUITE_DIR" --source-epochs "${SOURCE_EPOCHS:-12}" \
  --model-config "$MODEL_CONFIG" "${ARGS[@]}"
