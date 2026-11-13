#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/perfail_ops_tmp"
PY="$BASE/perfail_py311/bin/python"
ROOT="./data/perfail2027/mmfi"
SEEDS=(11 22 33)
INDEX="${SLURM_ARRAY_TASK_ID:?Slurm array ID missing}"
SEED="${SEEDS[$INDEX]}"
exec "$PY" -u "$BASE/mmfi_matrix_v2.py" train --seed "$SEED" \
 --input "$ROOT/compact_v2_48" --out "$ROOT/matrix_v2_48" --epochs 30
