#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/perfail_ops_tmp"
PY="$BASE/perfail_py311/bin/python"
ROOT="./data/perfail2027/mmfi"
SEEDS=(11 22 33)
INDEX="${SLURM_ARRAY_TASK_ID:?requires Slurm array}"
SEED="${SEEDS[$INDEX]}"
"$PY" -u "$BASE/mmfi_matrix_v1.py" train --seed "$SEED" --input "$ROOT/compact_v1" --out "$ROOT/matrix_v1" --epochs 30
