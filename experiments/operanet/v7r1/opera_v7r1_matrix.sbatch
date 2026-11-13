#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
BASE="$HOME/perfail_ops_tmp/perfail_v7r1"
ROOT="./data/perfail2027/operanet"
SEEDS=(11 22 33)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
PYTRAIN="$(sed -n 's/^PYTHON=//p' "$ROOT/matrix_v7r1_10/PROTOCOL.txt")"
exec "$PYTRAIN" -u "$BASE/perfail_operanet_matrix_v7.py" seed --seed "$SEED"   --root "$ROOT/aligned_v7r1_10" --out "$ROOT/matrix_v7r1_10"
