#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
EXPS=(29 32 33 56 57 58 59 60 61)
EXP="${EXPS[$SLURM_ARRAY_TASK_ID]}"
BASE="$HOME/perfail_ops_tmp/perfail_v7r1"
ROOT="./data/perfail2027/operanet/aligned_v7r1_10"
(cd "$BASE" && sha256sum -c SOURCE_SHA256.txt)
exec "$HOME/perfail_ops_tmp/perfail_py311/bin/python" -u "$BASE/operanet_build_windows_v7.py" --exp "$EXP" --output "$ROOT"
