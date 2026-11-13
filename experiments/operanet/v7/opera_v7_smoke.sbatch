#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
BASE="$HOME/perfail_ops_tmp/perfail_v7"
ROOT="./data/perfail2027/operanet/aligned_v7_10"
(cd "$BASE" && sha256sum -c SOURCE_SHA256.txt)
exec "$HOME/perfail_ops_tmp/perfail_py311/bin/python" -u "$BASE/operanet_build_windows_v7.py" --exp 18 --output "$ROOT"
