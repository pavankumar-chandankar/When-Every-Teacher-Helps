#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/perfail_ops_tmp"
PY="$BASE/perfail_py311/bin/python"
ROOT="./data/perfail2027/mmfi"
LOG="$BASE/perfail_logs"
"$PY" -u "$BASE/mmfi_compact_v2.py" prepare \
 --archive ./data/datasets/mmfi/MMFi_Dataset.zip \
 --out "$ROOT/compact_v2_48"
"$PY" - "$ROOT/compact_v2_48" <<'PY'
import json, pathlib,sys
import numpy as np
p=pathlib.Path(sys.argv[1]); m=json.loads((p/'manifest.json').read_text())
assert m['status']=='PREP_COMPLETE' and len(m['frames_sampled'])==48
assert len(list(p.glob('S[0-9][0-9].npz')))==40
for i in range(1,41):
 with np.load(p/f'S{i:02d}.npz',allow_pickle=False) as d:
  assert d['wifi'].shape==(27,48,570)
  assert d['rgbpose'].shape==(27,48,34)
print('MMFI_V2_48_COHORT_GATE_PASSED',flush=True)
PY
JOB=$(sbatch --parsable --array=0-2%3 --job-name=perfail-mmfi-v2-matrix \
  --cpus-per-task=4 --mem=12G --time=04:00:00 \
  --output="$LOG/mmfi_v2_matrix_%A_%a.log" "$BASE/mmfi_v2_array.sh")
echo "MMFI_V2_MATRIX_JOB=$JOB"
