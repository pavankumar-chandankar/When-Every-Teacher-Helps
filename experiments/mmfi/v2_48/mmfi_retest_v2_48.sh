#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/perfail_ops_tmp"
PY="$BASE/perfail_py311/bin/python"
ROOT="./data/perfail2027/mmfi"
LOG="$BASE/perfail_logs"
mkdir -p "$BASE" "$LOG" "$ROOT"

# Verify the exact frozen v1 code before making separate v2 copies.
(
  cd "$BASE"
  printf '%s\n' \
    '02a947fd8c5bb09ef480265fef78608ddb4931eae1a786f77af0bde778872900  mmfi_compact_v1.py' \
    '1ce9dc0697c6b5cf411ce1125b05a02464e89e0d42d4dc95e75b76617efd2e6b  mmfi_matrix_v1.py' \
  | sha256sum -c -
)

"$PY" - "$BASE" <<'PY'
import hashlib
from pathlib import Path
import sys
base=Path(sys.argv[1])
patches={
 'mmfi_compact_v1.py':[
  ('FREQ = 12','FREQ = 48'),
  ("compact_v1'", "compact_v2_48'"),
  ("'note':'rgbpose=2D keypoints; labels from path; mmwave empty frames=zero descriptor'",
   "'note':'Exploratory 48-frame replication; rgbpose=2D keypoints; labels from path; mmwave empty frames=zero descriptor'")
 ],
 'mmfi_matrix_v1.py':[
  ('(1080,12)','(1080,48)'),
  ('mmfi_sequence_12frames_subject_disjoint_v1','mmfi_sequence_48frames_subject_disjoint_v2'),
  ('(3,12,dim)','(3,48,dim)'),
  ("'./data/perfail2027/mmfi/compact_v1'",
   "'./data/perfail2027/mmfi/compact_v2_48'"),
  ("'./data/perfail2027/mmfi/matrix_v1'",
   "'./data/perfail2027/mmfi/matrix_v2_48'")
 ]
}
expected_sha={
 'mmfi_compact_v2.py':'43abc886f148d656fc344bfc30a9473e0972311b6baf06b07b1ad1032febe8b8',
 'mmfi_matrix_v2.py':'ee45746ea523bd0b3f3233efeea0aacb8dd32c253c0287ee72e3f40823440a8d'
}
for name,edits in patches.items():
 s=(base/name).read_text()
 for old,new in edits:
  assert s.count(old)==1,(name,'unexpected source',old,s.count(old))
  s=s.replace(old,new)
 dst=base/name.replace('_v1.py','_v2.py')
 h=hashlib.sha256(s.encode()).hexdigest()
 assert h==expected_sha[dst.name],(dst.name,'code checksum mismatch')
 if dst.exists():
  assert dst.read_text()==s,('v2 output exists with different code',str(dst))
 else:
  dst.write_text(s)
 print('FROZEN_V2',dst,'SHA256',h,flush=True)
PY

"$PY" -m py_compile "$BASE/mmfi_compact_v2.py" "$BASE/mmfi_matrix_v2.py"
"$PY" "$BASE/mmfi_compact_v2.py" self-test
"$PY" "$BASE/mmfi_matrix_v2.py" self-test
"$PY" -c 'import torch; from PIL import Image; import scipy; print("ENV_READY", torch.__version__)'

if [ -e "$ROOT/matrix_v2_48/seed_11.json" ] || \
   [ -e "$ROOT/matrix_v2_48/seed_22.json" ] || \
   [ -e "$ROOT/matrix_v2_48/seed_33.json" ]; then
    echo 'STOP: existing v2 seed output; inspect before submitting another training run.'
    exit 1
fi

cat > "$BASE/mmfi_v2_array.sh" <<'SH'
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
SH

cat > "$BASE/mmfi_v2_prep_then_train.sh" <<'SH'
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
SH
chmod +x "$BASE/mmfi_v2_array.sh" "$BASE/mmfi_v2_prep_then_train.sh"
JOB=$(sbatch --parsable --job-name=perfail-mmfi-v2-prep \
 --cpus-per-task=4 --mem=16G --time=06:00:00 \
 --output="$LOG/mmfi_v2_prep_%j.log" "$BASE/mmfi_v2_prep_then_train.sh")
echo "MMFI_V2_PREPROCESS_JOB=$JOB"
echo "PREPROCESS_LOG=$LOG/mmfi_v2_prep_${JOB}.log"
echo 'Protocol: 48 evenly spaced frames, unchanged subject splits, architecture, hyperparameters and seeds.'
echo 'Frozen v1 outputs untouched. No MATLAB.'
