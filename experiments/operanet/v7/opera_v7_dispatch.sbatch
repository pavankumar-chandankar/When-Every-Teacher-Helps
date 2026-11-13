#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/perfail_ops_tmp/perfail_v7"
ROOT="./data/perfail2027/operanet"
LOGS="$HOME/perfail_ops_tmp/perfail_logs"
(cd "$BASE" && sha256sum -c SOURCE_SHA256.txt)
if [[ ! -s "$ROOT/aligned_v7_10/exp018_aligned.npz" ]]; then
    echo 'SMOKE_FAILED: no exp018 output. No batch submitted.' >&2
    exit 2
fi
"$HOME/perfail_ops_tmp/perfail_py311/bin/python" - <<'PYDATA'
import numpy as np
from pathlib import Path
path=Path('./data/perfail2027/operanet/aligned_v7_10/exp018_aligned.npz')
with np.load(path,allow_pickle=False) as z:
    y=z['y']; assert len(y)>=5 and str(z['subject'])=='Three' and str(z['split'])=='train'
    for mod,d in [('wifi',270),('kinect',114),('pwr',600),('uwb',42)]:
        assert z['X_'+mod].shape==(len(y),15,d)
    print('V7_SMOKE_VERIFIED',len(y),'windows',flush=True)
PYDATA
JOB=$(sbatch --parsable --job-name=perfail-opera-v7batch   --array=0-8%2 --cpus-per-task=4 --mem=64G --time=02:00:00   --output="$LOGS/opera_v7_%A_%a.log" "$BASE/opera_v7_array.sbatch")
FIN=$(sbatch --parsable --dependency="afterany:$JOB"   --job-name=perfail-opera-v7final --cpus-per-task=2 --mem=8G --time=00:15:00   --output="$LOGS/opera_v7_finalize_%j.log" "$BASE/opera_v7_finalize.sbatch")
echo "V7_BATCH_JOB=$JOB V7_FINALIZER_JOB=$FIN"
echo "LOGS=$LOGS/opera_v7_${JOB}_*.log"
