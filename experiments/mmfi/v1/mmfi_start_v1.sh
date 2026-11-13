#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/perfail_ops_tmp"
LOG="$BASE/perfail_logs"
PY="$BASE/perfail_py311/bin/python"
ROOT="./data/perfail2027/mmfi"
mkdir -p "$LOG" "$ROOT"
"$PY" -c 'from PIL import Image' 2>/dev/null || "$PY" -m pip install --no-cache-dir pillow
"$PY" -c 'import numpy,scipy,torch; from PIL import Image; print("MMFI_ENV_READY",torch.__version__)'
"$PY" -m py_compile "$BASE/mmfi_compact_v1.py" "$BASE/mmfi_matrix_v1.py"
"$PY" "$BASE/mmfi_compact_v1.py" self-test
"$PY" "$BASE/mmfi_matrix_v1.py" self-test
if [ -e "$ROOT/matrix_v1/seed_11.json" ] || [ -e "$ROOT/matrix_v1/seed_22.json" ] || [ -e "$ROOT/matrix_v1/seed_33.json" ]; then
  echo "STOP: existing matrix seeds. Do not duplicate training."; exit 1
fi
# One Slurm processing job, auto-submitting seed array only after success.
cat > "$BASE/mmfi_prep_then_train_v1.sh" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/perfail_ops_tmp"
PY="$BASE/perfail_py311/bin/python"
LOG="$BASE/perfail_logs"
ROOT="./data/perfail2027/mmfi"
"$PY" -u "$BASE/mmfi_compact_v1.py" prepare --out "$ROOT/compact_v1"
[ -f "$ROOT/compact_v1/manifest.json" ] || { echo "NO_COMPLETE_MANIFEST"; exit 1; }
JOB=$(sbatch --parsable --array=0-2%3 --job-name=perfail-mmfi-matrix \
  --cpus-per-task=4 --mem=12G --time=03:00:00 \
  --output="$LOG/mmfi_matrix_%A_%a.log" \
  "$BASE/mmfi_train_array_v1.sh")
echo "MMFI_MATRIX_JOB=$JOB"
INNER
chmod +x "$BASE/mmfi_prep_then_train_v1.sh"
JOB=$(sbatch --parsable --job-name=perfail-mmfi-compact \
  --cpus-per-task=4 --mem=16G --time=06:00:00 \
  --output="$LOG/mmfi_compact_%j.log" "$BASE/mmfi_prep_then_train_v1.sh")
echo "MMFI_PREPROCESS_JOB=$JOB"
echo "LOG=$LOG/mmfi_compact_${JOB}.log"
echo "AFTER_PREPROCESS_SUCCESS: the log prints MMFI_MATRIX_JOB."
echo "No MATLAB; no bulk ZIP extraction; no frozen SensorTAL files modified."
