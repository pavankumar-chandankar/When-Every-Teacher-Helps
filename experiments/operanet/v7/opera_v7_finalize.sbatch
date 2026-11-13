#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/perfail_ops_tmp/perfail_v7"
ROOT="./data/perfail2027/operanet"
PY="$HOME/perfail_ops_tmp/perfail_py311/bin/python"
LOGS="$HOME/perfail_ops_tmp/perfail_logs"
(cd "$BASE" && sha256sum -c SOURCE_SHA256.txt)
echo '=== COHORT COVERAGE ==='
"$PY" "$BASE/operanet_cohort_summary_v7.py" || { echo 'STOP: V7_COHORT_FAILED'; exit 2; }
echo '=== PYTORCH CHECK ==='
PYTRAIN="${PERFAIL_TORCH_PYTHON:-$PY}"
if ! "$PYTRAIN" -c 'import numpy,torch;print("TORCH_OK",torch.__version__)'; then
  echo "NO_TORCH: install torch into $PYTRAIN or set PERFAIL_TORCH_PYTHON; data retained."
  exit 3
fi
"$PYTRAIN" "$BASE/perfail_operanet_matrix_v7.py" selftest
"$PYTRAIN" "$BASE/perfail_operanet_matrix_v7.py" check --root "$ROOT/aligned_v7_10"
if [[ -e "$ROOT/matrix_v7_10" ]]; then echo 'STOP: training output already exists'; exit 4; fi
mkdir "$ROOT/matrix_v7_10"
{
 sha256sum "$BASE/perfail_operanet_matrix_v7.py"
 printf 'PYTHON=%s\n' "$PYTRAIN"
 printf 'EXPS=18,29,32,33,56,57,58,59,60,61\n'
 printf 'EXCLUDED=30,31 missing required Kinect table headers\n'
 printf 'MAT_MCOS_PERSON_ID=unreadable, independent WiFi/UWB IDs plus MAT exp/room/timestamp/label corroboration\n'
 printf 'WINDOW=1.5sec STRIDE=0.5sec\nSEEDS=11,22,33\nEPOCHS=30\n'
 printf 'F1=FIXED_6_CLASS_MACRO_F1; MISSING_TEST_SUPPORT_SCORED_ZERO_AND_REPORTED\n'
} > "$ROOT/matrix_v7_10/PROTOCOL.txt"
JOB=$(sbatch --parsable --job-name=perfail-opera-v7matrix   --array=0-2%2 --cpus-per-task=4 --mem=24G --time=06:00:00   --output="$LOGS/opera_matrix_v7_%A_%a.log" "$BASE/opera_v7_matrix.sbatch")
echo "V7_MATRIX_JOB=$JOB"
echo "RESULT_DIR=$ROOT/matrix_v7_10"
