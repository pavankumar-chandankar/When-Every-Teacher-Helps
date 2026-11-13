#!/usr/bin/env bash
#SBATCH --job-name=har-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/har_base-%A_%a.out
#SBATCH --error=logs/har_base-%A_%a.err
set -euo pipefail
module purge >/dev/null 2>&1 || true
unset LD_LIBRARY_PATH CUDA_HOME CUDNN_HOME CUDNN_PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/user/CrossGuard-Fi
WS=$(ws_find crossguard); TAL=/home/user/CrossGuard-Fi/XRFV2-omni-repro
T=$SLURM_ARRAY_TASK_ID
if [ "$T" -lt 3 ]; then MOD=imu; else MOD=wifi; fi
SEED=$(( T % 3 + 1 ))
echo "task $T -> modality $MOD seed $SEED"
.venv/bin/python scripts/har_train.py \
  --manifest manifests/har_fold_a_windows.csv --data-dir "$WS/datasets/xrfv2_plus" \
  --stats-path "$TAL/cache/xrfv2_plus_train_stats_exact.json" \
  --modality "$MOD" --seed "$SEED" --out-dir "$WS/runs_har/${MOD}_seed${SEED}"
