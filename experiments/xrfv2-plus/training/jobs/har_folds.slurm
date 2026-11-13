#!/usr/bin/env bash
#SBATCH --job-name=har-folds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-35%12
#SBATCH --output=logs/har_folds-%A_%a.out
#SBATCH --error=logs/har_folds-%A_%a.err
set -euo pipefail
module purge >/dev/null 2>&1 || true
unset LD_LIBRARY_PATH CUDA_HOME CUDNN_HOME CUDNN_PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/user/CrossGuard-Fi
WS=$(ws_find crossguard); TAL=/home/user/CrossGuard-Fi/XRFV2-omni-repro
T=$SLURM_ARRAY_TASK_ID
if [ "$T" -lt 18 ]; then FOLD=fold_b_near_semantic; else FOLD=fold_c_cross_room; fi
case $(( (T % 18) / 3 )) in 0) MOD=imu;; 1) MOD=pose3d;; 2) MOD=pose2d;; 3) MOD=video;; 4) MOD=airpods;; 5) MOD=wifi;; esac
SEED=$(( T % 3 + 1 ))
echo "task $T -> $FOLD $MOD seed $SEED"
.venv/bin/python scripts/har_train.py \
  --manifest "manifests/har_${FOLD}_windows.csv" --data-dir "$WS/datasets/xrfv2_plus" \
  --stats-path "$TAL/cache/xrfv2_plus_train_stats_exact.json" \
  --cache-path "$WS/datasets/xrfv2_plus/modality_cache_50hz.h5" \
  --cache-stats "$WS/datasets/xrfv2_plus/modality_cache_50hz.h5.stats.json" \
  --modality "$MOD" --seed "$SEED" --out-dir "$WS/runs_har_folds/$FOLD/${MOD}_seed${SEED}"
