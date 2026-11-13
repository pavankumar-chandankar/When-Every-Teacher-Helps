#!/usr/bin/env bash
#SBATCH --job-name=har-mm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/har_mm-%A_%a.out
#SBATCH --error=logs/har_mm-%A_%a.err
set -euo pipefail
module purge >/dev/null 2>&1 || true
unset LD_LIBRARY_PATH CUDA_HOME CUDNN_HOME CUDNN_PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/user/CrossGuard-Fi
WS=$(ws_find crossguard); TAL=/home/user/CrossGuard-Fi/XRFV2-omni-repro
T=$SLURM_ARRAY_TASK_ID
case $(( (T % 12) / 3 )) in 0) MOD=airpods;; 1) MOD=pose2d;; 2) MOD=pose3d;; 3) MOD=video;; esac
SEED=$(( T % 3 + 1 ))
EXTRA=""; TAG="${MOD}_seed${SEED}"
if [ "$T" -ge 12 ]; then
  EXTRA="--teacher-ckpt $WS/runs_har/imu_seed${SEED}/best.pt --align-weight 1.0"
  TAG="${MOD}_from_imu_seed${SEED}"
fi
echo "task $T -> $TAG"
.venv/bin/python scripts/har_train.py \
  --manifest manifests/har_fold_a_windows.csv --data-dir "$WS/datasets/xrfv2_plus" \
  --stats-path "$TAL/cache/xrfv2_plus_train_stats_exact.json" \
  --cache-path "$WS/datasets/xrfv2_plus/modality_cache_50hz.h5" \
  --cache-stats "$WS/datasets/xrfv2_plus/modality_cache_50hz.h5.stats.json" \
  --modality "$MOD" --seed "$SEED" $EXTRA --out-dir "$WS/runs_har/$TAG"
