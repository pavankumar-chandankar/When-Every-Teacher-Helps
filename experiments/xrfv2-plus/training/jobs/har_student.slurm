#!/usr/bin/env bash
#SBATCH --job-name=har-student
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/har_student-%A_%a.out
#SBATCH --error=logs/har_student-%A_%a.err
set -euo pipefail
module purge >/dev/null 2>&1 || true
unset LD_LIBRARY_PATH CUDA_HOME CUDNN_HOME CUDNN_PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/user/CrossGuard-Fi
WS=$(ws_find crossguard); TAL=/home/user/CrossGuard-Fi/XRFV2-omni-repro
SEED=$(( SLURM_ARRAY_TASK_ID + 1 ))
echo "student seed $SEED, teacher imu_seed$SEED"
.venv/bin/python scripts/har_train.py \
  --manifest manifests/har_fold_a_windows.csv --data-dir "$WS/datasets/xrfv2_plus" \
  --stats-path "$TAL/cache/xrfv2_plus_train_stats_exact.json" \
  --modality wifi --teacher-ckpt "$WS/runs_har/imu_seed${SEED}/best.pt" --align-weight 1.0 \
  --seed "$SEED" --out-dir "$WS/runs_har/student_seed${SEED}"
