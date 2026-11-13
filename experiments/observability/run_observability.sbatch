#!/bin/bash
#SBATCH --job-name=pose-observe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=pose-observe-%j.out
set -euo pipefail

: "${DATASET:?set DATASET=operanet or mmfi}"
: "${INPUT_DIR:?set INPUT_DIR to verified matrix cache}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a new result directory}"
: "${PYTHON_BIN:?set PYTHON_BIN to your existing Python environment}"
: "${SCRIPT_DIR:?set SCRIPT_DIR to checkout path containing this file}"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/pose_observability.py" run \
  --dataset "$DATASET" --input "$INPUT_DIR" --out "$OUTPUT_DIR" \
  --seeds 11 22 33 --epochs 30 --batch 64 --device cuda
