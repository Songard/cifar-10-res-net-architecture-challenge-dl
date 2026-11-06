#!/bin/bash
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Submit from the repository root after validating this interpreter on the node.
PYTHON=${PYTHON:-python3}
export PYTHONUNBUFFERED=1
date --iso-8601=seconds
hostname -f
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-interactive}"
git rev-parse HEAD
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda); print("GPU", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())'

# Experiment settings remain ordinary train.py arguments, including explicit resume.
exec "$PYTHON" train.py train "$@" --device cuda
