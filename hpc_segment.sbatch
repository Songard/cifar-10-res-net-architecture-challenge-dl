#!/bin/bash
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
# Usage: sbatch hpc_segment.sbatch RUN ABSOLUTE_STOP_EPOCH [train.py options]
run=$1
stop=$2
shift 2
[[ "$stop" =~ ^[1-9][0-9]*$ ]]
[[ "${SLURM_JOB_ACCOUNT:-}" == cs_gy_6923-2026fa ]]
[[ $(date +%s) -lt $(TZ=America/New_York date -d '2026-11-05 12:00:00' +%s) ]]
mkdir -p "$run"
# A site requeue or accidental second chain must not write the same checkpoint concurrently.
exec 9>"$run/.writer.lock"
flock -n 9
resume=()
if [[ -f "$run/last.pt" ]]; then
    resume=(--resume "$run/last.pt")
fi
date --iso-8601=seconds
hostname -f
printf 'SLURM_JOB_ID=%s STOP_EPOCH=%s\n' "$SLURM_JOB_ID" "$stop"
git rev-parse HEAD
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# Total --epochs is fixed across all segments; only --stop-after advances.
exec bash hpc_python.sh -u train.py train "$@" --run "$run" --stop-after "$stop" "${resume[@]}" --device cuda
