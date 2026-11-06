#!/bin/bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=/share/apps/images/cuda-12.9.1-cudnn-9.10.2-ubuntu-24.04.2.sif
exec singularity exec --nv "$IMAGE" "$ROOT/venv/bin/python" "$@"
