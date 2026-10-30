#!/bin/zsh
# 100-epoch comparison at the baseline's epoch budget: improved recipe on two candidate architectures, then a recipe-only ablation.
set -e
cd "${0:A:h}/.."
.venv/bin/python train.py train --config configs/improved.json --epochs 100 --arch resnet18-64-352 --run runs/improved100-resnet18-64-352 > logs/improved100-resnet18-64-352.log 2>&1
.venv/bin/python train.py train --config configs/improved.json --epochs 100 --arch resnet20-x4 --run runs/improved100-resnet20-x4 > logs/improved100-resnet20-x4.log 2>&1
.venv/bin/python train.py train --config configs/improved.json --epochs 100 --arch resnet18-w32 --run runs/improved100-resnet18-w32 > logs/improved100-resnet18-w32.log 2>&1
