#!/bin/zsh
# Final recipe (configs/final.json, 200 epochs): submission model on all 50,000 images, its predictions, then the held-out 45k twin.
set -e
cd "${0:A:h}/.."
.venv/bin/python train.py train --config configs/final.json --full-train --run runs/final-resnet18-64-352-50k > logs/final-resnet18-64-352-50k.log 2>&1
.venv/bin/python train.py predict --checkpoint runs/final-resnet18-64-352-50k/last.pt --weights ema --tta flip --output runs/final-resnet18-64-352-50k/submission.csv > logs/final-resnet18-64-352-50k-predict.log 2>&1
.venv/bin/python train.py train --config configs/final.json --run runs/final-resnet18-64-352-45k > logs/final-resnet18-64-352-45k.log 2>&1
