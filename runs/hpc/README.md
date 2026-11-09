# Cluster runs cited in the report

Records of the runs on NYU's high-performance computing cluster (one A100 GPU per job) behind the report's tables and
figures. Each folder holds the resolved training settings (`config.json`, written by `train.py`), the per-epoch log
(`metrics.jsonl`) and, where a submission file was written, its audit (`submission.audit.json`). Checkpoints, prediction
CSVs and the cluster's console output are not in the repository; the checkpoints of the two final models are published
separately (see the top-level README). The laptop runs the report also cites are one level up, in `runs/<name>/`.

- **Validation %** is the accuracy on the fixed 5,000 validation images at the last epoch, with the averaged (EMA) weights
  and flip test-time augmentation, as logged by the training process (`tta_valid_accuracy`).
- **A100 hours** is the sum of the logged training-step times (`train_seconds`); evaluation, checkpointing, queueing and
  node start-up are not included.
- **Public %** and **Private %** are Kaggle's scores on the two halves of the hidden test set, for the runs that were
  uploaded (`artifacts/kaggle_submissions.jsonl`).

| Run | In the report | Where | Architecture | Epochs | Seed | Training images | Validation % | A100 hours | Public % | Private % | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ce-x4-1000-epochseed-s2026` | standard training, 1,000 epochs, repeat | Table 4; Fig. 2 | `resnet20-x4` | 1,000 | 2026 | 45,000 | 97.64 | 1.11 | 88.5 | 86.8 | differs from the first 1,000-epoch run only in its random numbers; evaluated again at prediction time: 97.62, one image below the log |
| `cloud-kd-x4-400-s2026` | KD, 400 epochs | Fig. 2 | `resnet20-x4` | 400 | 2026 | 45,000 | 97.40 | 0.23 |  |  | epochs 273-400 only: it resumed the laptop run `runs/kd-x4-400ep-45k` (epochs 1-272) on the A100 |
| `cloud-kd-x4-800-s2026` | KD, 800 epochs | Fig. 2 | `resnet20-x4` | 800 | 2026 | 45,000 | 97.66 | 1.02 | 88.7 | 86.6 | teachers `final-resnet18-64-352-45k` and `x4-400ep-45k`, both trained on the laptop |
| `cloud-kd-x4-800-s2027` | KD, 800 epochs | Fig. 2 | `resnet20-x4` | 800 | 2027 | 45,000 | 97.18 | 1.06 |  |  | same teachers |
| `cloud-r18-800-s2026` | `r18-64-352`, standard training, 800 epochs | Table 1; Fig. 2; Sec. 2.4, 3.3 | `resnet18-64-352` | 800 | 2026 | 45,000 | 97.40 | 0.78 | 88.8 | 86.0 | KD5 teacher |
| `cloud-x4-1000-s2026` | standard training, 1,000 epochs | Tables 3, 4, 6; Fig. 2; Sec. 2.4 | `resnet20-x4` | 1,000 | 2026 | 45,000 | 97.54 | 1.03 | 89.5 | 86.2 | first teacher of the KD2 models |
| `cloud-x4-1600-s2026` | standard training, 1,600 epochs | Table 4; Fig. 2; Sec. 2.4 | `resnet20-x4` | 1,600 | 2026 | 45,000 | 97.48 | 1.63 |  |  | KD5 teacher |
| `cloud-x4-400-s2026` | standard training, 400 epochs | Fig. 2 | `resnet20-x4` | 400 | 2026 | 45,000 | 97.24 | 0.40 |  |  |  |
| `cloud-x4-800-s2026` | standard training, 800 epochs | Fig. 2; Sec. 3.2, 3.3 | `resnet20-x4` | 800 | 2026 | 45,000 | 97.30 | 0.82 |  |  |  |
| `cloud-x4-800-s2027` | standard training, 800 epochs | Fig. 2; Sec. 2.4, 3.2, 3.3 | `resnet20-x4` | 800 | 2027 | 45,000 | 97.50 | 0.84 | 88.2 | 86.7 | second teacher of the KD2 models |
| `r2-aa1800-s2026` | AutoAugment, 1,800 epochs | Table 4; Fig. 2 | `resnet20-x4` | 1,800 | 2026 | 45,000 | 97.36 | 2.18 |  |  |  |
| `r2-aa1800-s2027` | AutoAugment, 1,800 epochs | Table 4; Fig. 2 | `resnet20-x4` | 1,800 | 2027 | 45,000 | 97.52 | 2.17 |  |  |  |
| `r2-aa400-s2026` | AutoAugment, 400 epochs | Table 5; Fig. 2 | `resnet20-x4` | 400 | 2026 | 45,000 | 97.28 | 0.48 |  |  |  |
| `r2-aa400-s2027` | AutoAugment, 400 epochs | Table 5; Fig. 2 | `resnet20-x4` | 400 | 2027 | 45,000 | 97.44 | 0.49 |  |  |  |
| `r2-highres400-s2026` | `highres` architecture, 400 epochs | Tables 1, 5; Fig. 2 | `resnet-highres` | 400 | 2026 | 45,000 | 97.34 | 0.58 |  |  |  |
| `r2-highres400-s2027` | `highres` architecture, 400 epochs | Tables 1, 5; Fig. 2 | `resnet-highres` | 400 | 2027 | 45,000 | 97.30 | 0.58 |  |  |  |
| `r2-kd2-ta1800-full-s2027` | KD2, 1,800 epochs, all 50,000 images | Table 6; Sec. 3.5 | `resnet20-x4` | 1,800 | 2027 | 50,000 | none (trained on them) | 2.76 | 88.8 | 87.6 | no validation score; the highest private score of the 21 submissions |
| `r2-kd2-ta1800-s2026` | **KD2, 1,800 epochs (final submission)** | Tables 3, 4, 6; Figs. 2-6; Sec. 3.5, 3.6 | `resnet20-x4` | 1,800 | 2026 | 45,000 | 97.62 | 2.53 | 90.1 | 87.1 |  |
| `r2-kd2-ta1800-s2027` | KD2, 1,800 epochs | Tables 4, 6; Figs. 2, 5, 6; Sec. 3.6 | `resnet20-x4` | 1,800 | 2027 | 45,000 | 97.66 | 2.48 | 89.7 | 86.4 |  |
| `r2-kd5-ta1800-s2026` | KD5, 1,800 epochs | Tables 4, 6; Fig. 2 | `resnet20-x4` | 1,800 | 2026 | 45,000 | 97.64 | 3.43 | 89.7 | 86.5 |  |
| `r2-kd5-ta1800-s2027` | KD5, 1,800 epochs | Table 4; Fig. 2 | `resnet20-x4` | 1,800 | 2027 | 45,000 | 97.66 | 3.36 | 88.8 | 87.3 |  |
| `r2-preact400-s2026` | `preact` architecture, 400 epochs | Tables 1, 5; Fig. 2 | `preact-highres` | 400 | 2026 | 45,000 | 97.40 | 0.57 |  |  |  |
| `r2-preact400-s2027` | `preact` architecture, 400 epochs | Tables 1, 5; Fig. 2 | `preact-highres` | 400 | 2027 | 45,000 | 97.34 | 0.57 |  |  |  |
| `r2-ra400-s2026` | RandAugment (2, 9), 400 epochs | Table 5; Fig. 2 | `resnet20-x4` | 400 | 2026 | 45,000 | 97.02 | 0.50 |  |  |  |
| `r2-ra400-s2027` | RandAugment (2, 9), 400 epochs | Table 5; Fig. 2 | `resnet20-x4` | 400 | 2027 | 45,000 | 96.94 | 0.49 |  |  |  |
| `r2-ta1800-s2026` | standard training, 1,800 epochs | Tables 3, 4; Figs. 2, 3, 5, 6; Sec. 3.6 | `resnet20-x4` | 1,800 | 2026 | 45,000 | 97.66 | 1.96 | 88.6 | 87.0 |  |
| `r2-ta1800-s2027` | standard training, 1,800 epochs | Table 4; Fig. 2 | `resnet20-x4` | 1,800 | 2027 | 45,000 | 97.50 | 1.99 |  |  |  |
| `r2-ta2300-s2026` | standard training, 2,300 epochs | Table 4; Fig. 2 | `resnet20-x4` | 2,300 | 2026 | 45,000 | 97.52 | 2.54 |  |  |  |
| `r2-ta2300-s2027` | standard training, 2,300 epochs | Table 4; Fig. 2 | `resnet20-x4` | 2,300 | 2027 | 45,000 | 97.52 | 2.53 | 88.6 | 87.3 | evaluated again at prediction time: 97.54, one image above the log |
| `r2-ta3000-s2026` | standard training, 3,000 epochs | Table 4; Fig. 2 | `resnet20-x4` | 3,000 | 2026 | 45,000 | 97.52 | 3.32 |  |  |  |
| `r2-ta3000-s2027` | standard training, 3,000 epochs | Table 4; Fig. 2 | `resnet20-x4` | 3,000 | 2027 | 45,000 | 97.46 | 3.30 |  |  |  |
| `r2-ta400-s2026` | standard training (TrivialAugment), 400 epochs | Table 5; Fig. 2 | `resnet20-x4` | 400 | 2026 | 45,000 | 97.36 | 0.43 |  |  |  |
| `r2-ta400-s2027` | standard training (TrivialAugment), 400 epochs | Table 5; Fig. 2 | `resnet20-x4` | 400 | 2027 | 45,000 | 97.32 | 0.45 |  |  |  |
| `sam-x4-1000-s2026` | SAM, 1,000 epochs | Table 4; Fig. 2 | `resnet20-x4` | 1,000 | 2026 | 45,000 | 97.72 | 1.77 | 89.3 | 86.6 |  |
| `sam-x4-1800-s2026` | **SAM, 1,800 epochs (final submission)** | Tables 3, 4, 6; Figs. 2, 3, 5, 6; Sec. 3.5, 3.6 | `resnet20-x4` | 1,800 | 2026 | 45,000 | 97.58 | 3.20 | 90.0 | 87.4 | the final test accuracy of the project |
| `w68-full-s2026` | `r20-w68`, standard training, 1,800 epochs, all 50,000 images | Tables 1, 6 | `resnet20-w68` | 1,800 | 2026 | 50,000 | none (trained on them) | 2.85 | 89.7 | 87.0 | no validation score |

The 36 runs add up to 56.5 A100 hours of training steps (the "about 56 GPU-hours in 36 runs" of Section 2.6).
KD2 and KD5 are distillation from two and from five of our own networks; the teachers named in a `config.json`
(`../runs/<name>/last.pt` on the cluster, `runs/<name>/last.pt` on the laptop) are the runs of the same name here or in
`runs/`. The `code_sha256` entry of each `config.json` records the revision of `train.py` and `model.py` that produced the
run; the top-level README maps those hashes to the published sources.
