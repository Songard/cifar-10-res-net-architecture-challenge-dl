# Cluster runs cited in the report

Per-run records of the NYU HPC (single A100) runs behind the report's tables and figures: the resolved training settings
(`config.json`, written by `train.py`), the per-epoch log (`metrics.jsonl`) and, where a submission file was written, its
audit (`submission.audit.json`). Checkpoints, prediction CSVs and cluster stdout are not in the repository; the two final
checkpoints are available separately (see the top-level README). Held-out accuracy is the EMA weights with horizontal-flip
TTA on the fixed 5,000-image split at the last epoch, as logged by the training process on the GPU; `A100 h` is the sum of
the logged training-step seconds (validation, checkpoint writes, queueing and node start-up excluded). The laptop runs the
report also cites live one level up, in `runs/<name>/`.

| Run | Report label | Where | Arch | Recipe | Epochs | Seed | Train imgs | Held-out % | A100 h | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ce-x4-1000-epochseed-s2026` | CE 1000, control | Table 4 | `resnet20-x4` | trivialaugment | 1000 | 2026 | 45,000 | 97.64 | 1.11 | same recipe with the per-epoch seeded policy; public 88.5 |
| `cloud-kd-x4-400-s2026` | KD 400 | Fig. 2a | `resnet20-x4` | trivialaugment, KD a=0.9 T=4 (2 teachers) | 400 | 2026 | 45,000 | 97.40 | 0.23 | epochs 273-400 only: resumed the laptop run runs/kd-x4-400ep-45k (epochs 1-272) on the A100 |
| `cloud-kd-x4-800-s2026` | KD 800 | Fig. 2a | `resnet20-x4` | trivialaugment, KD a=0.9 T=4 (2 teachers) | 800 | 2026 | 45,000 | 97.66 | 1.02 | teachers final-resnet18-64-352-45k and x4-400ep-45k (laptop runs); public 88.7 |
| `cloud-kd-x4-800-s2027` | KD 800 | Fig. 2a | `resnet20-x4` | trivialaugment, KD a=0.9 T=4 (2 teachers) | 800 | 2027 | 45,000 | 97.18 | 1.06 | same teachers, seed 2027 |
| `cloud-r18-800-s2026` | R18 800 | Sec. 5.3 text; KD5 teacher | `resnet18-64-352` | trivialaugment | 800 | 2026 | 45,000 | 97.40 | 0.78 | resnet18-64-352; public 88.8 |
| `cloud-x4-1000-s2026` | CE 1000 | Table 3, Table 4 (CE plain 1000); KD2 teacher | `resnet20-x4` | trivialaugment | 1000 | 2026 | 45,000 | 97.54 | 1.03 | first teacher of the KD2 models; public 89.5 |
| `cloud-x4-1600-s2026` | CE 1600 | Fig. 2a; KD5 teacher | `resnet20-x4` | trivialaugment | 1600 | 2026 | 45,000 | 97.48 | 1.63 |  |
| `cloud-x4-400-s2026` | CE 400 | Fig. 2a; Table 4 text | `resnet20-x4` | trivialaugment | 400 | 2026 | 45,000 | 97.24 | 0.40 |  |
| `cloud-x4-800-s2026` | CE 800 | Fig. 2a; Sec. 5.3 text | `resnet20-x4` | trivialaugment | 800 | 2026 | 45,000 | 97.30 | 0.82 |  |
| `cloud-x4-800-s2027` | CE 800 | Fig. 2a; Sec. 5.3 text; KD2 teacher | `resnet20-x4` | trivialaugment | 800 | 2027 | 45,000 | 97.50 | 0.84 | second teacher of the KD2 models |
| `r2-aa1800-s2026` | AutoAugment 1800 | Table 4 | `resnet20-x4` | autoaugment | 1800 | 2026 | 45,000 | 97.36 | 2.18 |  |
| `r2-aa1800-s2027` | AutoAugment 1800 | Table 4 | `resnet20-x4` | autoaugment | 1800 | 2027 | 45,000 | 97.52 | 2.17 |  |
| `r2-aa400-s2026` | AutoAugment 400 | Table 5 | `resnet20-x4` | autoaugment | 400 | 2026 | 45,000 | 97.28 | 0.48 |  |
| `r2-aa400-s2027` | AutoAugment 400 | Table 5 | `resnet20-x4` | autoaugment | 400 | 2027 | 45,000 | 97.44 | 0.49 |  |
| `r2-highres400-s2026` | highres 400 | Table 5 | `resnet-highres` | trivialaugment | 400 | 2026 | 45,000 | 97.34 | 0.58 | resnet-highres layout |
| `r2-highres400-s2027` | highres 400 | Table 5 | `resnet-highres` | trivialaugment | 400 | 2027 | 45,000 | 97.30 | 0.58 | resnet-highres layout |
| `r2-kd2-ta1800-full-s2027` | KD2 full 1800 | Sec. 5.5 text | `resnet20-x4` | trivialaugment, KD a=0.9 T=4 (2 teachers) | 1800 | 2027 | 50,000 | fit, not held out | 2.76 | all 50,000 images, no held-out score; public 88.8 |
| `r2-kd2-ta1800-s2026` | KD2 1800 (final submission) | Tables 3, 4, 6; Figs. 3, 4, 5 | `resnet20-x4` | trivialaugment, KD a=0.9 T=4 (2 teachers) | 1800 | 2026 | 45,000 | 97.62 | 2.53 | public 90.1 |
| `r2-kd2-ta1800-s2027` | KD2 1800 | Table 4 | `resnet20-x4` | trivialaugment, KD a=0.9 T=4 (2 teachers) | 1800 | 2027 | 45,000 | 97.66 | 2.48 | public 89.7 |
| `r2-kd5-ta1800-s2026` | KD5 1800 | Table 4, Table 6 | `resnet20-x4` | trivialaugment, KD a=0.9 T=4 (5 teachers) | 1800 | 2026 | 45,000 | 97.64 | 3.43 | public 89.7 |
| `r2-kd5-ta1800-s2027` | KD5 1800 | Table 4 | `resnet20-x4` | trivialaugment, KD a=0.9 T=4 (5 teachers) | 1800 | 2027 | 45,000 | 97.66 | 3.36 | public 88.8 |
| `r2-preact400-s2026` | preact 400 | Table 5 | `preact-highres` | trivialaugment | 400 | 2026 | 45,000 | 97.40 | 0.57 | preact-highres layout |
| `r2-preact400-s2027` | preact 400 | Table 5 | `preact-highres` | trivialaugment | 400 | 2027 | 45,000 | 97.34 | 0.57 | preact-highres layout |
| `r2-ra400-s2026` | RandAugment (2, 9) 400 | Table 5 | `resnet20-x4` | randaugment | 400 | 2026 | 45,000 | 97.02 | 0.50 |  |
| `r2-ra400-s2027` | RandAugment (2, 9) 400 | Table 5 | `resnet20-x4` | randaugment | 400 | 2027 | 45,000 | 96.94 | 0.49 |  |
| `r2-ta1800-s2026` | CE 1800 | Table 3, Table 4, Fig. 3 | `resnet20-x4` | trivialaugment | 1800 | 2026 | 45,000 | 97.66 | 1.96 | plain recipe; public 88.6 |
| `r2-ta1800-s2027` | CE 1800 | Table 4 | `resnet20-x4` | trivialaugment | 1800 | 2027 | 45,000 | 97.50 | 1.99 |  |
| `r2-ta2300-s2026` | CE 2300 | Table 4 | `resnet20-x4` | trivialaugment | 2300 | 2026 | 45,000 | 97.52 | 2.54 |  |
| `r2-ta2300-s2027` | CE 2300 | Table 4 | `resnet20-x4` | trivialaugment | 2300 | 2027 | 45,000 | 97.52 | 2.53 | cloud log 97.52; laptop re-evaluation 97.54 (used in the report); public 88.6 |
| `r2-ta3000-s2026` | CE 3000 | Table 4 | `resnet20-x4` | trivialaugment | 3000 | 2026 | 45,000 | 97.52 | 3.32 |  |
| `r2-ta3000-s2027` | CE 3000 | Table 4 | `resnet20-x4` | trivialaugment | 3000 | 2027 | 45,000 | 97.46 | 3.30 |  |
| `r2-ta400-s2026` | TA 400 | Table 5 | `resnet20-x4` | trivialaugment | 400 | 2026 | 45,000 | 97.36 | 0.43 | screening pair, seed 2026 |
| `r2-ta400-s2027` | TA 400 | Table 5 | `resnet20-x4` | trivialaugment | 400 | 2027 | 45,000 | 97.32 | 0.45 | screening pair, seed 2027 |
| `sam-x4-1000-s2026` | SAM 1000 | Table 4 | `resnet20-x4` | trivialaugment, SAM rho=0.05 | 1000 | 2026 | 45,000 | 97.72 | 1.77 | public 89.3 |
| `sam-x4-1800-s2026` | SAM 1800 (final submission) | Tables 3, 4, 6; Figs. 3, 4 | `resnet20-x4` | trivialaugment, SAM rho=0.05 | 1800 | 2026 | 45,000 | 97.58 | 3.20 | public 90.0 |
| `w68-full-s2026` | w68 full 1800 | Table 1, Table 6 | `resnet20-w68` | trivialaugment | 1800 | 2026 | 50,000 | fit, not held out | 2.85 | resnet20-w68 on all 50,000 images, no held-out score; public 89.7 |

Teacher checkpoints named in a `config.json` (`../runs/<name>/last.pt` on the cluster, `runs/<name>/last.pt` on the laptop)
refer to the runs of the same name in this directory or in `runs/`. The `code_sha256` entry of each `config.json` records the
revision of `train.py`/`model.py` that produced the run; the top-level README maps those hashes to the published sources.
