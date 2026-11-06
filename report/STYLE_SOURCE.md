# Report style and build

The course links the AAAI Press template:
https://www.overleaf.com/latex/templates/aaai-press-latex-template/jymjdgdpdmxp

The previously prepared report source required `aaai.sty` and `aaai.bst`, but neither was present locally.
The included files were downloaded unchanged on 2026-10-29 from this public mirror:

- https://raw.githubusercontent.com/quarbby/FYP-LaTeX-Template/master/aaai/aaai.sty
- https://raw.githubusercontent.com/quarbby/FYP-LaTeX-Template/master/aaai/aaai.bst

The style identifies itself as the 1 January 2015 AAAI conference style. Its original notices remain intact.
These are not claimed to be a byte-for-byte export of the course's Overleaf project. The failed legacy AAAI
author-kit download was not used, and certificate warnings were not bypassed.

From this directory, run `tectonic --keep-logs main.tex`. Alternatively, use pdfLaTeX, BibTeX, and two more
pdfLaTeX passes. The source selects OT1 encoding under XeTeX to preserve the template's Times, Courier,
bold and italic fonts, and handles PDF metadata for both engines. The style itself is unmodified.

On 2026-10-29 the report compiled to five US Letter pages. Every page was rendered and visually inspected;
the table overflow and font substitution found in the first build were fixed. The remaining first-page
GitHub placeholder is a submission blocker. `main.pdf` is a local draft, not evidence of a Brightspace
submission or a final approved report.

## Build notes (2026-11-06)

The report was rewritten on 2026-11-06 around the final competition results. It was compiled with Tectonic 0.15.0
(a copy of the binary that an earlier session had downloaded from the Tectonic GitHub release; any pdfLaTeX + BibTeX
toolchain with the AAAI 2015 style works as well) and renders to exactly eight US Letter pages with no overfull boxes.

Figures are generated, not drawn by hand:

- `figures/make_pipeline.py` draws Figure 1 (`pipeline.pdf`).
- `figures/make_results_figures.py` draws Figures 2-5 (`schedule.pdf`, `curves.pdf`, `scatter.pdf`, `confusion.pdf`)
  from `figures/data/experiments.csv`, `figures/data/submissions.csv`, `figures/data/curves_1800.json` and
  `figures/data/heldout_eval.json`. The `source` column of `experiments.csv` names the run directory or evidence
  archive every number comes from; `heldout_eval.json` was produced by re-evaluating the final checkpoints on the
  fixed 5,000-image split with the training code (EMA weights, flip TTA).

Run both scripts with the project virtual environment (`../.venv/bin/python`) from this directory's `figures/` folder,
then `tectonic --keep-logs main.tex` here.

Still to update before submission: the private-leaderboard scores of the two final submissions (released after the
competition closes on 2026-11-06 23:59 EST; Table 6 and the Limitations paragraph say "not available"), the final
public-leaderboard rank and median in the "Final selection" paragraph (the figures quoted are from 16:33 UTC on
2026-11-06; by 20:53 UTC the field had grown from 64 to 80 teams and the same score ranked 27th), and the GitHub
repository on page 1 must be public.

## Correction of two held-out values (2026-11-06, evening)

The session that prepared the public repository cross-checked `figures/data/experiments.csv` against the published
training logs (`runs/hpc/<run>/metrics.jsonl`) and found two rows that carried the prediction-time evaluation of the
checkpoint instead of the training log's final-epoch evaluation of the same weights: `r2-ta2300-s2027` (log 97.52,
prediction-time 97.54) and `ce-x4-1000-epochseed-s2026` (log 97.64, prediction-time 97.62). The 22 other cloud runs
that have both numbers agree exactly. Since the Limitations paragraph states that training-log numbers are reported,
both rows were changed to the log values in `experiments.csv` and `submissions.csv`, Table 4, the schedule paragraph,
the overview and the matched-cost comparison were updated, the Limitations sentence now describes the two cases
precisely, and Figures 2 and 4 were regenerated (the held-out/public fit was re-derived from the corrected data). The
A100 hours of `r2-ta400-s2027` were corrected from 0.43 (the seed-2026 value) to 0.451 (sum of its logged step times).

## Competition and benchmark reports consulted (2026-11-06)

Besides the papers listed above, the following were fetched and read in full to settle how competition-style
reports present results; the extracted texts were kept only in a temporary folder.

- Jordan (2024), "94% on CIFAR-10 in 3.29 Seconds on a Single GPU", arXiv:2404.00498. Reports mean accuracy of the
  final model over n runs (n stated per table), writes each method paragraph as a cumulative step with its new
  epochs-to-target and A100-seconds, and separates the run-to-run standard deviation measured on the fixed test set
  (0.157 pp) from the standard deviation of the true expected accuracy (0.037 pp). The report cites this in the
  evaluation protocol as the measured basis for not reading sub-SE differences as effects.
- Coleman et al. (2017), DAWNBench (NIPS 2017 ML Systems Workshop). Time-to-accuracy with a 94% CIFAR-10 threshold;
  single runs, no error bars; results carried by figures, with a cumulative/leave-one-out speed-up table.
- Ha, Liu and Liu (2020), SIIM-ISIC melanoma winning solution, arXiv:2010.05351. One table with cross-validation,
  public and private scores per model, and the spread of each metric across models; validation strategy is described
  before the model. Dieleman, Willett and Dambre (2015), Galaxy Challenge winner (MNRAS): public and private scores
  side by side for every ablation step. Our Table 6 follows this convention (held-out, public and, once released,
  private score per submission).
- Recht et al. (2019), re-read for the linear fit (slope 1.69, 95% bootstrap CI [1.63, 1.76], offset -72.7), the
  2,000-image size of CIFAR-10.1 and the Clopper-Pearson intervals used in its Table 1 and Figure 1.
- Page (2018-2019), "How to train your ResNet", could not be fetched: the myrtle.ai pages return 404 and the Internet
  Archive copies were served only as a verification page. The bibliography entry points to the archive and to the
  companion repository; nothing in the report is attributed to the posts' text.
