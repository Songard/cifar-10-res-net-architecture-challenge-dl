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

Still to do before submission: add the private-leaderboard scores and a short analysis of them to Section 3.5 and Table 6, and
make the GitHub repository on page 1 public.

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
Later the same evening, six further hour cells that differed from the logged sums by 3-36 s (`r2-ta1800-s2026/s2027`,
`r2-ta2300-s2027`, `r2-ta3000-s2026/s2027`, `r2-kd5-ta1800-s2027`) were set to the sums, so every `a100_hours` value
is now the sum of `train_seconds` over the run's `metrics.jsonl`; no two-decimal number in Table 4 or the text changed.
The scatter figure's annotation now derives the submission count from the data instead of a literal.

## Coherence review (2026-11-06, evening)

A full read-through for internal consistency, with every repeated number re-derived from `figures/data/*` and the
run logs, changed the following statements (all other cross-references checked out):

- Error analysis: the claim that every class improved from the 100-epoch `r18-64-352` to the final KD2 model and that
  cat gained the most was wrong. Re-evaluating that checkpoint per class (scratch script `perclass_r18.py`, EMA weights,
  flip TTA, 179 errors) shows the 60 fewer errors come mostly from bird (+2.8 pp), dog (+2.6) and airplane (+2.0); cat
  gained only +0.8 (automobile +0.2 and frog +0.6 gained less) and ship did not change. The cat-dog counts (41 to 33) were correct and were kept.
- Screening: TrivialAugment and AutoAugment tie only in the two-seed mean (97.34 vs 97.36); per seed they differ by 4
  and 6 images, not "within one image". The Table 5 caption now restricts the "at most two images" statement to the
  layout rows.
- Overview bullet 3 said five held-out images for the KD2 gain; after the held-out correction it is six (0.12 pp).
- Compute: the training-step hours of the 36 cluster runs sum to 56.5 (50.9 without the two full-data runs), so "about
  50" became "about 55/56"; the cluster-job count ("roughly fifty") became "more than a hundred" (111 distinct job ids
  in the candidates' sacct snapshots, smoke tests and prediction jobs included).
- "Counting the teachers' time reverses the comparison" overstated Figure 2b; it now says the advantage disappears,
  with the two cost-accuracy points spelled out. The SAM comparison with Foret et al. now names their augmentations
  (basic, Cutout, AutoAugment) and their two-epoch allowance for the SGD baseline instead of "weaker augmentation"
  (checked against Table 1 of the paper: WRN-28-10 at 1,800 epochs, 3.5 to 2.4, 2.7 to 2.1, 2.2 to 1.6 error).
- Section 3 claimed all alternative layouts were within noise, although `r20-w68` has no held-out score; the sentence
  now points to its public score in Section 5.5. Lesson (1) "misranked" became "exaggerated the differences" (the
  100-epoch lead became a tie, not a reversal). The conclusion's "only change worth more than one SE" is now qualified
  by "once the budget was spent on full-width early stages", since the layout steps of Table 3 (+1.7, +0.3 pp) were
  larger than one SE.

A second, independent read-only pass (a separate agent given only the source and the data files) added these:

- Abstract: "largest single lever" and "all moved the held-out accuracy by less than one SE" were false as written
  (the width step was +1.7 pp; RandAugment, Mixup, CutMix and random convolutions moved it by more than one SE,
  downward). Rewritten with the qualifier "once the network filled the budget" and the regularizers named.
- Lesson (3) claimed the KD/SAM gains "shrink" at equal cost; they are 0.06-0.16 pp at equal epochs and 0.12-0.16 pp
  at equal cost (SAM 1800: zero), below one SE either way. Rewritten accordingly; "the gain disappears once the
  teachers' time is counted" (overview) and "reverses the comparison" (5.2) are now the factual statement that the KD2
  students cost 2.2x a plain 1,800-epoch run for three more held-out images.
- Selection rule: Section 2 said every selection decision used the held-out split, while the finalists are the two
  best public scores. The report now states that the ten submissions with held-out 97.54-97.72% lie within one SE of
  each other and that the leaderboard was used only for that final choice (Section 2, overview bullet 4, 5.5).
- Discussion: by the stated speed criterion r18-64-352 (3.5 s/epoch, same plateau) would have served as well as r20-x4;
  the text now says so and gives the actual reason the layout was kept.
- Overview phases: the three-stage layout was chosen on the laptop (first phase), not on the A100.
- 5.5: three submissions, not two, lack a held-out score (R18 full 200 at 87.8% added); the fit is printed with three
  decimals (1.733, -80.13) so that the quoted 89.0% at 97.6% is reproducible; Table 6 now lists all four scores that
  follow the finals (KD2 seed 2027 at 89.7% was missing).
- Table 4: the 1,600-epoch run (97.48, 1.63 h) was missing from the table and the text; the CE 2300 public score is
  marked as the seed-2027 upload (--/88.6). Table 3's caption says which rows ran on the laptop.
- Error analysis: four confusion pairs have six errors (bird-airplane, cat-deer, horse-dog, automobile-truck), not two.

Later that evening the abstract was rewritten at the author's request: the previous one enumerated experiments and
carried thirteen numbers. The new one follows the four-part shape of the reference papers (problem, approach,
headline result, main finding) with four numbers; the private score and rank go into its third sentence once known.
The subtitle ("Long Schedules, Distillation and Sharpness-Aware Training under a 5M Budget") was dropped at the same
time, in the title and in the PDF metadata.

## Terminology pass (2026-11-06, evening)

At the author's request the whole report was rewritten in standard terminology, without changing any experimental
number (checked by diffing every number against the previous source). Terms replaced: "recipe" by "training procedure",
"plain recipe" and "CE" by "standard training", "schedule length" by "number of epochs" or "training length", "layout"
by "architecture", "budget" by "parameter limit", "held-out" by "validation" (the assignment uses "held-out portion" for
the test set), "families" by "training methods", "pp", "SE" and "TTA" by their full words, "control" by "repeat", and
expressions such as "lever", "ties", "lopsided", "cost-neutral", "smoke tests" and "cloud-bursting". Figure labels follow.

The longer wording pushed the report to nine pages against the assignment's limit of eight, including references.
To stay within eight pages: Figure 3 became a single-column figure with two panels (the zoom on the last 800 epochs was
dropped), Figures 2, 4 and 5 were made smaller, and sentences that only repeated a table or another paragraph were removed
(the parameter and operation differences of the high-resolution architectures, the check of the teachers' split, the
data-loader detail, and one overview sentence on the trade-off that the Discussion covers).

Corrections made during this pass: the second-phase "two seeds for every setting" became "most settings" (SAM has one
seed); the GPU-hour figure is now 56 in both places; the "constant gap" in the abstract is gone (the fit has slope 1.73,
so the gap is not constant); the conclusion no longer says checkpoints behind every number are in the repository, only
the two final models' checkpoints on its Releases page.

## Section structure (2026-11-06, evening)

The assignment asks for an overview, a methodology section and a results section. Former Sections 2-4 are now one
section, 2 Methodology (2.1 Data and evaluation protocol, 2.2 Network architecture, 2.3 Training procedure, 2.4 Knowledge
distillation, 2.5 Sharpness-aware minimization, 2.6 Implementation and computing resources), followed by 3 Results,
4 Discussion and 5 Conclusion. The pros and cons of the architectural choices and the lessons learned stay in the
Discussion because they rest on the results; the opening sentences of Section 2 point there. Section numbers quoted in the
older notes above refer to the earlier structure. Table and figure numbers did not change.

## Content changes requested by the author (2026-11-06, evening)

- Sections 2.4 and 2.5 now open with the problem each method addresses and explain its principle before its parameters. The
  motivations come from the project records: an average of networks is more accurate but exceeds the parameter limit (KD), and
  longer training stopped improving validation accuracy while training accuracy kept rising (SAM). The explanations of the methods
  follow the Hinton et al. and Foret et al. papers.
- Section 2.6 is reduced to the hardware and the two-hour job limit, because it does not affect the submissions.
- Defensive statements were removed throughout: the Limitations paragraph, the remark on regularization not being retuned, the
  remark on being unable to judge the full-data runs and similar hedges. The one-image differences between training logs and
  prediction-time evaluation remain documented in the README.
- Section 3.5 no longer contains the fit of public against validation accuracy, the comparison with Recht et al. or Figure 4 of the
  earlier version; the leaderboard analysis is to be written when the final scores are out. `scatter_figure()` in
  `make_results_figures.py` is kept for that purpose. The abstract and overview no longer claim that validation accuracy predicts
  the leaderboard; they can state it again if the private scores support it.
- Section 3.6 was expanded with a per-image analysis of six models on the validation set. Every number in it is in
  `figures/data/error_analysis.json`, produced by `figures/make_error_analysis.py`, which also draws `hard_examples.pdf`.
  It uses only the labeled validation images and never reads the hidden test images.

Figure numbers changed: Figure 4 is the confusion matrix, Figure 5 (`errors.pdf`, from `make_results_figures.py`) shows errors per class
and shared errors, and Figure 6 shows the misclassified images. Figure 3 is back to three panels.

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

## Private leaderboard results and corrections (2026-11-09)

The private scores came out after the competition closed. Table 6 gained a Private column and the row with the highest private
score, without the Params column (the caption now states the parameter counts). The abstract, the last overview bullet, Section 3.5
and the conclusion report the final result: 87.4% (SAM; KD2 87.1%), rank 27 of 81 teams on the private leaderboard, median 86.9%.
Section 3.5 states three conclusions computed from `figures/data/submissions.csv`, which now has a `private` column.

Two statements about the hidden test images were wrong and were corrected. The overview said the images "may only be used to write
the submission file"; the rule reads "Evaluation images must not be manually labeled or used for training, fine-tuning, or
unauthorized model selection." Section 2.1, Figure 1 and the README said the images were "read once" with no inspection, but on
2026-10-30 they were examined without labels to study the gap between validation and public accuracy. Section 2.1 and the README
now say so, and the README lists every use.

To stay within eight pages: venue names in `references.bib` are abbreviated (CVPR, ICLR, and so on), lessons 5 and 6 are removed because
Section 2.6 and Section 3.5 say the same, the public-score example in lesson 4 and two secondary sentences in Section 3.4 are removed,
and the sentence on how often the two final models disagree is removed from Section 3.5.
