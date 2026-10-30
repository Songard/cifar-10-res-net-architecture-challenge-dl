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
