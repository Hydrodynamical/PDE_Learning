# Reproducing the paper figures

The main analysis uses the public Kaggle dataset
[Elliptic Differential Equation LLM Reasoning, version 1](https://www.kaggle.com/datasets/jkmiller/elliptic-differential-equation-llm-reasoning/versions/1),
released on 16 April 2026. It contains 180 scored run JSON files and a sweep
summary: six models × three difficulties × seeds 1–10. The original
[write-up](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/can-language-models-learn-a-pde#3442067)
provides the project context.

## Run the analysis

From the repository root, using the Python 3.12 environment in the main README:

```bash
curl -L --fail \
  'https://www.kaggle.com/api/v1/datasets/download/jkmiller/elliptic-differential-equation-llm-reasoning?datasetVersionNumber=1' \
  -o /tmp/pde-runs-v1.zip

python scripts/reproduce_paper_figures.py --archive /tmp/pde-runs-v1.zip
```

The script does not make network or model calls. It reads the ZIP without
extracting or executing its contents and verifies the SHA-256 digest against
[data_manifest.json](data_manifest.json). It also checks cell sizes, shared
problem instances, recorded settings, finite scores, and agreement with the
archived sweep summary. A different archive fails validation.

To verify and regenerate only the numerical summary:

```bash
python scripts/reproduce_paper_figures.py \
  --archive /tmp/pde-runs-v1.zip --summary-only
```

Outputs are [summary.json](summary.json) and five vector figures under `images/`:
`paper_success.pdf`, `paper_actions.pdf`, `paper_improvement.pdf`,
`paper_diversity.pdf`, and `paper_calibration.pdf`. Original PNGs are retained as
historical assets; the manuscript uses the regenerated figures.

## What is measured

- **Success:** total absolute coefficient error
  `E = sum(max(abs(predicted_q - true_q)))` over `q = a, b, c, f`, with thresholds
  `E < 0.01` and `E < 0.1`. These are coefficient errors, not relative or
  held-out-grid function errors.
- **Actions:** separate median counts of solves, checks, and term-integral calls.
  Their stacked sum need not equal the median total action count.
- **Improvement:** the archived first-to-last solve error ratio. Missing ratios
  retain the historical plotting default of one. A large ratio need not imply
  accurate final coefficients.
- **Diversity:** archived normalized query-family entropy. Expression categories
  are heuristic and do not establish linear independence of measurements. The
  archived taxonomy is measured at the final solve; later queries in six runs
  do not enter that entropy.
- **Calibration:** complete confidence reports for `a`, `b`, and `c` paired to
  the latest strictly preceding solve in the assistant transcript, with correctness
  defined by absolute coefficient error below `0.1`. Reports without a preceding
  solve are excluded. Bins are disjoint, with confidence one in the final bin.
  The summary records bin counts and mean confidence; observations within a run
  are dependent.

The recovered logs correct the older strict-threshold chart: GPT-5.4 mini has
**2/10** hard successes, confirmed by individual logs and the sweep summary.
The archive also establishes basis sizes **2/3/4**, rather than the write-up's
3/5/8. All 180 runs record `prompt_condition: standard`; the write-up calls the
augmented prompt enhanced.

The confidence reanalysis also fixes two historical plotting issues: matching a
report to a future solve and counting exact confidence-bin endpoints in two bins.
The archive contains 733 reports, of which 707 have a preceding solve and 26 do
not. The 707 paired reports contribute 2,121 coefficient observations. The
reanalysis changes the matched solve for 71 paired reports; 827 paired
coefficient observations lie on interior boundaries that the older code could
count twice. The fixed confidence-order plot and legacy sensitivity-based
monitoring/control statistics are distinct diagnostics and are not substituted
for these calibration curves.

## Scope and provenance

The archive includes scored records, not a complete failed-attempt/retry history.
Temperature and exact implementation versions are not recorded per run. The
separate [PDE Library snapshot, version 9](https://www.kaggle.com/datasets/jkmiller/pde-library/versions/9)
postdates some records and is not asserted to be the exact source for every run.
The manifest records its provenance separately.

The original draft's three-model, two-seed pilot tables are preserved in
[legacy_pilot_tables.tex](legacy_pilot_tables.tex) and the paper appendix. Their
underlying logs have not been recovered. The older draft described their errors
as mean absolute deviations, a convention that cannot be verified and differs
from the main study's maximum-deviation metric; they are not pooled or compared
numerically with the main study. No new provider benchmark was run for this revision.

## Build the manuscript

With a TeX installation providing `latexmk`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

`paper.tex` uses the single BibTeX source `references.bib`, includes the legacy
pilot tables, and references the regenerated vector figures.
