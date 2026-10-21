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
[legacy_pilot_tables.tex](legacy_pilot_tables.tex) for provenance. Their
underlying logs have not been recovered. The older draft described their errors
as mean absolute deviations, a convention that cannot be verified and differs
from the main study's maximum-deviation metric; they are not pooled or compared
numerically with the main study. The research revision adds numerical controls without new provider calls.

## Build the manuscript

With a TeX installation providing `latexmk`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

`paper.tex` uses the single BibTeX source `references.bib`, the official ICML
preprint style, generated research tables, and the regenerated vector figures.
Unverified legacy pilot tables remain outside the manuscript evidence.

## Research controls and uncertainty analysis

The October research revision adds three offline experiments. Run these after
setting up the pinned Python 3.12 environment and downloading the pinned ZIP:

```bash
python scripts/analyze_archive_statistics.py --archive /tmp/pde-runs-v1.zip
python scripts/control_baselines.py --archive /tmp/pde-runs-v1.zip
python scripts/audit_identifiability.py --archive /tmp/pde-runs-v1.zip
python scripts/plot_research_audit.py
python scripts/verify_paper_artifacts.py
```

These commands make no model or network calls. A full control experiment takes
roughly one to two minutes on the development machine; timings depend on hardware.

- `statistical_analysis.json` retains Wilson success intervals, all 45 paired
  comparisons, discordance counts, exact paired tests, and Holm-adjusted values
  within the 15 pairs of each difficulty. Percentile bootstrap intervals use
  20,000 draws, seed `20261020`, and shared tasks within difficulty. Confidence
  summaries average within a run before resampling whole runs; reports and
  coefficient blocks do not become independent tasks. The estimates condition
  on scored archive inclusion and, for confidence, availability of a paired report.
- `control_baselines.json.gz` contains the full new experiment, compressed with
  a fixed gzip timestamp; `control_baselines_summary.json` provides a compact
  human-readable view. Fixed sine, boundary-weighted Legendre, and ten random
  mixed-family query lists are evaluated on the same 30 regenerated truths.
  There are 1,080 policy/grid runs across original, 2×, and 4× grid resolutions,
  plus 360 native-versus-banded comparisons. The random design lists are reused
  across the ten tasks in each setting: 100 outcomes represent ten designs
  crossed with ten PDEs. They do not increase the number of independent tasks.
- `identifiability_audit.json` records exact polynomial-rank certificates,
  reconstructed joint spectra, 687 solve replays, all 2,874 query-response
  comparisons, free-check counts, and common-amplitude quantization tests.
  Its CLI reads the validated ZIP directly. An optional `--runs-dir` verifies an
  extracted copy byte for byte against the pinned archive before using it.
- Three additional vector figures are `paper_controls.pdf`,
  `paper_numerical_audit.pdf`, and `paper_confidence_uncertainty.pdf`.
  Generated TeX tables and `research_figure_sources.json` bind their numbers to
  the exact derived inputs and plotting script.

Fixed sine succeeds at `E < 0.01` on all 30 reconstructed tasks already at the
minimum permitted `4N` stored queries. This result limits any claim that adaptive
model measurement selection is necessary on these instances. It is a new
numerical control, not an exact replay of the historical environment. Although
all archived query RHS responses match exactly, reconstructed solve coefficients
can differ materially from their logged values. The paper reports this gap.

Original-resolution controls use the native dense forward solver and unchanged
query/solve tools. Refinements use an independently assembled equivalent banded
solver, checked against the native solver on every original-resolution task.
Both threshold classifications agree at every checkpoint. Primary control scores
use the internal unrounded final estimate, matching the current submission
flow. Display-rounded scores are also saved, with identical classifications at
both thresholds. The amplitude audit separately scores 8-decimal diagnostic
estimates, using all stored rows; it is not an archived final-submission score.

Source hashes and executed runtime versions are included in each research
output. `verify_paper_artifacts.py` uses the standard library to reject stale
outputs after an implementation, input, or plotting change. Floating-point
values can still vary across numerical platforms; integrity checks do not claim
bitwise portability of newly rerun solves.

The public paper uses the unmodified official ICML 2026 template with its
`preprint` option; see [template_provenance.md](template_provenance.md). The main
text follows the eight-page constraint, with references and detailed appendices
following it. It is a public preprint and retains author information and public
repository links. An actual conference submission requires the target year's
instructions and anonymization.
