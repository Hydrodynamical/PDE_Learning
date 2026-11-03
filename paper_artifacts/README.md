# Reproducing the preprint

The current manuscript is **Can Language Models Learn a PDE? A Benchmark for
Scientific Reasoning through Experimental Design** (Joseph K. Miller, 2026).
It analyzes 180 scored runs from six models on 30 shared tasks. The original
[Kaggle write-up](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/can-language-models-learn-a-pde)
provides the project context.

## Build the PDF

The checked-in figures and tables are sufficient to build the manuscript; no
dataset download or model calls are needed. From the repository root, with a
TeX installation providing `latexmk`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

The official ICML style is used in preprint mode. This does not imply submission
to or acceptance by ICML; see [template provenance](template_provenance.md).

## Data and final figures

Use the numerical environment in the main README. The underlying records are
available in [Kaggle dataset version 1](https://www.kaggle.com/datasets/jkmiller/elliptic-differential-equation-llm-reasoning/versions/1),
released on 16 April 2026. Download that version separately; Kaggle may require
sign-in. The ZIP contains 180 run JSON files and a sweep summary. Its expected
SHA-256 digest and study settings are in [data_manifest.json](data_manifest.json).
Do not replace the pinned archive with a ZIP assembled from another checkout.

From the repository root, replacing `/path/to/pde-runs-v1.zip` with your download:

```bash
# Validate the archived records against the saved aggregate/per-run analyses.
python analyze_review_runs.py --archive /path/to/pde-runs-v1.zip

# Regenerate the nine figures used by the current manuscript.
python plot_kaggle_revision.py --archive /path/to/pde-runs-v1.zip
python plot_kaggle_task.py
```

These commands do not call model providers or access the network. ZIP records
are read directly, without extraction. The final figures are
`images/kaggle_{task,success,actions,improvement,diversity,reliability,confidence,controls,numerics}.pdf`
and their PNG previews. `plot_kaggle_task.py` reuses the diagram geometry in
`plot_review_revision.py`; the latter's other figures are historical variants.
The analysis command also writes two historical diagnostic plots that are not
included in the current paper.

The November revision was also validated against the author's original local
run files. Their hashes are in [../data/local_run_manifest.json](../data/local_run_manifest.json).
For those exact records, place all 180 run JSON files and `sweep_results.json`
in `data/recorded_runs/` and use
`--runs-dir data/recorded_runs` instead of `--archive` in the two commands above.
The local records reproduce the saved summaries, but byte identity between a
locally assembled ZIP and the historical Kaggle ZIP is not claimed.

## Saved controls and statistical summaries

```bash
# Verify the provenance of the saved controls and statistical outputs.
python scripts/verify_paper_artifacts.py
```

This check uses only the standard library. The following scripts preserve the
analysis and control implementation used to generate the saved results:

- `scripts/reproduce_paper_figures.py`: validated archive summary and earlier
  figure layouts; use `--summary-only` to regenerate just `summary.json`.
- `scripts/analyze_archive_statistics.py`: run-balanced confidence summaries
  and paired success comparisons.
- `scripts/audit_identifiability.py`: identifiability and numerical sensitivity.
- `scripts/control_baselines.py`: fixed/random query controls on reconstructed
  archived tasks.
- `scripts/validate_sine_generalization.py`: fixed sine queries on 300 fresh
  tasks (100 per setting).
- `scripts/plot_research_audit.py`: control/statistical tables and earlier audit
  figures; the current manuscript uses the `kaggle_*` figure files above.

Use each script's `--help` for its arguments. Numerical controls can take time
to rerun, but require no provider API keys. Recorded source hashes in the saved
JSON files identify the implementation used; they are an integrity check, not
a promise of identical floating-point output on every platform.

## Interpretation and limits

- Success uses the sum of each coefficient block's maximum absolute coordinate
  error, with primary threshold `E < 0.01`.
- Stacked tool-use bars show mean counts. Improvement plots separate the
  159 repeated-solve runs from the 21 single-solve cases.
- Query-family entropy describes expression diversity, not matrix independence
  or information gain.
- Confidence is paired with the latest solve returned on a strictly earlier
  turn. The 707 supported reports yield 2,121 observations for `a`, `b`, and `c`.
  Main reliability curves pool reports; appendix Brier scores and signed gaps
  weight supported runs equally. The source coefficient `f` is excluded because
  it satisfies the correctness threshold at every recorded solve.
- The logs do not preserve a complete failed-attempt/retry history or exact
  historical prompts/source versions. Reference prompts are supplied in
  `reference_prompt_components.txt`; they are not claimed to reconstruct every
  historical message.
- Fixed sine queries solve all 30 reconstructed archived tasks. The study does
  not demonstrate an advantage from adaptive measurement selection.

[Technical revision notes](REVISION_NOTES.txt) retain the explanation of the
legacy matrix-based monitoring/control scores removed from the manuscript.
Those scores are not used in the current confidence analysis.
