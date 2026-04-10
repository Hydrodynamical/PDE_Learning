# Can Language Models Learn a PDE?

A benchmark testing whether LLMs can identify an unknown elliptic PDE by probing
it with test functions through the weak formulation. The model receives solution
data, a budget of queries, and diagnostic tools — but must design its own
experimental strategy.

---

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib anthropic openai google-genai

# Set your API key
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here      # if using OpenAI models
export GOOGLE_API_KEY=your_key_here      # if using Google models

# Run with Anthropic (default)
python main_loop.py --difficulty hard --seed 42

# Run with OpenAI
python main_loop.py --provider openai --model gpt-4o --difficulty hard --seed 42

# Run with Google Gemini
python main_loop.py --provider google --model gemini-3-flash-preview --difficulty hard --seed 42

# Test with mock LLM (no API key needed)
python main_loop.py --mock --difficulty easy

# Ablation: minimal prompt without mathematical context
python main_loop.py --difficulty hard --seed 42 --baseline
```

Budget and turn limits are auto-scaled from difficulty — no manual tuning needed.
All output files are saved to the current directory (override with `--output-dir`).

---

## The PDE

The model must identify the coefficient functions in:

```text
-(a(x) u'(x))' + b(x) u'(x) + c(x) u(x) = f(x)    on [0, 1]
u(0) = 0,  u(1) = 0
```

Each coefficient is a linear combination of shifted Legendre polynomials.
The number of basis functions (and thus unknowns) scales with difficulty.

---

## Tools Available to the Model

| Command | Returns | Cost |
| --- | --- | --- |
| `QUERY: <expression>` | ∫f·φ dx | 1 query |
| `COMPUTE: eval_solution x1 x2 ...` | u(x) and u'(x) at given points | free |
| `COMPUTE: term_integrals <expr>` | All three term weights at once: ∫u'φ' dx (diffusion), ∫u'φ dx (advection), ∫uφ dx (reaction) | free |
| `COMPUTE: solve` | Least-squares coefficient recovery + delta from previous solve | free |
| `COMPUTE: check <expression>` | Predicted LHS vs actual ∫fφ dx for a new test function | free |
| `PREDICT:` | End session and submit last `COMPUTE: solve` coefficients as answer | — |

`COMPUTE: solve` shows per-coefficient change from the previous solve, so the
model can tell when its estimates have converged.

`COMPUTE: term_integrals` returns all three weights in a single call, making it
easy to screen candidate test functions for reaction-term informativeness before
spending a query.

`COMPUTE: check` is a free consistency diagnostic. It compares the predicted
LHS (from current â, b̂, ĉ estimates) against the actual ∫fφ dx for a new test
function. Zero discrepancy does NOT confirm correctness — it only means this test
function is consistent with the current fit. Unlike QUERY, check is read-only: it
does not add the test function to the solver's equation system.

---

## Difficulty Levels

Budget and turn limits are derived automatically. Override with `--max-queries` and `--max-turns`.

| Difficulty | Basis size | Unknowns | Auto budget | Auto turns |
| --- | --- | --- | --- | --- |
| `easy`    | 2 | 8 | 12 | 24 |
| `medium`  | 3 | 12 | 18 | 36 |
| `hard`    | 4 | 16 | 24 | 48 |
| `extreme` | 8 | 32 | 48 | 96 |

---

## Output Files

Each run produces three co-indexed files sharing a common stem:

```
run_<difficulty>_s<seed>_<model>_<timestamp>.json
run_<difficulty>_s<seed>_<model>_<timestamp>_dashboard.png
run_<difficulty>_s<seed>_<model>_<timestamp>_auc.png
```

The JSON log contains: full turn transcript, per-query records, per-solve
coefficient estimates and errors vs ground truth, verifications, term integrals,
efficiency error curves, final scores, ground truth, behavioral metrics, and
metacognitive metrics.

**Behavioral metrics** (in `behavioral_metrics`):

| Field | Description |
| --- | --- |
| `learning_rate` | Slope of log₁₀(error) vs query count across solves (negative = improving) |
| `improvement_ratio` | First-solve error / last-solve error (> 1 means within-session learning) |
| `wasted_turns` / `wasted_turn_fraction` | Turns with no recognized action (malformed commands or pure reasoning) |
| `max_unproductive_streak` | Longest consecutive run of wasted turns |
| `tail_after_last_solve` | Turns between final solve and session end |
| `max_gap_without_progress` | Longest stretch without a QUERY or COMPUTE: solve after the first solve |
| `duplicate_queries` | Count of repeated test functions (working memory failures) |
| `family_entropy` / `family_entropy_normalized` | Shannon entropy of test function family distribution (0 = all one family) |
| `family_counts` | Per-family query counts (polynomial, trigonometric, exponential, localized, other) |
| `query_space_blocks` | Per-block SVD analysis: effective rank, log-volume, condition number for diffusion/advection/reaction blocks |
| `mean_monitoring_accuracy` | Mean Spearman correlation between stated and true uncertainty rankings ($\bar{M}$) |
| `mean_control_efficiency` | Mean fraction of information gain on the most uncertain coefficient ($\bar{C}$) |
| `n_confidence_reports` | Number of times the model reported per-coefficient confidence |

Use `--output-dir runs/` to collect logs from multiple runs in one place.

---

## CLI Reference

```
python main_loop.py [options]

Core options:
  --difficulty {easy,medium,hard,extreme}   Problem difficulty (default: medium)
  --seed INT                                Random seed (default: 42)
  --provider {anthropic,openai,google}      LLM provider (default: anthropic)
  --model MODEL                             Model name (default: claude-haiku-4-5-20251001
                                            for anthropic, gpt-4o for openai,
                                            gemini-3-flash-preview for google)

Budget overrides (auto-scaled by default):
  --max-queries INT                         Query budget
  --max-turns INT                           Max conversation turns
  --min-queries INT                         Min queries before PREDICT is allowed

Output:
  --output-dir DIR                          Directory for JSON and PNG output (default: .)
  --save-transcript FILE                    Also save full transcript as plain text
  --save-plot FILE                          Override dashboard plot path

Prompt:
  --baseline                                Minimal prompt, no mathematical context
                                            (default: standard prompt with ill-conditioning hint)

Testing:
  --mock                                    Use scripted mock LLM (no API key needed)
```

---

## Results (Seed 42, Hard Difficulty)

| Metric | Baseline | Standard |
| --- | --- | --- |
| Total coefficient error | 2.2742 | **0.0085** |
| a(x) error | 0.0659 | 0.0001 |
| b(x) error | 1.2281 | 0.0012 |
| c(x) error | 0.9803 | 0.0072 |
| f(x) error | 0.0000 | 0.0000 |
| Queries used | 28 | 28 |
| Test function diversity | 25 polynomial | 6 poly, 19 trig |

The baseline used all polynomials — x^k(1-x)^m — which are nearly collinear in
the weak-form integral space. The standard prompt model used a mix of sines and
polynomials, giving a well-conditioned overdetermined system.

---

## File Structure

### Main entry points

| File | Description |
|---|---|
| `main_loop.py` | **Primary entry point.** Probe-only benchmark with Anthropic/OpenAI/Google support, auto-scaled budget, convergence delta tracking, and full JSON logging. |
| `kaggle.py` | Kaggle Benchmarks notebook adapter. Uses `dispatch_turn()` from main_loop.py with kbench's `llm.prompt()` interface. |
| `probe_loop.py` | Earlier version of main_loop.py (without convergence deltas). |
| `llm_loop.py` | Legacy variant supporting DECOMPOSE commands and auto-progress feedback. Uses `interactive.py`. |

### Core library (imported by main entry points)

| File | Description |
|---|---|
| `pde.py` | PDE generation and numerical solving |
| `basis.py` | Shifted Legendre polynomial basis |
| `weak_form.py` | Weak-form integral computation |
| `expression_parser.py` | Symbolic test function parsing |
| `benchmark.py` | Scoring and evaluation |
| `diagnostics.py` | Result plotting |
| `interactive.py` | Session manager (used only by `llm_loop.py`) |

### Standalone scripts (not imported by main entry points)

| File | Description |
|---|---|
| `generate_transcripts.py` | Batch runner for multi-seed experiments |
| `analyze_adaptation.py` | Post-hoc analysis of LLM reasoning patterns |
| `oracle_decompose.py` | Reference oracle implementation |
| `demo.py` / `demo_benchmark.py` / `demo_full.py` | Example scripts |
| `test_basis.py` / `test_functions.py` / `test_interactive.py` / `test_parser.py` / `test_weak_form.py` | Unit tests |

---

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `matplotlib`
- `anthropic` (for Anthropic models) — `pip install anthropic`
- `openai` (for OpenAI models) — `pip install openai`
- `google-genai` (for Google models) — `pip install google-genai`
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and/or `GOOGLE_API_KEY` environment variables
