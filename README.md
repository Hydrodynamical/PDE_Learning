# Can Language Models Learn a PDE?

A benchmark testing whether LLMs can identify an unknown elliptic PDE by probing
it with test functions through the weak formulation. The model receives solution
data, a budget of queries, and diagnostic tools — but must design its own
experimental strategy.

---

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib anthropic openai

# Set your API key (Anthropic or OpenAI)
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here      # if using OpenAI models

# Run with Anthropic (default)
python main_loop.py --difficulty hard --seed 42

# Run with OpenAI
python main_loop.py --provider openai --model gpt-4o --difficulty hard --seed 42

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
| `COMPUTE: term_integral <type> <expr>` | ∫u'φ' dx, ∫u'φ dx, or ∫uφ dx | free |
| `COMPUTE: solve` | Least-squares coefficient recovery + delta from previous solve | free |
| `COMPUTE: check <expression>` | Predicted LHS vs actual ∫fφ dx for a new test function | 1 query |
| `PREDICT:` | End session and submit last `COMPUTE: solve` coefficients as answer | — |

`COMPUTE: solve` shows per-coefficient change from the previous solve, so the
model can tell when its estimates have converged.

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
efficiency error curves, final scores, ground truth, and behavioral metrics.

Use `--output-dir runs/` to collect logs from multiple runs in one place.

---

## CLI Reference

```
python main_loop.py [options]

Core options:
  --difficulty {easy,medium,hard,extreme}   Problem difficulty (default: medium)
  --seed INT                                Random seed (default: 42)
  --provider {anthropic,openai}             LLM provider (default: anthropic)
  --model MODEL                             Model name (default: claude-haiku-4-5-20251001
                                            for anthropic, gpt-4o for openai)

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
| `main_loop.py` | **Primary entry point.** Probe-only benchmark with Anthropic/OpenAI support, auto-scaled budget, convergence delta tracking, and full JSON logging. |
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
- `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` environment variables
