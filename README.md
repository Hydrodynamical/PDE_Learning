# Can Language Models Learn a PDE?

A benchmark testing whether LLMs can identify an unknown elliptic PDE by probing it with test functions through the weak formulation. The model receives solution data, a budget of queries, and diagnostic tools. We measure scientific reasoning ability by comparing a baseline prompt to an enhanced prompt that explains the ill-conditioning structure of the weak-form linear system. Same code, same tools, same PDE, same seed — the only controlled variable is the initial message.

## Quick Start

```bash
# Baseline run
python probe_loop.py --difficulty hard --seed 42 --max-queries 50

# Enhanced prompt (adds mathematical context about ill-conditioning)
python probe_loop.py --difficulty hard --seed 42 --max-queries 50 --enhanced

# Easy difficulty (good for testing)
python probe_loop.py --difficulty easy --seed 42 --max-queries 30

# Mock LLM (no API key needed)
python probe_loop.py --mock --difficulty medium --seed 42
```

## The PDE

The model must identify all coefficients of:

```text
-(a(x) u'(x))' + b(x) u'(x) + c(x) u(x) = f(x)    on [0, 1]
u(0) = 0,  u(1) = 0
```

Each coefficient function a(x), b(x), c(x), f(x) is a linear combination of shifted Legendre polynomials. The model receives sampled values of the solution u(x) and must recover all coefficient vectors from weak-form queries.

## Tools Available to the Model

| Command | Cost | Description |
| --- | --- | --- |
| `QUERY: <expression>` | 1 query | Returns ∫f·φ dx — the weak-form RHS |
| `COMPUTE: eval_solution x1 x2 ...` | free | Returns u(x) and u'(x) at given points |
| `COMPUTE: term_integral <diffusion\|advection\|reaction> <expr>` | free | Returns ∫u'φ' dx, ∫u'φ dx, or ∫uφ dx |
| `COMPUTE: solve` | free | Least-squares coefficient recovery from all queries (needs ≥4N) |
| `COMPUTE: verify <expression>` | 1 query | Compares predicted LHS to actual ∫f·φ dx |

`term_integral` requires only the known solution u(x) and the chosen test function φ — no knowledge of a, b, c, f. It lets the model check whether a candidate test function will usefully constrain each PDE term before spending a query on it.

## The Benchmark

The controlled variable is the initial prompt only:

- **Baseline**: `"Begin."`
- **Enhanced**: One paragraph explaining the weak-form structure, the concept of term dominance (if the diffusion weight is 100× larger than the reaction weight, that equation tells you almost nothing about c(x)), and the recommendation to use `COMPUTE: term_integral` to screen test functions before querying.

Everything else is identical: the same PDE, the same tools, the same budget, the same seed.

## Results

| Condition | Total coeff error | Key behavior |
| --- | --- | --- |
| Baseline | 1.78 | Queries uniformly, ignores term weights |
| Enhanced | 1.03 | Checks weight balance before querying |

42% reduction in coefficient error. The enhanced model uses `term_integral` to identify test functions that stress-test underrepresented terms — especially the reaction term c(x) — producing a better-conditioned linear system.

## Difficulty Levels

| Level | N (basis size) | Unknowns | Min queries for solve |
| --- | --- | --- | --- |
| easy | 2 | 8 | 8 |
| medium | 3 | 12 | 12 |
| hard | 4 | 16 | 16 |
| extreme | 5 | 20 | 20 |

## File Structure

| File | Description |
| --- | --- |
| `probe_loop.py` | Main benchmark entrypoint |
| `llm_loop.py` | Legacy entrypoint with DECOMPOSE and auto-progress scaffolding |
| `pde.py` | PDE generation and finite-difference solver |
| `basis.py` | Shifted Legendre polynomial basis |
| `weak_form.py` | Weak-form integral computation and linear system assembly |
| `expression_parser.py` | Symbolic test function parsing |
| `benchmark.py` | Scoring (coefficient error, pointwise error, efficiency AUC) |
| `diagnostics.py` | Result plotting |
| `paper.tex` | Writeup |

## CLI Reference

```text
python probe_loop.py [options]

  --difficulty {easy,medium,hard,extreme}   PDE complexity (default: medium)
  --seed INT                                Random seed for PDE generation (default: 42)
  --max-queries INT                         Query budget (default: 30)
  --enhanced                                Add ill-conditioning context to the initial prompt
  --model MODEL                             Anthropic model ID (default: claude-sonnet-4-20250514)
  --max-turns INT                           Maximum conversation turns (default: 100)
  --save-transcript FILE                    Save full conversation to a text file
  --save-plot FILE                          Save results plot (default: probe_<difficulty>_<seed>.png)
  --mock                                    Use mock LLM for testing (no API key needed)
```

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `matplotlib`
- `anthropic` SDK (`pip install anthropic`)
- `ANTHROPIC_API_KEY` environment variable (not needed for `--mock` runs)
