# Weak-Form PDE Identification Benchmark

**Can an AI learn the structure of a PDE by probing it with test functions?**

This benchmark tests whether language models can perform genuine scientific reasoning — specifically, whether they can identify an unknown partial differential equation by examining its weak-form responses, the way a mathematician would.

## The idea

A physicist encountering an unknown PDE doesn't just stare at solution data. They *probe* it: multiply by a test function, integrate by parts, and read off structural information from the resulting identities. Each test function is a question asked to the data, and the weak-form integral is the answer.

This benchmark gives an AI the same tools. The model receives:

- Sample values of a solution u(x) to an unknown elliptic PDE
- Weak-form integral responses for a set of test functions
- The ability to submit *new* test functions as strings (e.g. `"bump(0.2, 0.8)"`) and receive back the corresponding weak-form integrals

The model must determine the PDE: what are the coefficient functions a(x), b(x), c(x), f(x) in

$$-(a(x)\, u')' + b(x)\, u' + c(x)\, u = f(x) \quad \text{on } [0, 1]$$

This tests **learning** (can the model extract structure from examples?), **metacognition** (does it know what it doesn't know yet?), and **executive function** (can it plan which test functions to try next?).

## Project structure

```text
weak_form_benchmark/
├── basis.py              # Pluggable basis function families
├── pde.py                # Elliptic PDE definition + 4th-order FD solver
├── test_functions.py     # String-parseable test functions (fixed library)
├── expression_parser.py  # Safe math expression evaluator (arbitrary test functions)
├── weak_form.py          # Weak-form integrator + linear system assembly
├── interactive.py        # Live LLM session environment
├── diagnostics.py        # Visualization and data export
├── benchmark.py          # Task generator, scorer, benchmark suite
├── llm_loop.py           # LLM driver loop (Anthropic API + mock backend)
├── oracle_decompose.py   # Oracle baseline using DECOMPOSE strategy
├── demo.py               # Basic demo
├── demo_full.py          # Full diagnostic visualization demo
├── demo_benchmark.py     # Benchmark suite demo with oracle solver
├── test_basis.py         # Unit tests for basis framework
├── test_weak_form.py     # Unit tests for weak-form integrator
├── test_interactive.py   # Tests + demo for interactive session
└── test_parser.py        # Unit tests for LLM response parser
```

## Modules

### `basis.py` — Coefficient representation

PDE coefficients are represented as linear combinations of basis functions:
`f(x) = Σ cⱼ ψⱼ(x)`

Three basis families are implemented, and new ones require only a subclass with three methods (`evaluate`, `evaluate_derivative`, `label`):

| Family | Functions | Best for |
| --- | --- | --- |
| `LegendreBasis` | Shifted Legendre polynomials on [a,b] | Smooth coefficients, orthogonality |
| `FourierBasis` | {1, cos(kθ), sin(kθ)} | Periodic problems |
| `MonomialBasis` | {1, x, x², ...} | Debugging, simple cases |

`CoefficientFunction` wraps a basis + coefficient vector into a callable with derivatives, LaTeX output, and serialization. Factory helpers generate random coefficients with controllable sparsity.

### `pde.py` — PDE definition and solver

`EllipticPDE` holds the four coefficient functions (a, b, c, f), boundary conditions, and metadata. The solver uses second-order centered finite differences with half-grid diffusion coefficients, solved via a banded linear system. Solution derivatives use 4th-order FD stencils for high accuracy in weak-form integration. Typical residuals are O(10⁻¹⁰).

`EllipticSolution` stores the solution and provides derived quantities: 4th-order accurate derivatives, flux a(x)u'(x), energy norm, pointwise residual, and a diagnostics table.

`make_random_elliptic_pde` generates well-posed problems with guaranteed ellipticity (a(x) > 0).

### `test_functions.py` — The LLM interface

An AI specifies a test function as a plain string and gets back a callable with precomputed derivatives:

```python
phi = parse_test_function("bump(0.2, 0.8)")
phi = parse_test_function("sin(3*pi*x)")
phi = parse_test_function("0.5 * bump(0.1, 0.5) + bump(0.5, 0.9)")
```

| Syntax | Description |
| --- | --- |
| `bump(a, b)` | sin² bump supported on [a, b] — smooth, localized |
| `sin(k*pi*x)` | k-th Fourier sine mode — global, satisfies BCs |
| `hat(a, peak, b)` | Piecewise-linear hat — simple, C⁰ only |
| `poly_bubble(p)` | ((x−x₀)(x₁−x))^p normalized — smooth, global |

Primitives compose via scaling (`0.5 * bump(...)`) and addition (`expr1 + expr2`).

### `expression_parser.py` — Arbitrary test function evaluator

Extends the test function interface to accept any valid math expression, not just the fixed library. Uses Python's `ast` module to parse and evaluate expressions safely — no `eval` or `exec`.

Allowed constructs: variable `x`, constants (`pi`, `e`), standard math functions (`sin`, `cos`, `exp`, `log`, `sqrt`, `tanh`, `heaviside`, etc.), and all arithmetic operators including `^` as an alias for `**`. Anything else raises an `ExpressionError` at parse time.

```python
from expression_parser import make_test_function_from_string

tf, err = make_test_function_from_string("x*(1-x)*sin(3*pi*x)")
# tf is a TestFunction with numerical derivatives (4th-order FD)
# err is None on success, or an error message string
```

Validation checks: expression parses, evaluates to finite values, satisfies boundary conditions φ(0) = φ(1) = 0, and is not identically zero.

### `interactive.py` — Live LLM session

`InteractiveSession` is the environment for the active identification task. Instead of receiving a fixed set of precomputed responses, the LLM chooses its own test functions adaptively within a query budget.

```python
session = InteractiveSession.from_difficulty("medium", seed=42)
print(session.system_prompt())   # PDE description + u(x) data + rules

resp = session.query("sin(pi*x)")
# → {"status": "ok", "integral_f_phi": 0.1234, "integral_u_phi": 0.0567,
#    "queries_remaining": 49, ...}

resp = session.query("cos(pi*x)")   # violates BCs
# → {"status": "error", "message": "Boundary condition violated: ...", ...}
# Error responses do NOT consume budget.

score = session.submit_prediction({"a": [...], "b": [...], "c": [...], "f": [...]})
# → {"status": "scored", "summary": "...", "ground_truth": {...}, ...}
```

The session tracks full query history (spec, response, timestamp) and can export a transcript. This is where the LLM's test function choice strategy gets tested: Fourier modes for global structure, localized bumps or Gaussians for spatial resolution, exponential probes for edge behavior.

### `llm_loop.py` — LLM driver loop

`run_session()` manages the turn-by-turn conversation between an LLM and an `InteractiveSession`. The LLM receives the system prompt once (PDE data + protocol), then iterates: generate response → parse actions → execute against the session → feed results back as user messages.

The protocol supports four action types the LLM can emit:

| Command | Effect | Costs budget? |
| --- | --- | --- |
| `QUERY: <expr>` | Returns ∫fφ and ∫uφ | Yes |
| `DECOMPOSE: <expr>` | Returns full linear system row (G_diff, G_adv, G_react, G_src per basis function) | Yes |
| `COMPUTE: basis_info` | Returns basis descriptions and evaluation at sample points | No |
| `PREDICT:` followed by `a_coeffs = [...]` etc. | Scores and ends the session | — |

`parse_llm_response()` scans the LLM's text line-by-line for these prefixes (case-insensitive). Reasoning text is kept in the conversation history but produces no actions. If a turn has no actions, the LLM is nudged to produce one.

Two backends: `AnthropicBackend` (live API, configurable model) and `MockBackend` (scripted responses, no API key needed for testing).

```bash
python llm_loop.py --difficulty medium --seed 42 --max-queries 30
python llm_loop.py --mock                        # scripted mock, no API key needed
python llm_loop.py --difficulty hard --seed 123 --save-transcript out.txt
```

### `oracle_decompose.py` — Oracle baseline

Demonstrates the theoretically optimal strategy: use `DECOMPOSE` to collect one linear system row per test function, then solve the resulting overdetermined system via least squares. This is the best-case ceiling against which real LLM runs are compared.

Strategy:

1. `COMPUTE: basis_info` (free) to confirm basis size
2. Select ~1.5× the number of unknowns worth of test functions (sine modes + polynomial bubbles + localized Gaussians)
3. `DECOMPOSE` each one to get the full matrix rows
4. `QUERY` the same functions to get the RHS (∫fφ)
5. Solve: first recover f from G_src, then recover a, b, c from the LHS system

```bash
python oracle_decompose.py --difficulty hard --seed 42
```

### `weak_form.py` — Weak-form integrator

The core computation. After integration by parts, each test function φ gives one scalar equation:

∫ a u' φ' dx + ∫ b u' φ dx + ∫ c u φ dx = ∫ f φ dx

The module provides:

- `WeakFormResult` — per-test-function breakdown of all integral terms (diffusion, advection, reaction, rhs, residual)
- `compute_weak_form_battery` — runs the full library against a solution
- `weak_form_from_string` — the LLM-facing API: string in → scalar out
- `WeakFormLinearSystem` — assembles the full linear system connecting unknown basis coefficients to weak-form observations, with SVD diagnostics for identifiability analysis

Integration uses composite Simpson's rule on the solution grid.

### `benchmark.py` — Task generator and scorer

`BenchmarkTask` packages everything the LLM receives: solution data, weak-form responses, basis information, and a natural-language prompt. Tasks are serializable to JSON.

Three difficulty levels control:

- Number of basis functions (2 / 3 / 4)
- Coefficient sparsity and scale
- Solution data density (how many u(x) points the LLM sees)

`score_prediction` evaluates predictions on multiple axes: coefficient-space error, pointwise error, and sparsity pattern matching (did the model correctly identify which terms are zero?).

`generate_suite` creates a balanced benchmark with configurable numbers of easy/medium/hard tasks.

### `diagnostics.py` — Visualization

Dashboard plots, weak-form decomposition bar charts, convergence studies, coefficient recovery comparisons, and singular value spectra. Also exports tabular data for LLM consumption.

## Quick start

**Static benchmark** (fixed test function library):

```python
from benchmark import generate_task, score_prediction
from basis import LegendreBasis

task = generate_task("test_001", difficulty="medium", seed=42)
print(task.to_llm_prompt())   # full prompt with precomputed responses

prediction = {"a": [1.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0],
              "c": [0.0, 0.0, 0.0], "f": [1.0, 0.0, 0.0]}
basis = LegendreBasis(n_basis=3, domain=(0.0, 1.0))
score = score_prediction(task, prediction, basis)
print(score.summary())
```

**Interactive mode** (LLM chooses its own test functions):

```python
from interactive import InteractiveSession

session = InteractiveSession.from_difficulty("medium", seed=42)
print(session.system_prompt())

# Submit arbitrary math expressions as test functions
resp = session.query("sin(2*pi*x)")
resp = session.query("x*(1-x)*exp(-50*(x-0.3)**2)")

# When ready, submit the prediction
score = session.submit_prediction({"a": [...], "b": [...], "c": [...], "f": [...]})
print(score["summary"])
```

## Key findings so far

- **4th-order FD derivatives are essential.** Upgrading from 2nd to 4th-order FD stencils for u'(x) improved weak-form residuals from O(10⁻⁵) to O(10⁻⁸) and coefficient recovery from ~10% error to ~0.01% error.

- **The system conditioning matters.** With n_basis=4 and 15 test functions, the system has 16 unknowns but only 15 equations — it's underdetermined and the recovery fails catastrophically. The LLM needs to reason about how many test functions are needed.

- **Different coefficients have different identifiability.** The singular value spectrum drops 10 orders of magnitude, meaning some coefficient components are nearly invisible to the weak form. The diffusion coefficient a(x) is easiest to recover; reaction c(x) is hardest.

- **Some PDE configurations are inherently harder.** Even the optimal oracle solver fails on ~10% of randomly generated tasks due to ill-conditioning.

## What's next

**Interactive mode** — the LLM chooses which test functions to submit (not from a fixed library), making this a sequential decision problem. This is where executive function and metacognition get tested: *which test function should I try next to maximally reduce my uncertainty about the PDE?*

## Context

This benchmark is designed for the [Google DeepMind × Kaggle AGI Benchmarks Hackathon](https://www.kaggle.com/competitions/benchmarks-for-agi), targeting the **Learning** and **Metacognition** tracks. It tests whether AI models can perform structured scientific inquiry — not by memorizing known PDEs, but by reasoning about functional relationships through the weak formulation.
