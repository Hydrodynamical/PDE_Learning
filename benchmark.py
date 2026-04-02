"""
Benchmark task generator and scorer.

A BenchmarkTask packages everything the LLM receives:
    - Solution data u(x) on a grid
    - A way to query weak-form integrals (via test function strings)
    - Metadata about the basis and domain

The Scorer compares LLM predictions against ground truth:
    - Coefficient-space error (are the basis coefficients right?)
    - Pointwise error (does the recovered function match?)
    - Operator-level error (does the recovered PDE produce the same solution?)

A BenchmarkSuite generates multiple tasks at varying difficulty.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import numpy as np

from basis import BasisFamily, LegendreBasis, CoefficientFunction
from pde import EllipticPDE, EllipticSolution, solve_elliptic, make_random_elliptic_pde
from test_functions import standard_library, parse_test_function
from weak_form import (
    compute_weak_form,
    compute_weak_form_battery,
    weak_form_from_string,
    assemble_linear_system,
    battery_to_table,
)


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkTask:
    """
    A single benchmark problem.

    The LLM receives:
        - prompt: natural-language description of the task
        - solution_data: u(x) values on a grid (what the LLM can "see")
        - basis_info: what basis to express the answer in
        - weak_form_api: callable that takes a test function string → response

    The scorer holds:
        - ground_truth: the true PDE coefficients
    """
    task_id: str
    difficulty: str  # "easy", "medium", "hard"

    # What the LLM sees
    prompt: str
    solution_data: dict  # {"x": [...], "u": [...]}
    basis_info: dict     # {"type": "LegendreBasis", "n_basis": 3, "domain": [0, 1]}
    precomputed_responses: dict  # weak-form results for standard library

    # Hidden ground truth
    ground_truth: dict   # {"a": [...], "b": [...], "c": [...], "f": [...]}
    pde_description: str

    # Internal objects (not serialized)
    _solution: Optional[EllipticSolution] = field(default=None, repr=False)

    def query(self, spec: str) -> dict:
        """
        LLM submits a test function string, gets back the weak-form response.
        This is the interactive API for the future version.
        """
        if self._solution is None:
            raise RuntimeError("Solution not available — use precomputed_responses")
        return weak_form_from_string(self._solution, spec)

    def to_llm_prompt(self) -> str:
        """
        Generate the full prompt that would be sent to an LLM.
        """
        x_str = ", ".join(f"{v:.6f}" for v in self.solution_data["x"])
        u_str = ", ".join(f"{v:.6f}" for v in self.solution_data["u"])

        responses_str = ""
        for spec, resp in self.precomputed_responses.items():
            responses_str += (
                f"  test_function: \"{spec}\"\n"
                f"    integral_f_phi = {resp['integral_f_phi']:.10f}\n"
                f"    integral_u_phi = {resp['integral_u_phi']:.10f}\n"
            )

        return f"""{self.prompt}

## Solution data

x = [{x_str}]
u = [{u_str}]

## Basis information

The coefficient functions a(x), b(x), c(x), f(x) are each expressed as:
  g(x) = sum_j c_j * psi_j(x)

where psi_j are {self.basis_info['type']} basis functions of degree 0 to {self.basis_info['n_basis'] - 1}
on the domain {self.basis_info['domain']}.

## Weak-form responses

For each test function phi(x), you are given:
  integral_f_phi = integral of f(x) * phi(x) dx
  integral_u_phi = integral of u(x) * phi(x) dx

{responses_str}
## Your task

Determine the coefficient vectors for a(x), b(x), c(x), and f(x).
Express your answer as four lists of {self.basis_info['n_basis']} numbers each:
  a_coeffs = [...]
  b_coeffs = [...]
  c_coeffs = [...]
  f_coeffs = [...]
"""

    def to_json(self) -> str:
        """Serialize the task (without internal objects) for storage."""
        return json.dumps({
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "solution_data": {
                "x": [float(v) for v in self.solution_data["x"]],
                "u": [float(v) for v in self.solution_data["u"]],
            },
            "basis_info": self.basis_info,
            "precomputed_responses": self.precomputed_responses,
            "ground_truth": self.ground_truth,
            "pde_description": self.pde_description,
        }, indent=2)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    """Scoring result for one task."""
    task_id: str

    # Per-coefficient errors (in basis coefficient space)
    coeff_error_a: float
    coeff_error_b: float
    coeff_error_c: float
    coeff_error_f: float

    # Pointwise errors (max |true(x) - predicted(x)| over a fine grid)
    pointwise_error_a: float
    pointwise_error_b: float
    pointwise_error_c: float
    pointwise_error_f: float

    # Aggregate
    total_coeff_error: float
    total_pointwise_error: float

    # Did the model get the sparsity pattern right?
    sparsity_match_a: bool
    sparsity_match_b: bool
    sparsity_match_c: bool

    def summary(self) -> str:
        lines = [
            f"Task {self.task_id}:",
            f"  Coefficient errors:  a={self.coeff_error_a:.4f}, b={self.coeff_error_b:.4f}, "
            f"c={self.coeff_error_c:.4f}, f={self.coeff_error_f:.4f}",
            f"  Pointwise errors:    a={self.pointwise_error_a:.4f}, b={self.pointwise_error_b:.4f}, "
            f"c={self.pointwise_error_c:.4f}, f={self.pointwise_error_f:.4f}",
            f"  Total coeff error:   {self.total_coeff_error:.4f}",
            f"  Total pointwise err: {self.total_pointwise_error:.4f}",
            f"  Sparsity match:      a={self.sparsity_match_a}, b={self.sparsity_match_b}, "
            f"c={self.sparsity_match_c}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "coeff_error_a": self.coeff_error_a,
            "coeff_error_b": self.coeff_error_b,
            "coeff_error_c": self.coeff_error_c,
            "coeff_error_f": self.coeff_error_f,
            "pointwise_error_a": self.pointwise_error_a,
            "pointwise_error_b": self.pointwise_error_b,
            "pointwise_error_c": self.pointwise_error_c,
            "pointwise_error_f": self.pointwise_error_f,
            "total_coeff_error": self.total_coeff_error,
            "total_pointwise_error": self.total_pointwise_error,
            "sparsity_match_a": self.sparsity_match_a,
            "sparsity_match_b": self.sparsity_match_b,
            "sparsity_match_c": self.sparsity_match_c,
        }


def score_prediction(
    task: BenchmarkTask,
    predicted: dict,
    basis: BasisFamily,
    sparsity_tol: float = 0.05,
    n_eval: int = 500,
) -> ScoreResult:
    """
    Score a prediction against ground truth.

    Parameters
    ----------
    task : BenchmarkTask
    predicted : dict with keys "a", "b", "c", "f", each a list of coefficients
    basis : BasisFamily (must match task.basis_info)
    sparsity_tol : float
        Coefficients below this threshold are considered "zero" for sparsity matching.
    n_eval : int
        Number of points for pointwise evaluation.
    """
    gt = task.ground_truth
    domain = tuple(task.basis_info["domain"])
    x_eval = np.linspace(domain[0], domain[1], n_eval)

    errors_coeff = {}
    errors_pw = {}
    sparsity_match = {}

    for name in ["a", "b", "c", "f"]:
        true_coeffs = np.array(gt[name])
        pred_coeffs = np.array(predicted[name])

        # Coefficient-space error (L-inf on coefficients)
        errors_coeff[name] = float(np.max(np.abs(true_coeffs - pred_coeffs)))

        # Pointwise error
        true_fn = CoefficientFunction(basis, true_coeffs)
        pred_fn = CoefficientFunction(basis, pred_coeffs)
        errors_pw[name] = float(np.max(np.abs(true_fn(x_eval) - pred_fn(x_eval))))

        # Sparsity pattern
        if name != "f":
            true_sparse = np.abs(true_coeffs) < sparsity_tol
            pred_sparse = np.abs(pred_coeffs) < sparsity_tol
            sparsity_match[name] = bool(np.array_equal(true_sparse, pred_sparse))

    return ScoreResult(
        task_id=task.task_id,
        coeff_error_a=errors_coeff["a"],
        coeff_error_b=errors_coeff["b"],
        coeff_error_c=errors_coeff["c"],
        coeff_error_f=errors_coeff["f"],
        pointwise_error_a=errors_pw["a"],
        pointwise_error_b=errors_pw["b"],
        pointwise_error_c=errors_pw["c"],
        pointwise_error_f=errors_pw["f"],
        total_coeff_error=sum(errors_coeff.values()),
        total_pointwise_error=sum(errors_pw.values()),
        sparsity_match_a=sparsity_match.get("a", True),
        sparsity_match_b=sparsity_match.get("b", True),
        sparsity_match_c=sparsity_match.get("c", True),
    )


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------

TASK_PROMPT = """You are given a solution u(x) to an unknown elliptic PDE of the form:

    -(a(x) u'(x))' + b(x) u'(x) + c(x) u(x) = f(x)    on [0, 1]

with boundary conditions u(0) = 0, u(1) = 0.

The coefficient functions a(x), b(x), c(x), and f(x) are each represented as a
linear combination of basis functions. Your goal is to determine the coefficients
by analyzing the weak-form responses provided below.

Each weak-form response corresponds to a test function phi(x) and gives you:
  integral_f_phi = integral of f(x) * phi(x) dx over [0, 1]
  integral_u_phi = integral of u(x) * phi(x) dx over [0, 1]

The key identity (after integration by parts) is:
  integral of a*u'*phi' dx + integral of b*u'*phi dx + integral of c*u*phi dx = integral of f*phi dx
"""

DIFFICULTY_CONFIGS = {
    "easy": {
        "n_basis": 2,
        "sparsity": 0.3,
        "scale": 0.5,
        "a_min": 1.0,
        "n_grid": 2001,
        "n_solution_points": 51,  # how many u(x) points the LLM sees
    },
    "medium": {
        "n_basis": 3,
        "sparsity": 0.2,
        "scale": 1.0,
        "a_min": 0.5,
        "n_grid": 2001,
        "n_solution_points": 31,
    },
    "hard": {
        "n_basis": 4,
        "sparsity": 0.1,
        "scale": 1.5,
        "a_min": 0.3,
        "n_grid": 4001,
        "n_solution_points": 21,
    },
    "extreme": {
        "n_basis": 8,
        "sparsity": 0.4,
        "scale": 1.0,
        "a_min": 0.3,
        "n_grid": 4001,
        "n_solution_points": 21,
    },
}


def generate_task(
    task_id: str,
    difficulty: str = "medium",
    seed: Optional[int] = None,
    extra_test_fns: Optional[list[str]] = None,
) -> BenchmarkTask:
    """
    Generate a single benchmark task.

    Parameters
    ----------
    task_id : str
    difficulty : "easy", "medium", "hard"
    seed : random seed for reproducibility
    extra_test_fns : additional test function specs beyond the standard library
    """
    config = DIFFICULTY_CONFIGS[difficulty]
    rng = np.random.default_rng(seed)

    basis = LegendreBasis(n_basis=config["n_basis"], domain=(0.0, 1.0))
    pde = make_random_elliptic_pde(
        basis, rng=rng,
        a_min=config["a_min"],
        sparsity=config["sparsity"],
        scale=config["scale"],
    )
    sol = solve_elliptic(pde, n_grid=config["n_grid"])

    # Subsample solution for the LLM
    n_pts = config["n_solution_points"]
    stride = max(1, (config["n_grid"] - 1) // (n_pts - 1))
    x_sub = sol.x[::stride]
    u_sub = sol.u[::stride]

    # Compute weak-form responses for standard library
    test_fns = standard_library()
    if extra_test_fns:
        for spec in extra_test_fns:
            test_fns.append(parse_test_function(spec))

    responses = {}
    for tf in test_fns:
        result = compute_weak_form(sol, tf)
        responses[tf.spec] = result.blind_response()

    # Ground truth
    ground_truth = {
        "a": pde.a.coeffs.tolist(),
        "b": pde.b.coeffs.tolist(),
        "c": pde.c.coeffs.tolist(),
        "f": pde.f.coeffs.tolist(),
    }

    return BenchmarkTask(
        task_id=task_id,
        difficulty=difficulty,
        prompt=TASK_PROMPT,
        solution_data={"x": x_sub.tolist(), "u": u_sub.tolist()},
        basis_info={
            "type": "LegendreBasis",
            "n_basis": config["n_basis"],
            "domain": [0.0, 1.0],
        },
        precomputed_responses=responses,
        ground_truth=ground_truth,
        pde_description=pde.describe(),
        _solution=sol,
    )


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

def generate_suite(
    n_easy: int = 3,
    n_medium: int = 4,
    n_hard: int = 3,
    base_seed: int = 2026,
) -> list[BenchmarkTask]:
    """
    Generate a full benchmark suite with tasks at varying difficulty.
    """
    tasks = []
    seed = base_seed

    for i in range(n_easy):
        tasks.append(generate_task(f"easy_{i+1:02d}", "easy", seed=seed))
        seed += 1

    for i in range(n_medium):
        tasks.append(generate_task(f"medium_{i+1:02d}", "medium", seed=seed))
        seed += 1

    for i in range(n_hard):
        tasks.append(generate_task(f"hard_{i+1:02d}", "hard", seed=seed))
        seed += 1

    return tasks


def save_suite(tasks: list[BenchmarkTask], path: str):
    """Save a benchmark suite to a JSON file."""
    data = []
    for t in tasks:
        data.append(json.loads(t.to_json()))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_suite(path: str) -> list[BenchmarkTask]:
    """Load a benchmark suite from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    tasks = []
    for d in data:
        tasks.append(BenchmarkTask(
            task_id=d["task_id"],
            difficulty=d["difficulty"],
            prompt=d["prompt"],
            solution_data=d["solution_data"],
            basis_info=d["basis_info"],
            precomputed_responses=d["precomputed_responses"],
            ground_truth=d["ground_truth"],
            pde_description=d["pde_description"],
        ))
    return tasks


def score_suite(
    tasks: list[BenchmarkTask],
    predictions: dict[str, dict],
) -> list[ScoreResult]:
    """
    Score predictions for a full suite.

    predictions: dict mapping task_id → {"a": [...], "b": [...], "c": [...], "f": [...]}
    """
    results = []
    for task in tasks:
        if task.task_id not in predictions:
            continue
        n_basis = task.basis_info["n_basis"]
        basis = LegendreBasis(n_basis=n_basis, domain=tuple(task.basis_info["domain"]))
        result = score_prediction(task, predictions[task.task_id], basis)
        results.append(result)
    return results


def suite_summary(scores: list[ScoreResult]) -> str:
    """Print a summary of suite scores."""
    lines = [
        f"{'Task':<15} {'Coeff Err':>10} {'PW Err':>10} {'Sparsity':>10}",
        "-" * 47,
    ]
    for s in scores:
        sp = "yes" if (s.sparsity_match_a and s.sparsity_match_b and s.sparsity_match_c) else "no"
        lines.append(
            f"{s.task_id:<15} {s.total_coeff_error:>10.4f} "
            f"{s.total_pointwise_error:>10.4f} {sp:>10}"
        )

    total_ce = np.mean([s.total_coeff_error for s in scores])
    total_pw = np.mean([s.total_pointwise_error for s in scores])
    lines.append("-" * 47)
    lines.append(f"{'MEAN':<15} {total_ce:>10.4f} {total_pw:>10.4f}")
    return "\n".join(lines)
