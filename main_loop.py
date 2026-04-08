"""
PDE identification benchmark — main entry point.

The LLM receives solution data, can query weak-form integrals (QUERY),
evaluate the solution (COMPUTE), and submit a coefficient prediction (PREDICT).
Solve results show per-coefficient deltas from the previous solve to help
the model track convergence.

Usage:
    python main_loop.py --mock                       # run with mock LLM
    python main_loop.py --difficulty medium --seed 42
    python main_loop.py --mock --save-transcript out.txt
"""

from __future__ import annotations
import re
import argparse
import json
import os
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from basis import LegendreBasis
from pde import EllipticPDE, EllipticSolution, solve_elliptic, make_random_elliptic_pde
from weak_form import _composite_simpson
from expression_parser import make_test_function_from_string
from benchmark import score_prediction, ScoreResult, DIFFICULTY_CONFIGS, BenchmarkTask


# ---------------------------------------------------------------------------
# Difficulty config
# ---------------------------------------------------------------------------

DIFFICULTY_CONFIG = {
    "easy":    {"n_basis": 2, "unknowns":  8, "budget": 12},
    "medium":  {"n_basis": 3, "unknowns": 12, "budget": 18},
    "hard":    {"n_basis": 4, "unknowns": 16, "budget": 24},
    "extreme": {"n_basis": 8, "unknowns": 32, "budget": 48},
}

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class ProbeSession:
    pde: EllipticPDE
    solution: EllipticSolution
    basis: LegendreBasis
    difficulty: str
    seed: int
    max_queries: int = 50
    queries_used: int = 0
    history: list = field(default_factory=list)  # list of dicts with stored data
    start_time: float = 0.0
    prediction_submitted: bool = False
    final_score: Optional[ScoreResult] = None
    solution_data: dict = field(default_factory=dict)
    last_solve_coeffs: Optional[dict] = None

    def __post_init__(self):
        self.start_time = time.time()
        if not self.solution_data:
            config = DIFFICULTY_CONFIGS[self.difficulty]
            n_pts = config["n_solution_points"]
            stride = max(1, (len(self.solution.x) - 1) // (n_pts - 1))
            self.solution_data = {
                "x": self.solution.x[::stride].tolist(),
                "u": self.solution.u[::stride].tolist(),
            }

    @classmethod
    def from_difficulty(cls, difficulty="medium", seed=42, max_queries=50):
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
        return cls(pde=pde, solution=sol, basis=basis, difficulty=difficulty,
                   seed=seed, max_queries=max_queries)

    def system_prompt(self) -> str:
        n_b = self.basis.n_basis
        x_str = ", ".join(f"{v:.6f}" for v in self.solution_data["x"])
        u_str = ", ".join(f"{v:.6f}" for v in self.solution_data["u"])
        return f"""You are a mathematician tasked with identifying an unknown elliptic PDE.

The solution u(x) satisfies:
    -(a(x) u'(x))' + b(x) u'(x) + c(x) u(x) = f(x)    on [0, 1]
    u(0) = 0,  u(1) = 0

Each coefficient function a(x), b(x), c(x), f(x) is smooth and represented
by {n_b} parameters. There are {4 * n_b} unknown parameters total.

The weak form identity (integration by parts) gives:
    ∫ a(x)u'(x)φ'(x) dx + ∫ b(x)u'(x)φ(x) dx + ∫ c(x)u(x)φ(x) dx = ∫ f(x)φ(x) dx

for any test function φ with φ(0) = φ(1) = 0.

You have the solution data:
x = [{x_str}]
u = [{u_str}]

Available commands:
    QUERY: <expression>                              → returns ∫f·φ dx  (costs 1 from budget)
    COMPUTE: eval_solution x1 x2 ...                → returns u(x) and u'(x) at given points (free)
    COMPUTE: solve                                   → recovers coefficients from your queries (free, needs ≥{4 * n_b} queries)
                                                       Each QUERY automatically stores all information needed for
                                                       COMPUTE: solve — you do not need to compute term integrals
                                                       manually before solving.
    COMPUTE: check <expression>                      → consistency check: computes predicted LHS from your
                                                       current â,b̂,ĉ and compares to actual ∫f·φ dx for a NEW
                                                       test function. Zero discrepancy does NOT prove correctness —
                                                       it only means this test function is consistent with your
                                                       current fit. Costs 1 query. Also adds this test function
                                                       to your data for future COMPUTE: solve.
                                                       (requires solve first)
    COMPUTE: term_integral <type> <expression>       → returns one weight integral (free)
        diffusion: ∫ u'(x)·φ'(x) dx
        advection: ∫ u'(x)·φ(x) dx
        reaction:  ∫ u(x)·φ(x) dx

    PREDICT:                                         → end the session and submit your current
                                                       COMPUTE: solve coefficients as your answer.
                                                       (requires solve first)

You have {self.max_queries} queries. Your score depends on both accuracy and efficiency.
"""

    def query(self, spec: str) -> dict:
        if self.prediction_submitted:
            return {"status": "error", "message": "Session complete.", "queries_remaining": 0}

        tf, error = make_test_function_from_string(spec, domain=self.pde.domain)
        if error is not None:
            return {
                "status": "error",
                "message": error,
                "queries_remaining": self.max_queries - self.queries_used,
            }

        if self.queries_used >= self.max_queries:
            return {"status": "error", "message": "Query budget exhausted.", "queries_remaining": 0}

        # Evaluate on solution grid
        x_q = self.solution.x
        u_q = self.solution.u
        u_x_q = self.solution.u_x()
        dx = x_q[1] - x_q[0]
        phi_q = tf(x_q)
        dphi_q = tf.derivative(x_q, 1)

        if np.any(np.isnan(phi_q)) or np.any(np.isnan(dphi_q)):
            return {"status": "error", "message": "Test function produced NaN values.",
                    "queries_remaining": self.max_queries - self.queries_used}

        I = lambda y: _composite_simpson(y, dx)

        # The observable: ∫f·φ dx  (= weak form RHS)
        f_vals = self.pde.f(x_q)
        integral_f_phi = round(float(I(f_vals * phi_q)), 12)

        # Internal G matrices for post-hoc scoring (not shown to model)
        psi_q = self.basis.evaluate(x_q)
        n_b = self.basis.n_basis
        G_diff  = [round(I(psi_q[j] * u_x_q * dphi_q), 12) for j in range(n_b)]
        G_adv   = [round(I(psi_q[j] * u_x_q * phi_q),  12) for j in range(n_b)]
        G_react = [round(I(psi_q[j] * u_q   * phi_q),  12) for j in range(n_b)]
        G_src   = [round(I(psi_q[j]          * phi_q),  12) for j in range(n_b)]

        self.queries_used += 1

        record = {
            "spec": spec,
            "integral_f_phi": integral_f_phi,
            "query_number": self.queries_used,
            "queries_remaining": self.max_queries - self.queries_used,
            # Internal only:
            "_G_diff": G_diff, "_G_adv": G_adv,
            "_G_react": G_react, "_G_src": G_src,
        }
        self.history.append(record)
        return {"status": "ok", **{k: v for k, v in record.items() if not k.startswith("_")}}

    def term_integral(self, term_type: str, spec: str) -> dict:
        """
        Compute the weight integral for a specific PDE term.
        Free — no budget cost.

        term_type: 'diffusion', 'advection', or 'reaction'
        spec: test function expression string
        """
        tf, error = make_test_function_from_string(spec, domain=self.pde.domain)
        if error is not None:
            return {"status": "error", "message": error}

        x_q = self.solution.x
        u_q = self.solution.u
        u_x_q = self.solution.u_x()
        dx = x_q[1] - x_q[0]
        phi_q = tf(x_q)
        dphi_q = tf.derivative(x_q, 1)

        if np.any(np.isnan(phi_q)) or np.any(np.isnan(dphi_q)):
            return {"status": "error", "message": "Test function produced NaN values."}

        I = lambda y: _composite_simpson(y, dx)

        if term_type == "diffusion":
            value = I(u_x_q * dphi_q)
        elif term_type == "advection":
            value = I(u_x_q * phi_q)
        elif term_type == "reaction":
            value = I(u_q * phi_q)
        else:
            return {
                "status": "error",
                "message": f"Unknown term type '{term_type}'. Use: diffusion, advection, reaction.",
            }

        return {
            "status": "ok",
            "type": "term_integral",
            "term": term_type,
            "spec": spec,
            "value": round(value, 12),
        }

    def solve(self) -> dict:
        """
        Recover coefficients from QUERY history via least squares.
        Free — no budget cost. Requires at least 4*n_basis queries.
        """
        n_b = self.basis.n_basis
        n_unknowns = 4 * n_b

        rows = [r for r in self.history if "_G_diff" in r]
        n_rows = len(rows)

        if n_rows < n_unknowns:
            return {
                "status": "error",
                "message": (
                    f"Need at least {n_unknowns} queries to solve "
                    f"({n_rows} collected so far)."
                ),
            }

        G_diff  = np.array([r["_G_diff"]  for r in rows])
        G_adv   = np.array([r["_G_adv"]   for r in rows])
        G_react = np.array([r["_G_react"] for r in rows])
        G_src   = np.array([r["_G_src"]   for r in rows])
        rhs     = np.array([r["integral_f_phi"] for r in rows])

        try:
            A_lhs = np.hstack([G_diff, G_adv, G_react])
            theta, _, _, _ = np.linalg.lstsq(A_lhs, rhs, rcond=None)
            a_coeffs = theta[:n_b].tolist()
            b_coeffs = theta[n_b:2*n_b].tolist()
            c_coeffs = theta[2*n_b:3*n_b].tolist()

            f_coeffs, _, _, _ = np.linalg.lstsq(G_src, rhs, rcond=None)
            f_coeffs = f_coeffs.tolist()
        except Exception as e:
            return {"status": "error", "message": f"Solve failed: {e}"}

        self.last_solve_coeffs = {
            "a": a_coeffs,
            "b": b_coeffs,
            "c": c_coeffs,
            "f": f_coeffs,
        }

        type_counts = {"polynomial": 0, "trigonometric": 0, "exponential": 0, "localized": 0, "other": 0}
        for r in rows:
            spec = r.get("spec", "").lower()
            if "exp(" in spec and ("*(" in spec or "*(x" in spec):
                if any(c in spec for c in ["-", "**2"]):
                    type_counts["localized"] += 1
                else:
                    type_counts["exponential"] += 1
            elif "sin" in spec or "cos" in spec:
                type_counts["trigonometric"] += 1
            elif "exp" in spec:
                type_counts["exponential"] += 1
            else:
                type_counts["polynomial"] += 1

        return {
            "status": "ok",
            "type": "solve",
            "n_rows": n_rows,
            "n_unknowns": n_unknowns,
            "a_coeffs": [round(v, 8) for v in a_coeffs],
            "b_coeffs": [round(v, 8) for v in b_coeffs],
            "c_coeffs": [round(v, 8) for v in c_coeffs],
            "f_coeffs": [round(v, 8) for v in f_coeffs],
            "type_counts": type_counts,
        }

    def verify(self, spec: str) -> dict:
        """
        Check estimated coefficients against the true PDE response.
        Costs 1 query. Compares predicted LHS (from â,b̂,ĉ) to actual ∫fφ.
        Also stores G data so future COMPUTE: solve benefits from this query.
        Requires COMPUTE: solve to have been called first.
        """
        if self.last_solve_coeffs is None:
            return {"status": "error", "message": "Call COMPUTE: solve first to get coefficient estimates."}

        if self.prediction_submitted:
            return {"status": "error", "message": "Session complete.", "queries_remaining": 0}

        tf, error = make_test_function_from_string(spec, domain=self.pde.domain)
        if error is not None:
            return {
                "status": "error",
                "message": error,
                "queries_remaining": self.max_queries - self.queries_used,
            }

        if self.queries_used >= self.max_queries:
            return {"status": "error", "message": "Query budget exhausted.", "queries_remaining": 0}

        x_q = self.solution.x
        dx = x_q[1] - x_q[0]
        phi_q = tf(x_q)
        dphi_q = tf.derivative(x_q, 1)

        if np.any(np.isnan(phi_q)) or np.any(np.isnan(dphi_q)):
            return {"status": "error", "message": "Test function produced NaN values.",
                    "queries_remaining": self.max_queries - self.queries_used}

        u_q = self.solution.u
        u_x_q = self.solution.u_x()
        I = lambda y: _composite_simpson(y, dx)

        # Actual RHS from the true PDE
        actual_f_phi = round(float(I(self.pde.f(x_q) * phi_q)), 12)

        # Predicted LHS from estimated a, b, c
        psi_q = self.basis.evaluate(x_q)
        n_b = self.basis.n_basis
        a_coeffs = self.last_solve_coeffs["a"]
        b_coeffs = self.last_solve_coeffs["b"]
        c_coeffs = self.last_solve_coeffs["c"]

        a_pred = sum(a_coeffs[j] * psi_q[j] for j in range(n_b))
        b_pred = sum(b_coeffs[j] * psi_q[j] for j in range(n_b))
        c_pred = sum(c_coeffs[j] * psi_q[j] for j in range(n_b))

        predicted_lhs = round(
            I(a_pred * u_x_q * dphi_q)
            + I(b_pred * u_x_q * phi_q)
            + I(c_pred * u_q * phi_q),
            12,
        )

        # Store G data so future solves benefit from this query
        G_diff  = [round(I(psi_q[j] * u_x_q * dphi_q), 12) for j in range(n_b)]
        G_adv   = [round(I(psi_q[j] * u_x_q * phi_q),  12) for j in range(n_b)]
        G_react = [round(I(psi_q[j] * u_q   * phi_q),  12) for j in range(n_b)]
        G_src   = [round(I(psi_q[j]          * phi_q),  12) for j in range(n_b)]

        self.queries_used += 1

        record = {
            "spec": spec,
            "integral_f_phi": actual_f_phi,
            "query_number": self.queries_used,
            "queries_remaining": self.max_queries - self.queries_used,
            "_G_diff": G_diff, "_G_adv": G_adv,
            "_G_react": G_react, "_G_src": G_src,
        }
        self.history.append(record)

        return {
            "status": "ok",
            "type": "verify",
            "spec": spec,
            "predicted_lhs": predicted_lhs,
            "actual_f_phi": actual_f_phi,
            "discrepancy": round(abs(predicted_lhs - actual_f_phi), 12),
            "query_number": self.queries_used,
            "queries_remaining": self.max_queries - self.queries_used,
        }

    def eval_solution(self, args: str) -> dict:
        try:
            x_vals = np.array([float(v.strip().rstrip(',')) for v in args.split() if v.strip()])
        except ValueError:
            return {"status": "error", "message": f"Cannot parse x-values: '{args}'"}
        if len(x_vals) == 0:
            return {"status": "error", "message": "No x-values provided."}
        u_vals = np.interp(x_vals, self.solution.x, self.solution.u)
        ux_vals = np.interp(x_vals, self.solution.x, self.solution.u_x())
        table = "x\tu(x)\tu'(x)\n"
        for i, xi in enumerate(x_vals):
            table += f"{xi:.6f}\t{u_vals[i]:.10f}\t{ux_vals[i]:.10f}\n"
        return {"status": "ok", "type": "eval_solution", "table": table}

    def submit_prediction(self, prediction: dict) -> dict:
        if self.prediction_submitted:
            return {"status": "error", "message": "Prediction already submitted."}

        n_b = self.basis.n_basis
        for key in ["a", "b", "c", "f"]:
            if key not in prediction:
                return {"status": "error", "message": f"Missing key '{key}'."}
            if len(prediction[key]) != n_b:
                return {"status": "error", "message": f"'{key}' must have {n_b} values."}

        self.prediction_submitted = True
        ground_truth = {k: getattr(self.pde, k).coeffs.tolist() for k in ["a", "b", "c", "f"]}

        task = BenchmarkTask(
            task_id=f"probe_{self.difficulty}_{self.seed}",
            difficulty=self.difficulty,
            prompt="",
            solution_data=self.solution_data,
            basis_info={"type": "LegendreBasis", "n_basis": n_b, "domain": [0.0, 1.0]},
            precomputed_responses={},
            ground_truth=ground_truth,
            pde_description=self.pde.describe(),
        )
        self.final_score = score_prediction(task, prediction, self.basis)
        return {
            "status": "scored",
            "score": self.final_score.to_dict(),
            "summary": self.final_score.summary(),
            "ground_truth": ground_truth,
            "pde_description": self.pde.describe(),
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_probe_response(text: str) -> list[dict]:
    """Parse LLM response into actions: query, compute, predict, reasoning."""
    actions = []
    lines = text.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'^QUERY:\s*(.+)$', line, re.IGNORECASE)
        if m:
            actions.append({"action": "query", "spec": m.group(1).strip().strip('"\'`')})
            i += 1
            continue
        m = re.match(r'^COMPUTE:\s*(.+)$', line, re.IGNORECASE)
        if m:
            actions.append({"action": "compute", "command": m.group(1).strip()})
            i += 1
            continue
        if re.match(r'^PREDICT:', line, re.IGNORECASE):
            actions.append({"action": "predict"})
            i += 1
            continue
        if line:
            actions.append({"action": "reasoning", "text": line})
        i += 1
    return actions


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_query_result(response: dict) -> str:
    if response["status"] == "ok":
        return (
            f"QUERY result for '{response['spec']}':\n"
            f"  ∫f·φ dx = {response['integral_f_phi']:+.12f}\n"
            f"  (Query {response['query_number']}, {response['queries_remaining']} remaining)\n"
            f"  [Query stored — call COMPUTE: solve when ready]"
        )
    return f"Error: {response['message']}"


def format_eval_solution(response: dict) -> str:
    if response["status"] == "ok":
        return f"Solution values:\n  {response['table'].replace(chr(10), chr(10) + '  ')}"
    return f"Error: {response['message']}"


def format_solve_result(response: dict, prev_coeffs: dict = None) -> str:
    if response["status"] != "ok":
        return f"Error: {response['message']}"
    lines = [
        f"Solve result ({response['n_rows']} equations, {response['n_unknowns']} unknowns):",
        f"  a_coeffs = {response['a_coeffs']}",
        f"  b_coeffs = {response['b_coeffs']}",
        f"  c_coeffs = {response['c_coeffs']}",
        f"  f_coeffs = {response['f_coeffs']}",
    ]
    if "type_counts" in response:
        tc = response["type_counts"]
        parts = [f"{v} {k}" for k, v in tc.items() if v > 0]
        lines.append(f"  Test functions used: {', '.join(parts)}")
    # Show deltas from previous solve so model can assess convergence
    if prev_coeffs is not None:
        deltas = {}
        for key in ["a", "b", "c", "f"]:
            prev = prev_coeffs.get(key, [])
            curr = response.get(f"{key}_coeffs", [])
            if prev and curr and len(prev) == len(curr):
                max_delta = max(abs(c - p) for c, p in zip(curr, prev))
                deltas[key] = max_delta
        if deltas:
            delta_parts = [f"{k}={v:.2e}" for k, v in deltas.items()]
            lines.append(f"  Max coeff change from prev solve: {', '.join(delta_parts)}")
    return "\n".join(lines)


def format_check_result(response: dict) -> str:
    if response["status"] != "ok":
        return f"Error: {response['message']}"
    return (
        f"Consistency check for '{response['spec']}':\n"
        f"  Predicted LHS (â·∫u'φ' + b̂·∫u'φ + ĉ·∫uφ) = {response['predicted_lhs']:+.12f}\n"
        f"  Actual    ∫f·φ dx                           = {response['actual_f_phi']:+.12f}\n"
        f"  Discrepancy:                                  {response['discrepancy']:.6e}\n"
        f"  (Query {response['query_number']}, {response['queries_remaining']} remaining)"
    )


def format_term_integral(response: dict) -> str:
    if response["status"] == "ok":
        return (
            f"Term integral ({response['term']}) for '{response['spec']}': "
            f"{response['value']:+.12f}"
        )
    return f"Error: {response['message']}"


def format_score_result(result: dict) -> str:
    return (
        f"=== FINAL SCORE ===\n\n"
        f"{result['summary']}\n\n"
        f"Ground truth PDE:\n{result['pde_description']}\n"
    )


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _compute_stability(solves: list) -> dict | None:
    """Return mean per-coefficient std across the last (up to 3) solves."""
    if len(solves) < 2:
        return None
    recent = solves[-min(3, len(solves)):]
    stability = {}
    for name in ["a", "b", "c", "f"]:
        values = [s["coefficients"][name] for s in recent]
        arr = np.array(values)
        stability[name] = float(np.mean(np.std(arr, axis=0)))
    return stability


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class AnthropicBackend:
    """Calls the Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 4096):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Install the anthropic SDK: pip install anthropic\n"
                "And set ANTHROPIC_API_KEY environment variable."
            )
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, system: str, messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text


class OpenAIBackend:
    """Calls the OpenAI API."""

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 4096):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Install the openai SDK: pip install openai\n"
                "And set OPENAI_API_KEY environment variable."
            )
        self.client = openai.OpenAI()
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, system: str, messages: list[dict]) -> str:
        import openai as _openai
        openai_messages = [{"role": "system", "content": system}] + messages
        for attempt in range(8):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=self.max_tokens,
                    messages=openai_messages,
                )
                return response.choices[0].message.content
            except _openai.RateLimitError:
                if attempt == 7:
                    raise
                wait = min(2 ** attempt * 5, 120)
                print(f"\n  [rate limit, retry {attempt+1}/7 in {wait}s]", end=" ", flush=True)
                time.sleep(wait)


class MockBackend:
    """Simple mock that queries a few test functions then predicts zeros."""
    def __init__(self):
        self.model = "mock"
        self.call_count = 0
        self.script = [
            "COMPUTE: eval_solution 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9",
            "QUERY: sin(pi*x)",
            "QUERY: sin(2*pi*x)",
            "QUERY: sin(3*pi*x)",
            "QUERY: x*(1-x)",
            "QUERY: x*(1-x)*(2*x-1)",
            "QUERY: x**2*(1-x)**2",
            "QUERY: x*(1-x)*sin(pi*x)",
            "PREDICT:\na_coeffs = [1.0, 0.0, 0.0]\nb_coeffs = [0.0, 0.0, 0.0]\nc_coeffs = [0.0, 0.0, 0.0]\nf_coeffs = [0.0, 0.0, 0.0]",
        ]

    def chat(self, system, messages):
        idx = min(self.call_count, len(self.script) - 1)
        self.call_count += 1
        return self.script[idx]


# ---------------------------------------------------------------------------
# Post-hoc efficiency curve
# ---------------------------------------------------------------------------

def compute_efficiency_curve(session: ProbeSession) -> dict:
    """Replay stored G matrices, solve at each step, compare to ground truth."""
    import numpy as np

    n_b = session.basis.n_basis
    pde = session.pde

    # Collect rows with stored G data
    history_rows = [r for r in session.history if "_G_diff" in r]
    if not history_rows:
        return {"queries": [], "errors": [], "auc": float("inf")}

    errors, queries = [], []
    a_errors, b_errors, c_errors, f_errors = [], [], [], []

    for k in range(1, len(history_rows) + 1):
        rows = history_rows[:k]
        G_diff  = np.array([r["_G_diff"]  for r in rows])
        G_adv   = np.array([r["_G_adv"]   for r in rows])
        G_react = np.array([r["_G_react"] for r in rows])
        G_src   = np.array([r["_G_src"]   for r in rows])
        rhs     = np.array([r["integral_f_phi"] for r in rows])
        queries.append(k)
        if k < n_b:
            errors.append(float("inf"))
            a_errors.append(float("inf"))
            b_errors.append(float("inf"))
            c_errors.append(float("inf"))
            f_errors.append(float("inf"))
            continue
        try:
            f_rec, _, _, _ = np.linalg.lstsq(G_src, rhs, rcond=None)
            A_lhs = np.hstack([G_diff, G_adv, G_react])
            theta, _, _, _ = np.linalg.lstsq(A_lhs, rhs, rcond=None)

            a_err = float(np.mean(np.abs(theta[:n_b]      - pde.a.coeffs)))
            b_err = float(np.mean(np.abs(theta[n_b:2*n_b] - pde.b.coeffs)))
            c_err = float(np.mean(np.abs(theta[2*n_b:]    - pde.c.coeffs)))
            f_err = float(np.mean(np.abs(f_rec             - pde.f.coeffs)))

            a_errors.append(a_err)
            b_errors.append(b_err)
            c_errors.append(c_err)
            f_errors.append(f_err)
            errors.append(a_err + b_err + c_err + f_err)
        except Exception:
            errors.append(float("inf"))
            a_errors.append(float("inf"))
            b_errors.append(float("inf"))
            c_errors.append(float("inf"))
            f_errors.append(float("inf"))

    def auc_from(errs):
        finite = [e for e in errs if np.isfinite(e)]
        return float(np.mean(finite)) if finite else float("inf")

    return {
        "queries":    queries,
        "errors":     errors,
        "a_errors":   a_errors,
        "b_errors":   b_errors,
        "c_errors":   c_errors,
        "f_errors":   f_errors,
        "auc":        auc_from(errors),
        "auc_a":      auc_from(a_errors),
        "auc_b":      auc_from(b_errors),
        "auc_c":      auc_from(c_errors),
        "auc_f":      auc_from(f_errors),
        "final_error": errors[-1] if errors else float("inf"),
        "queries_to_below_1":   next((q for q, e in zip(queries, errors) if e < 1.0),  None),
        "queries_to_below_0.1": next((q for q, e in zip(queries, errors) if e < 0.1), None),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

protocol = """
Rules:
- ONE QUERY per response.
- COMPUTE: eval_solution is free and unlimited.
- COMPUTE: solve is free and unlimited — call it after every few queries
  to track how your estimates evolve. This is good experimental practice.
- You may NOT PREDICT in the same response as a QUERY.
"""


def run_probe_session(session: ProbeSession, backend, verbose: bool = True,
                      max_turns: int = 100, baseline: bool = False,
                      min_queries: int = None, run_log: Optional[dict] = None) -> dict:
    full_system = session.system_prompt() + protocol
    full_system += """
After each COMPUTE: solve, report your confidence (0-100%) that \
each coefficient vector (a, b, c, f) is within 0.1 of the true \
value. Format exactly as:
  Confidence: a=XX%, b=XX%, c=XX%, f=XX%
Then give one sentence explaining the main source of your uncertainty.
"""
    if min_queries:
        full_system += (
            f"\nYou must submit at least {min_queries} queries before calling COMPUTE: solve or PREDICT. "
            f"You have {max_turns} turns total. Each QUERY or COMPUTE uses one turn, so plan efficiently — "
            f"you do not need to call term_integral for every query.\n"
        )
    if baseline:
        initial_message = (
            "Begin. Before each action, briefly explain your reasoning: "
            "what you expect to learn from it and why you chose it over alternatives."
        )
    else:
        initial_message = (
            "Begin. Before each action, briefly explain your reasoning: "
            "what you expect to learn from it and why you chose it over alternatives. "
            "Note: in elliptic inverse problems of this type, the three "
            "coefficient functions a(x), b(x), c(x) are not equally identifiable "
            "from weak-form data. The difficulty of recovering each coefficient "
            "depends on the magnitude of its contribution to the weak-form "
            "equations relative to the others — and this varies with the solution "
            "u(x) and the test functions you choose."
        )
    messages = [{"role": "user", "content": initial_message}]
    turn = 0
    done = False
    explicit_prediction = None
    _prev_solve_coeffs = {}  # track previous solve for delta display

    if verbose:
        print("=" * 70)
        print("PROBE SESSION START")
        print("=" * 70)
        model_name = getattr(backend, "model", "unknown")
        condition = "baseline" if baseline else "standard"
        print(f"Model:      {model_name}")
        print(f"Prompt:     {condition}")
        print(f"Difficulty: {session.difficulty}, n_basis: {session.basis.n_basis}, "
              f"seed: {session.seed}, budget: {session.max_queries}, turns: {max_turns}")
        print()

    while not done and turn < max_turns:
        turn += 1
        if verbose:
            print(f"--- Turn {turn} ---")

        llm_text = backend.chat(full_system, messages)
        if verbose:
            display = llm_text[:500] + ("..." if len(llm_text) > 500 else "")
            print(f"LLM: {display}\n")

        messages.append({"role": "assistant", "content": llm_text})
        actions = parse_probe_response(llm_text)
        result_parts = []
        query_executed = False

        if run_log is not None:
            run_log["turns"].append({
                "turn": turn,
                "timestamp": time.time(),
                "role": "assistant",
                "content": llm_text,
                "parsed_actions": [a["action"] for a in actions],
            })

        # Pass 1: free COMPUTE actions
        for act in actions:
            if act["action"] == "compute":
                cmd = act["command"]
                if cmd.startswith("eval_solution"):
                    cmd_args = cmd[len("eval_solution"):].strip()
                    resp = session.eval_solution(cmd_args)
                    result_text = format_eval_solution(resp)
                elif cmd == "solve":
                    if min_queries and session.queries_used < min_queries:
                        resp = {"status": "error",
                                "message": f"Need at least {min_queries} queries before solving "
                                           f"({session.queries_used} collected so far)."}
                        prev = None
                    else:
                        prev = _prev_solve_coeffs.copy() if _prev_solve_coeffs else None
                        resp = session.solve()
                    result_text = format_solve_result(resp, prev_coeffs=prev)
                    if resp.get("status") == "ok":
                        _prev_solve_coeffs.update({
                            "a": resp["a_coeffs"], "b": resp["b_coeffs"],
                            "c": resp["c_coeffs"], "f": resp["f_coeffs"],
                        })
                    if run_log is not None and resp.get("status") == "ok":
                        pde = session.pde
                        a_err = float(np.mean(np.abs(np.array(resp["a_coeffs"]) - pde.a.coeffs)))
                        b_err = float(np.mean(np.abs(np.array(resp["b_coeffs"]) - pde.b.coeffs)))
                        c_err = float(np.mean(np.abs(np.array(resp["c_coeffs"]) - pde.c.coeffs)))
                        f_err = float(np.mean(np.abs(np.array(resp["f_coeffs"]) - pde.f.coeffs)))
                        solve_entry = {
                            "solve_number": len(run_log["solves"]) + 1,
                            "turn": turn,
                            "queries_at_solve": session.queries_used,
                            "n_equations": resp["n_rows"],
                            "n_unknowns": resp["n_unknowns"],
                            "coefficients": {
                                "a": resp["a_coeffs"],
                                "b": resp["b_coeffs"],
                                "c": resp["c_coeffs"],
                                "f": resp["f_coeffs"],
                            },
                            "coeff_errors": {"a": a_err, "b": b_err, "c": c_err, "f": f_err,
                                             "total": a_err + b_err + c_err + f_err},
                            "test_function_taxonomy": resp.get("type_counts", {}),
                            "timestamp": time.time(),
                        }
                        run_log["solves"].append(solve_entry)
                        run_log["error_curves"].append({
                            "query_count": session.queries_used,
                            "a_error": a_err,
                            "b_error": b_err,
                            "c_error": c_err,
                            "f_error": f_err,
                            "total_error": a_err + b_err + c_err + f_err,
                        })
                elif cmd.startswith("check") or cmd.startswith("verify"):
                    keyword = "check" if cmd.startswith("check") else "verify"
                    cmd_args = cmd[len(keyword):].strip()
                    if not cmd_args:
                        result_text = "Usage: COMPUTE: check <expression>"
                        resp = None
                    else:
                        resp = session.verify(cmd_args)
                        result_text = format_check_result(resp)
                        # Warn if system was exactly determined at last solve
                        if resp.get("status") == "ok" and resp.get("discrepancy", 1) < 1e-10:
                            n_eqs = len([r for r in session.history if "_G_diff" in r]) - 1  # minus this check
                            n_unk = 4 * session.basis.n_basis
                            if n_eqs <= n_unk:
                                result_text += (
                                    f"\n  ⚠ Note: with {n_eqs} equations for {n_unk} unknowns, "
                                    f"the system was exactly determined or underdetermined at last solve. "
                                    f"Zero discrepancy is expected and does NOT confirm correctness — "
                                    f"any solution to the linear system will satisfy test functions "
                                    f"in the span of your queries."
                                )
                    if run_log is not None and resp is not None and resp.get("status") == "ok":
                        run_log["verifications"].append({
                            "turn": turn,
                            "query_number": resp["query_number"],
                            "test_function": resp["spec"],
                            "predicted_lhs": float(resp["predicted_lhs"]),
                            "actual_rhs": float(resp["actual_f_phi"]),
                            "discrepancy": float(resp["discrepancy"]),
                            "timestamp": time.time(),
                        })
                elif cmd.startswith("term_integral"):
                    parts = cmd[len("term_integral"):].strip().split(None, 1)
                    if len(parts) < 2:
                        result_text = "Usage: COMPUTE: term_integral <diffusion|advection|reaction> <expression>"
                        resp = None
                    else:
                        resp = session.term_integral(parts[0].lower(), parts[1])
                        result_text = format_term_integral(resp)
                    if run_log is not None and resp is not None and resp.get("status") == "ok":
                        run_log["term_integrals"].append({
                            "turn": turn,
                            "term": resp["term"],
                            "test_function": resp["spec"],
                            "value": float(resp["value"]),
                            "timestamp": time.time(),
                        })
                else:
                    result_text = f"Unknown command: '{cmd}'. Available: eval_solution, solve, check, term_integral"
                result_parts.append(result_text)
                if verbose:
                    print(f"  → {result_text}\n")

        # Pass 2: one QUERY, then PREDICT (only if no query this turn)
        for act in actions:
            if act["action"] == "compute":
                continue
            elif act["action"] == "query":
                if not query_executed:
                    resp = session.query(act["spec"])
                    result_text = format_query_result(resp)
                    result_parts.append(result_text)
                    query_executed = True
                    if run_log is not None and resp.get("status") == "ok":
                        run_log["queries"].append({
                            "query_number": resp["query_number"],
                            "turn": turn,
                            "test_function": resp["spec"],
                            "rhs_value": float(resp["integral_f_phi"]),
                            "timestamp": time.time(),
                        })
                    if verbose:
                        print(f"  → {result_text}\n")
                else:
                    msg = "One QUERY per turn. Review the result above, then submit your next action."
                    result_parts.append(msg)
            elif act["action"] == "predict":
                if query_executed:
                    msg = "You may not PREDICT in the same response as a QUERY. Submit PREDICT alone."
                    result_parts.append(msg)
                elif min_queries and session.queries_used < min_queries:
                    msg = (f"You have only used {session.queries_used} queries. "
                           f"Submit at least {min_queries} queries before predicting "
                           f"to ensure the system is well-overdetermined.")
                    result_parts.append(msg)
                elif session.last_solve_coeffs is None:
                    msg = "No COMPUTE: solve has been run yet. Call COMPUTE: solve before PREDICT."
                    result_parts.append(msg)
                else:
                    prediction = session.last_solve_coeffs
                    result = session.submit_prediction(prediction)
                    if result["status"] == "scored":
                        score_text = format_score_result(result)
                        result_parts.append(score_text)
                        done = True
                        explicit_prediction = prediction
                        if verbose:
                            print(f"\n{score_text}\n")
                    else:
                        result_parts.append(f"Error: {result['message']}")

        if not any(a["action"] in ("query", "compute", "predict") for a in actions):
            result_parts.append(
                "No recognized action found. Use QUERY: <expression>, "
                "COMPUTE: eval_solution <x1> ..., COMPUTE: solve, "
                "COMPUTE: term_integral <type> <expr>, or PREDICT:."
            )

        if result_parts and not done:
            result_parts.append(f"[Turn {turn}/{max_turns}, {session.queries_used} queries used]")
            messages.append({"role": "user", "content": "\n\n".join(result_parts)})

        if session.queries_used >= session.max_queries and not session.prediction_submitted:
            messages.append({
                "role": "user",
                "content": f"Budget exhausted ({session.max_queries} queries). Call COMPUTE: solve if you haven't, then submit PREDICT: to end the session."
            })

    output = {
        "turns": turn,
        "queries_used": session.queries_used,
        "prediction_submitted": session.prediction_submitted,
        "messages": messages,
    }
    if session.final_score:
        output["score"] = session.final_score.to_dict()
    if explicit_prediction is not None:
        output["prediction"] = explicit_prediction

    # Auto-submit from best solve if model never predicted (or ran out of turns)
    if not session.prediction_submitted and session.last_solve_coeffs is not None:
        if verbose:
            print("(Auto-submitting from last COMPUTE: solve result)")
        prediction = session.last_solve_coeffs
        result = session.submit_prediction(prediction)
        if result["status"] == "scored":
            output["score"] = result["score"].to_dict() if hasattr(result["score"], "to_dict") else result["score"]
            output["prediction"] = prediction
            output["prediction_submitted"] = True
            output["auto_submitted"] = True
            if verbose:
                print(format_score_result(result))

    # Also compute what the best solve would have scored (for analysis)
    if run_log is not None and run_log["solves"]:
        best_solve = min(run_log["solves"], key=lambda s: s["coeff_errors"]["total"])
        output["best_solve_error"] = best_solve["coeff_errors"]["total"]
        output["best_solve_number"] = best_solve["solve_number"]
        output["best_solve_queries"] = best_solve["queries_at_solve"]

    output["efficiency"] = compute_efficiency_curve(session)
    return output


# ---------------------------------------------------------------------------
# Metacognitive metrics
# ---------------------------------------------------------------------------

def parse_confidence_reports(messages: list) -> list:
    """
    Extract confidence reports from assistant messages.
    Returns list of {query_count, a, b, c, f} dicts.
    """
    import re
    reports = []
    query_count = 0

    for msg in messages:
        if msg["role"] == "user":
            query_count += msg["content"].count("Query stored")
        if msg["role"] == "assistant":
            pattern = r'Confidence:\s*a=(\d+)%,\s*b=(\d+)%,\s*c=(\d+)%,\s*f=(\d+)%'
            for m in re.finditer(pattern, msg["content"], re.IGNORECASE):
                reports.append({
                    "query_count": query_count,
                    "a": int(m.group(1)) / 100,
                    "b": int(m.group(2)) / 100,
                    "c": int(m.group(3)) / 100,
                    "f": int(m.group(4)) / 100,
                })
    return reports


def compute_metacognitive_metrics(session: ProbeSession,
                                   messages: list) -> dict:
    """
    Compute monitoring accuracy, control efficiency, and calibration
    at each query step from stored G matrices.
    """
    from scipy.stats import spearmanr

    n_b = session.basis.n_basis
    rows = [r for r in session.history if "_G_diff" in r]

    confidence_reports = parse_confidence_reports(messages)

    monitoring = []
    control = []
    sigma_curves = []

    for k in range(n_b + 1, len(rows) + 1):
        subset = rows[:k]
        G_diff  = np.array([r["_G_diff"]  for r in subset])
        G_adv   = np.array([r["_G_adv"]   for r in subset])
        G_react = np.array([r["_G_react"] for r in subset])
        A = np.hstack([G_diff, G_adv, G_react])

        try:
            ATA = A.T @ A
            U, s, Vt = np.linalg.svd(ATA)
            threshold = 1e-10 * s[0]
            s_inv = np.where(s > threshold, 1.0 / s, 0.0)
            ATA_inv = (Vt.T * s_inv) @ U.T
            sigma_a = np.sqrt(np.mean(np.diag(ATA_inv)[:n_b]))
            sigma_b = np.sqrt(np.mean(np.diag(ATA_inv)[n_b:2*n_b]))
            sigma_c = np.sqrt(np.mean(np.diag(ATA_inv)[2*n_b:3*n_b]))
            sigma_curves.append({
                "k": k, "sigma_a": sigma_a,
                "sigma_b": sigma_b, "sigma_c": sigma_c
            })

            if k < len(rows):
                next_row = rows[k]
                g_next = np.concatenate([
                    next_row["_G_diff"],
                    next_row["_G_adv"],
                    next_row["_G_react"]
                ])

                # Full Sherman-Morrison update on the (3*n_b x 3*n_b) inverse
                Hg = ATA_inv @ g_next
                denom = 1.0 + g_next @ Hg
                ATA_inv_new = ATA_inv - np.outer(Hg, Hg) / denom

                def block_sigma(inv_matrix, start, end):
                    return np.sqrt(np.mean(np.diag(inv_matrix)[start:end]))

                ig_a = sigma_a - block_sigma(ATA_inv_new, 0,       n_b)
                ig_b = sigma_b - block_sigma(ATA_inv_new, n_b,   2*n_b)
                ig_c = sigma_c - block_sigma(ATA_inv_new, 2*n_b, 3*n_b)

                igs = {"a": ig_a, "b": ig_b, "c": ig_c}
                max_ig = max(igs.values())
                if max_ig > 1e-12:
                    sigmas = {"a": sigma_a, "b": sigma_b, "c": sigma_c}
                    j_star = max(sigmas, key=sigmas.get)
                    C_k = igs[j_star] / max_ig
                    control.append({"k": k, "C_k": C_k, "j_star": j_star})

        except np.linalg.LinAlgError:
            continue

    for report in confidence_reports:
        k = report["query_count"]
        matching = [s for s in sigma_curves if s["k"] == k]
        if not matching:
            if not sigma_curves:
                continue
            matching = [min(sigma_curves, key=lambda s: abs(s["k"] - k))]
        s = matching[0]
        true_uncertainty = [s["sigma_a"], s["sigma_b"], s["sigma_c"]]
        stated_uncertainty = [1 - report["a"], 1 - report["b"], 1 - report["c"]]
        if len(set(true_uncertainty)) > 1:
            rho, _ = spearmanr(true_uncertainty, stated_uncertainty)
            monitoring.append({
                "k": k, "M_k": rho,
                "true": true_uncertainty,
                "stated": stated_uncertainty
            })

    return {
        "sigma_curves": sigma_curves,
        "monitoring": monitoring,
        "control": control,
        "confidence_reports": confidence_reports,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def plot_auc_curves(efficiency_results: dict, save_path: Optional[str] = None):
    import matplotlib.pyplot as plt
    import numpy as np

    coeff_colors = {
        "a": "steelblue",
        "b": "darkorange",
        "c": "firebrick",
        "f": "forestgreen",
    }

    # One linestyle per model to distinguish them
    model_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    coeff_axes = {"a": axes[0, 0], "b": axes[0, 1],
                  "c": axes[1, 0], "f": axes[1, 1]}
    coeff_names = {
        "a": "Diffusion a(x)",
        "b": "Advection b(x)",
        "c": "Reaction c(x)",
        "f": "Source f(x)",
    }

    for (model_name, result), lstyle in zip(efficiency_results.items(), model_styles):
        queries = result["queries"]

        for coeff in ["a", "b", "c", "f"]:
            ax = coeff_axes[coeff]
            errs = result[f"{coeff}_errors"]
            finite = [(q, e) for q, e in zip(queries, errs) if np.isfinite(e)]
            if not finite:
                continue
            qs, es = zip(*finite)
            auc = result[f"auc_{coeff}"]
            ax.plot(qs, es,
                    color=coeff_colors[coeff],
                    linestyle=lstyle,
                    linewidth=2,
                    label=f"{model_name}  (AUC={auc:.3f})")
            ax.scatter([qs[-1]], [es[-1]],
                       color=coeff_colors[coeff], s=50, zorder=5)

    for coeff, ax in coeff_axes.items():
        ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.8,
                   alpha=0.7, label="0.1 threshold")
        ax.set_yscale("log")
        ax.set_title(coeff_names[coeff], fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean absolute error", fontsize=9)
        ax.set_xlabel("Queries submitted", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Coefficient recovery curves by model\n"
                 "AUC = mean error over session (lower = better)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Probe-only PDE identification benchmark")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard", "extreme"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Override auto-scaled query budget (default: from DIFFICULTY_CONFIG)")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"],
                        help="Which LLM provider to use (default: anthropic)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: claude-haiku-4-5-20251001 for anthropic, gpt-4o for openai)")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Override auto-scaled turn limit (default: 2x budget)")
    parser.add_argument("--save-transcript", type=str, default=None)
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Save results plot to file (default: probe_<difficulty>_<seed>.png)")
    parser.add_argument("--baseline", action="store_true",
                        help="Use minimal prompt without mathematical context (for ablation)")
    parser.add_argument("--min-queries", type=int, default=None,
                        help="Minimum queries before PREDICT is allowed")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for JSON log and plot output (default: current dir)")
    args = parser.parse_args()

    if args.model is None:
        args.model = "gpt-4o" if args.provider == "openai" else "claude-haiku-4-5-20251001"

    config = DIFFICULTY_CONFIG[args.difficulty]
    max_queries = args.max_queries if args.max_queries is not None else config["budget"]
    max_turns = args.max_turns if args.max_turns is not None else 2 * max_queries

    os.makedirs(args.output_dir, exist_ok=True)

    session = ProbeSession.from_difficulty(args.difficulty, args.seed, max_queries)
    if args.mock:
        backend = MockBackend()
    elif args.provider == "openai":
        backend = OpenAIBackend(model=args.model)
    else:
        backend = AnthropicBackend(model=args.model)

    run_log = {
        "config": {
            "model": args.model,
            "provider": args.provider,
            "difficulty": args.difficulty,
            "n_basis": config["n_basis"],
            "unknowns": config["unknowns"],
            "seed": args.seed,
            "max_queries": max_queries,
            "max_turns": max_turns,
            "budget_source": "override" if args.max_queries is not None else "auto",
            "prompt_condition": "baseline" if args.baseline else "standard",
            "timestamp": datetime.now().isoformat(),
        },
        "turns": [],
        "solves": [],
        "queries": [],
        "verifications": [],
        "term_integrals": [],
        "error_curves": [],
        "results": None,
        "ground_truth": None,
        "behavioral_metrics": None,
    }

    result = run_probe_session(session, backend, verbose=True,
                               max_turns=max_turns, baseline=args.baseline,
                               min_queries=args.min_queries, run_log=run_log)

    # Finalize run_log
    pde = session.pde
    run_log["ground_truth"] = {
        "a_coeffs": pde.a.coeffs.tolist(),
        "b_coeffs": pde.b.coeffs.tolist(),
        "c_coeffs": pde.c.coeffs.tolist(),
        "f_coeffs": pde.f.coeffs.tolist(),
        "pde_string": pde.describe(),
    }

    if "score" in result:
        sc = result["score"]
        run_log["results"] = {
            "coefficient_errors": {
                "a": sc.get("a_coeff_error"), "b": sc.get("b_coeff_error"),
                "c": sc.get("c_coeff_error"), "f": sc.get("f_coeff_error"),
                "total": sc.get("total_coeff_error"),
            },
            "pointwise_errors": {
                "a": sc.get("a_pointwise_error"), "b": sc.get("b_pointwise_error"),
                "c": sc.get("c_pointwise_error"), "f": sc.get("f_pointwise_error"),
                "total": sc.get("total_pointwise_error"),
            },
        }

    if "efficiency" in result:
        eff = result["efficiency"]
        if run_log["results"] is None:
            run_log["results"] = {}
        run_log["results"]["efficiency_auc"] = eff.get("auc")
        run_log["results"]["queries_to_error_1"] = eff.get("queries_to_below_1")
        run_log["results"]["queries_to_error_01"] = eff.get("queries_to_below_0.1")

    run_log["behavioral_metrics"] = {
        "total_turns": result["turns"],
        "queries_used": result["queries_used"],
        "budget_utilization": result["queries_used"] / max_queries,
        "solve_count": len(run_log["solves"]),
        "check_count": len(run_log["verifications"]),
        "term_integral_count": len(run_log["term_integrals"]),
        "stopped_early": result["queries_used"] < max_queries,
        "unused_queries": max_queries - result["queries_used"],
        "coefficient_stability": _compute_stability(run_log["solves"]),
        "auto_submitted": result.get("auto_submitted", False),
    }

    # Track best solve vs submitted prediction for analysis
    if run_log["solves"]:
        best_solve = min(run_log["solves"], key=lambda s: s["coeff_errors"]["total"])
        run_log["behavioral_metrics"]["best_solve_error"] = best_solve["coeff_errors"]["total"]
        run_log["behavioral_metrics"]["best_solve_number"] = best_solve["solve_number"]
        run_log["behavioral_metrics"]["submitted_error"] = (
            run_log["results"]["coefficient_errors"]["total"]
            if run_log.get("results") and run_log["results"].get("coefficient_errors")
            else None
        )

    model_slug = args.model.replace("/", "_")
    run_stem = (
        f"run_{args.difficulty}_s{args.seed}_{model_slug}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    log_path = os.path.join(args.output_dir, run_stem + ".json")
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2, default=str)
    print(f"Run log saved to {log_path}")

    if args.save_transcript and "messages" in result:
        with open(args.save_transcript, "w") as f:
            for msg in result["messages"]:
                f.write(f"[{msg['role'].upper()}]\n{msg['content']}\n\n")
        print(f"Transcript saved to {args.save_transcript}")

    # Generate results plot
    if session.prediction_submitted and "prediction" in result:
        plot_path = args.save_plot
        if plot_path is None:
            plot_path = os.path.join(args.output_dir, run_stem + "_dashboard.png")
        try:
            import matplotlib
            matplotlib.use("Agg")
            from diagnostics import plot_session_results
            fig = plot_session_results(session, result["prediction"], save_path=plot_path)
            import matplotlib.pyplot as _plt
            _plt.close(fig)
            print(f"Results plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not generate plot: {e}")

    if "efficiency" in result:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            efficiency_results = {args.model: result["efficiency"]}
            auc_path = os.path.join(args.output_dir, run_stem + "_auc.png")
            fig = plot_auc_curves(efficiency_results, save_path=auc_path)
            _plt.close(fig)
            print(f"AUC comparison plot saved to {auc_path}")
        except Exception as e:
            print(f"Could not generate AUC plot: {e}")

    print(f"\n{'=' * 70}")
    print(f"Session complete: {result['turns']} turns, {result['queries_used']} queries used")
    if "score" in result:
        print(f"Total coefficient error: {result['score']['total_coeff_error']:.4f}")
        print(f"Total pointwise error:   {result['score']['total_pointwise_error']:.4f}")
    if "efficiency" in result:
        eff = result["efficiency"]
        print(f"Efficiency AUC:          {eff['auc']:.4f}")
        print(f"Queries to error < 1.0:  {eff['queries_to_below_1']}")
        print(f"Queries to error < 0.1:  {eff['queries_to_below_0.1']}")


if __name__ == "__main__":
    main()