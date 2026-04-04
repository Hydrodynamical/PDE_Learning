"""
Probe-only PDE identification benchmark.

A stripped-down alternative to llm_loop.py. The LLM receives solution data,
can query weak-form integrals (QUERY), evaluate the solution (COMPUTE), and
submit a coefficient prediction (PREDICT).

Usage:
    python probe_loop.py --mock                       # run with mock LLM
    python probe_loop.py --difficulty medium --seed 42
    python probe_loop.py --mock --save-transcript out.txt
"""

from __future__ import annotations
import re
import argparse
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from basis import LegendreBasis
from pde import EllipticPDE, EllipticSolution, solve_elliptic, make_random_elliptic_pde
from weak_form import _composite_simpson
from expression_parser import make_test_function_from_string
from benchmark import score_prediction, ScoreResult, DIFFICULTY_CONFIGS, BenchmarkTask


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
    COMPUTE: verify <expression>                     → compares your estimated LHS (â·∫u'φ'+b̂·∫u'φ'+ĉ·∫uφ)
                                                       to the actual ∫f·φ dx. Large discrepancy means your
                                                       coefficients are wrong. Costs 1 query. Also adds this
                                                       test function to your data for future COMPUTE: solve.
                                                       (costs 1 query, requires solve first)
    COMPUTE: term_integral <type> <expression>       → returns one weight integral (free)
        diffusion: ∫ u'(x)·φ'(x) dx
        advection: ∫ u'(x)·φ(x) dx
        reaction:  ∫ u(x)·φ(x) dx

    The weak form equation is:
        a·(diffusion weight) + b·(advection weight) + c·(reaction weight) = ∫f·φ dx
    where the right-hand side comes from QUERY.

    PREDICT:                                         → submit your answer
        a_coeffs = [...]
        b_coeffs = [...]
        c_coeffs = [...]
        f_coeffs = [...]

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

def _parse_prediction(text: str) -> Optional[dict]:
    """
    Extract coefficient arrays from prediction text.

    Looks for patterns like:
        a_coeffs = [1.0, 0.5, -0.3]
        a = [1.0, 0.5, -0.3]
    """
    result = {}
    for key in ["a", "b", "c", "f"]:
        # Try "X_coeffs = [...]" or "X = [...]"
        pattern = rf'{key}(?:_coeffs)?\s*=\s*\[([^\]]+)\]'
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                vals = [float(v.strip()) for v in m.group(1).split(",")]
                result[key] = vals
            except ValueError:
                return None
        else:
            return None
    return result


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
            pred_lines = []
            i += 1
            while i < len(lines):
                pred_lines.append(lines[i])
                i += 1
            pred_text = "\n".join(pred_lines)
            prediction = _parse_prediction(pred_text)
            if prediction:
                actions.append({"action": "predict", **prediction})
            else:
                actions.append({"action": "reasoning", "text": f"PREDICT:\n{pred_text}  [could not parse]"})
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
            f"  (Query {response['query_number']}, {response['queries_remaining']} remaining)"
        )
    return f"Error: {response['message']}"


def format_eval_solution(response: dict) -> str:
    if response["status"] == "ok":
        return f"Solution values:\n  {response['table'].replace(chr(10), chr(10) + '  ')}"
    return f"Error: {response['message']}"


def format_solve_result(response: dict) -> str:
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
    return "\n".join(lines)


def format_verify_result(response: dict) -> str:
    if response["status"] != "ok":
        return f"Error: {response['message']}"
    return (
        f"Verification for '{response['spec']}':\n"
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


class MockBackend:
    """Simple mock that queries a few test functions then predicts zeros."""
    def __init__(self):
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

    true_coeffs = np.concatenate([
        pde.a.coeffs, pde.b.coeffs, pde.c.coeffs, pde.f.coeffs
    ])

    errors, queries = [], []
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
            continue
        try:
            f_rec, _, _, _ = np.linalg.lstsq(G_src, rhs, rcond=None)
            A_lhs = np.hstack([G_diff, G_adv, G_react])
            theta, _, _, _ = np.linalg.lstsq(A_lhs, rhs, rcond=None)
            recovered = np.concatenate([theta, f_rec])
            errors.append(float(np.max(np.abs(recovered - true_coeffs))))
        except Exception:
            errors.append(float("inf"))

    finite = [e for e in errors if np.isfinite(e)]
    auc = float(np.trapezoid(finite)) / max(len(finite), 1) if finite else float("inf")
    return {
        "queries": queries, "errors": errors, "auc": auc,
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
- You may NOT PREDICT in the same response as a QUERY.
"""


def run_probe_session(session: ProbeSession, backend, verbose: bool = True,
                      max_turns: int = 100, enhanced: bool = False) -> dict:
    full_system = session.system_prompt() + protocol
    if enhanced:
        initial_message = (
            "Begin. Important background: the weak form equation is\n\n"
            "  ∫a(x)u'(x)φ'(x)dx + ∫b(x)u'(x)φ(x)dx + ∫c(x)u(x)φ(x)dx = ∫f(x)φ(x)dx\n\n"
            "The first term (diffusion) involves u' and φ'. The second term (advection) "
            "involves u' and φ. The third term (reaction) involves u and φ.\n\n"
            "Each test function φ produces one equation with three unknown contributions "
            "on the left. If one contribution dominates (say the diffusion weight is 100x "
            "larger than the reaction weight), then that equation primarily constrains "
            "the diffusion coefficient a(x) and tells you almost nothing about c(x). "
            "An ill-conditioned system arises when all your equations are dominated by "
            "the same term.\n\n"
            "Use COMPUTE: term_integral to check the three weights for a test function "
            "before spending a query on it. Look for test functions that give balanced "
            "weights across the three terms, or that emphasize a term that your existing "
            "equations have not yet constrained well."
        )
    else:
        initial_message = "Begin."
    messages = [{"role": "user", "content": initial_message}]
    turn = 0
    done = False

    if verbose:
        print("=" * 70)
        print("PROBE SESSION START")
        print("=" * 70)
        print(f"Difficulty: {session.difficulty}, n_basis: {session.basis.n_basis}, "
              f"budget: {session.max_queries}")
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

        # Pass 1: free COMPUTE actions
        for act in actions:
            if act["action"] == "compute":
                cmd = act["command"]
                if cmd.startswith("eval_solution"):
                    args = cmd[len("eval_solution"):].strip()
                    resp = session.eval_solution(args)
                    result_text = format_eval_solution(resp)
                elif cmd == "solve":
                    resp = session.solve()
                    result_text = format_solve_result(resp)
                elif cmd.startswith("verify"):
                    args = cmd[len("verify"):].strip()
                    if not args:
                        result_text = "Usage: COMPUTE: verify <expression>"
                    else:
                        resp = session.verify(args)
                        result_text = format_verify_result(resp)
                elif cmd.startswith("term_integral"):
                    parts = cmd[len("term_integral"):].strip().split(None, 1)
                    if len(parts) < 2:
                        result_text = "Usage: COMPUTE: term_integral <diffusion|advection|reaction> <expression>"
                    else:
                        resp = session.term_integral(parts[0].lower(), parts[1])
                        result_text = format_term_integral(resp)
                else:
                    result_text = f"Unknown command: '{cmd}'. Available: eval_solution, solve, verify, term_integral"
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
                    if verbose:
                        print(f"  → {result_text}\n")
                else:
                    msg = "One QUERY per turn. Review the result above, then submit your next action."
                    result_parts.append(msg)
            elif act["action"] == "predict":
                if query_executed:
                    msg = "You may not PREDICT in the same response as a QUERY. Submit PREDICT alone."
                    result_parts.append(msg)
                else:
                    prediction = {k: act[k] for k in ["a", "b", "c", "f"]}
                    result = session.submit_prediction(prediction)
                    if result["status"] == "scored":
                        score_text = format_score_result(result)
                        result_parts.append(score_text)
                        done = True
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
            messages.append({"role": "user", "content": "\n\n".join(result_parts)})

        if session.queries_used >= session.max_queries and not session.prediction_submitted:
            messages.append({
                "role": "user",
                "content": f"Budget exhausted ({session.max_queries} queries). Submit PREDICT: now."
            })

    output = {
        "turns": turn,
        "queries_used": session.queries_used,
        "prediction_submitted": session.prediction_submitted,
        "messages": messages,
    }
    if session.final_score:
        output["score"] = session.final_score.to_dict()

    # Extract prediction from last assistant message
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            for act in reversed(parse_probe_response(msg["content"])):
                if act["action"] == "predict":
                    output["prediction"] = {k: act[k] for k in ["a", "b", "c", "f"]}
                    break
        if "prediction" in output:
            break

    output["efficiency"] = compute_efficiency_curve(session)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Probe-only PDE identification benchmark")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard", "extreme"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=30)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--save-transcript", type=str, default=None)
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Save results plot to file (default: probe_<difficulty>_<seed>.png)")
    parser.add_argument("--enhanced", action="store_true",
                        help="Add strategic guidance to the initial prompt")
    args = parser.parse_args()

    session = ProbeSession.from_difficulty(args.difficulty, args.seed, args.max_queries)
    backend = MockBackend() if args.mock else AnthropicBackend(model=args.model)

    result = run_probe_session(session, backend, verbose=True,
                               max_turns=args.max_turns, enhanced=args.enhanced)

    if args.save_transcript and "messages" in result:
        with open(args.save_transcript, "w") as f:
            for msg in result["messages"]:
                f.write(f"[{msg['role'].upper()}]\n{msg['content']}\n\n")
        print(f"Transcript saved to {args.save_transcript}")

    # Generate results plot
    if session.prediction_submitted and "prediction" in result:
        plot_path = args.save_plot
        if plot_path is None:
            plot_path = f"probe_{args.difficulty}_{args.seed}.png"
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
