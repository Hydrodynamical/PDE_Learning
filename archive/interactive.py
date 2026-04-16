"""
Interactive session for LLM-driven PDE identification.

The LLM interacts with this environment by:
    1. Reading the system prompt (describes the problem, rules, available tools)
    2. Submitting test function strings (arbitrary math expressions)
    3. Receiving weak-form responses (or error messages)
    4. Eventually submitting a prediction of the PDE coefficients

The session tracks everything: query history, budget, timing.

Usage:
    session = InteractiveSession.from_difficulty("medium", seed=42)
    print(session.system_prompt())

    # LLM submits a test function
    response = session.query("x*(1-x)*sin(pi*x)")
    # → {"status": "ok", "spec": "...", "integral_f_phi": ..., "integral_u_phi": ..., ...}

    # Or gets an error
    response = session.query("sin(x)")
    # → {"status": "error", "message": "Boundary condition violated: ..."}

    # LLM submits prediction
    score = session.submit_prediction({"a": [...], "b": [...], "c": [...], "f": [...]})
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time
import json
import numpy as np

from basis import LegendreBasis, CoefficientFunction
from pde import EllipticPDE, EllipticSolution, solve_elliptic, make_random_elliptic_pde
from weak_form import compute_weak_form, _composite_simpson
from expression_parser import make_test_function_from_string, validate_expression
from benchmark import score_prediction, ScoreResult, DIFFICULTY_CONFIGS, BenchmarkTask


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class QueryRecord:
    """One query in the session history."""
    query_number: int
    spec: str
    status: str          # "ok" or "error"
    response: dict       # the full response dict
    timestamp: float     # seconds since session start


@dataclass
class InteractiveSession:
    """
    The environment an LLM interacts with to identify a PDE.
    """
    # Problem setup
    pde: EllipticPDE
    solution: EllipticSolution
    basis: LegendreBasis
    difficulty: str
    seed: int

    # Budget
    max_queries: int = 50
    queries_used: int = 0

    # History
    history: list[QueryRecord] = field(default_factory=list)
    start_time: float = 0.0

    # State
    prediction_submitted: bool = False
    final_score: Optional[ScoreResult] = None
    decompose_enabled: bool = True

    # What the LLM sees about the solution
    solution_data: dict = field(default_factory=dict)

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
    def from_difficulty(
        cls,
        difficulty: str = "medium",
        seed: int = 42,
        max_queries: int = 50,
    ) -> InteractiveSession:
        """Create a session with a random PDE at the specified difficulty."""
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

        return cls(
            pde=pde,
            solution=sol,
            basis=basis,
            difficulty=difficulty,
            seed=seed,
            max_queries=max_queries,
        )

    # ----- System prompt -----

    def system_prompt(self) -> str:
        """
        The prompt that initializes the LLM for this session.
        """
        n_b = self.basis.n_basis
        x_str = ", ".join(f"{v:.6f}" for v in self.solution_data["x"])
        u_str = ", ".join(f"{v:.6f}" for v in self.solution_data["u"])

        return f"""You are a mathematician tasked with identifying an unknown elliptic PDE.

## The PDE

The solution u(x) satisfies:

    -(a(x) u'(x))' + b(x) u'(x) + c(x) u(x) = f(x)    on [0, 1]
    u(0) = 0,  u(1) = 0

Each coefficient function a(x), b(x), c(x), f(x) is a smooth function that can
be represented by {n_b} parameters. There are {4 * n_b} unknown parameters total.

## Solution data

x = [{x_str}]
u = [{u_str}]

## Available commands

### QUERY (costs 1 from budget)
Submit a test function and receive ∫f·φ dx and ∫u·φ dx.
    QUERY: <math expression>

{self._decompose_section()}### COMPUTE (free, no budget cost)
Request intermediate calculations:

    COMPUTE: basis_info
        → descriptions of the {n_b} basis functions

    COMPUTE: eval_basis <x1> <x2> ...
        → evaluate all {n_b} basis functions at the given x-values

    COMPUTE: eval_solution <x1> <x2> ...
        → interpolate u(x) at the given x-values

    COMPUTE: solve
        → solves the linear system from your queries and returns recovered coefficients
          with a ready-to-paste PREDICT block. Free, no budget cost.

### Rules for test functions (QUERY and DECOMPOSE)
    - Must satisfy φ(0) = 0 and φ(1) = 0 (boundary conditions)
    - Must be a valid math expression using: x, pi, e, sin, cos, exp, log, sqrt, etc.
    - Examples: "x*(1-x)*sin(pi*x)", "sin(2*pi*x)", "x**2*(1-x)**2*cos(3*pi*x)"

## Scoring

Your score depends on both the accuracy of your final prediction AND the efficiency
of your queries. Every query costs from your limited budget. Uninformative queries
waste budget and hurt your score.

## Budget

You have {self.max_queries} queries. Use them wisely.
Queries remaining: {self.max_queries - self.queries_used}
(COMPUTE commands are free and unlimited.)

## To submit your answer

When ready, provide your prediction as:
    a_coeffs = [p_0, p_1, ..., p_{n_b-1}]
    b_coeffs = [p_0, p_1, ..., p_{n_b-1}]
    c_coeffs = [p_0, p_1, ..., p_{n_b-1}]
    f_coeffs = [p_0, p_1, ..., p_{n_b-1}]
"""

    # ----- System prompt helpers -----

    def _decompose_section(self) -> str:
        if not self.decompose_enabled:
            return ""
        n_b = self.basis.n_basis
        return f"""### DECOMPOSE (costs 1 from budget)
Submit a test function and receive the FULL per-basis-function weak-form integrals.
For each basis function P_j(x), you get:
    G_diff[j]  = ∫ P_j(x) u'(x) φ'(x) dx     (diffusion term integrals)
    G_adv[j]   = ∫ P_j(x) u'(x) φ(x) dx      (advection term integrals)
    G_react[j] = ∫ P_j(x) u(x)  φ(x) dx      (reaction term integrals)
    G_src[j]   = ∫ P_j(x)       φ(x) dx      (source term integrals)

These are exactly the rows of the linear system:
    Σ_j a_j G_diff[j] + Σ_j b_j G_adv[j] + Σ_j c_j G_react[j] = Σ_j f_j G_src[j]

So each DECOMPOSE call gives you one equation in {4 * n_b} unknowns.
With enough equations, you can solve the linear system.

    DECOMPOSE: <math expression>

"""

    # ----- Query interface -----

    def query(self, spec: str) -> dict:
        """
        Submit a test function string and get back the weak-form response.

        Returns a dict with either:
            {"status": "ok", "spec": ..., "integral_f_phi": ..., "integral_u_phi": ...,
             "query_number": ..., "queries_remaining": ...}
        or:
            {"status": "error", "message": ..., "query_number": ..., "queries_remaining": ...}

        Errors do NOT count against the budget.
        """
        if self.prediction_submitted:
            return {
                "status": "error",
                "message": "Session is complete — prediction already submitted.",
                "query_number": self.queries_used,
                "queries_remaining": 0,
            }

        # Try to parse and validate
        tf, error = make_test_function_from_string(spec, domain=self.pde.domain)

        if error is not None:
            # Errors don't count against budget
            response = {
                "status": "error",
                "message": error,
                "query_number": self.queries_used,
                "queries_remaining": self.max_queries - self.queries_used,
            }
            self.history.append(QueryRecord(
                query_number=self.queries_used,
                spec=spec,
                status="error",
                response=response,
                timestamp=time.time() - self.start_time,
            ))
            return response

        # Check budget (only for valid queries)
        if self.queries_used >= self.max_queries:
            return {
                "status": "error",
                "message": f"Query budget exhausted ({self.max_queries} queries used).",
                "query_number": self.queries_used,
                "queries_remaining": 0,
            }

        # Compute weak form
        result = compute_weak_form(self.solution, tf)
        self.queries_used += 1

        # Compute and store G matrices internally (not shown to model)
        # This ensures COMPUTE: solve uses exact-consistent matrix rows
        x_q = self.solution.x
        u_q = self.solution.u
        u_x_q = self.solution.u_x()
        dx = x_q[1] - x_q[0]
        phi_q = tf(x_q)
        dphi_q = tf.derivative(x_q, 1)
        psi_q = self.basis.evaluate(x_q)
        n_b = self.basis.n_basis
        I = lambda y: _composite_simpson(y, dx)

        G_diff  = [I(psi_q[j] * u_x_q * dphi_q) for j in range(n_b)]
        G_adv   = [I(psi_q[j] * u_x_q * phi_q)  for j in range(n_b)]
        G_react = [I(psi_q[j] * u_q   * phi_q)  for j in range(n_b)]
        G_src   = [I(psi_q[j]          * phi_q)  for j in range(n_b)]

        response = {
            "status": "ok",
            "type": "query",
            "spec": spec,
            "integral_f_phi": round(result.rhs, 12),
            "integral_u_phi": round(result.u_phi, 12),
            "query_number": self.queries_used,
            "queries_remaining": self.max_queries - self.queries_used,
            # Internal fields for COMPUTE: solve (not shown to model)
            "_G_diff":  G_diff,
            "_G_adv":   G_adv,
            "_G_react": G_react,
            "_G_src":   G_src,
        }

        self.history.append(QueryRecord(
            query_number=self.queries_used,
            spec=spec,
            status="ok",
            response=response,
            timestamp=time.time() - self.start_time,
        ))

        return response

    # ----- Decompose interface (per-basis weak-form integrals) -----

    def decompose(self, spec: str) -> dict:
        """
        Submit a test function and get back the per-basis-function weak-form integrals.

        Costs 1 query. Returns the full row of the linear system:
            G_diff[j]  = ∫ P_j u' φ' dx
            G_adv[j]   = ∫ P_j u' φ dx
            G_react[j] = ∫ P_j u  φ dx
            G_src[j]   = ∫ P_j    φ dx
        """
        if self.prediction_submitted:
            return {
                "status": "error",
                "message": "Session is complete — prediction already submitted.",
                "query_number": self.queries_used,
                "queries_remaining": 0,
            }

        # Parse and validate
        tf, error = make_test_function_from_string(spec, domain=self.pde.domain)

        if error is not None:
            response = {
                "status": "error",
                "message": error,
                "query_number": self.queries_used,
                "queries_remaining": self.max_queries - self.queries_used,
            }
            self.history.append(QueryRecord(
                query_number=self.queries_used,
                spec=f"[DECOMPOSE] {spec}",
                status="error",
                response=response,
                timestamp=time.time() - self.start_time,
            ))
            return response

        # Check budget
        if self.queries_used >= self.max_queries:
            return {
                "status": "error",
                "message": f"Query budget exhausted ({self.max_queries} queries used).",
                "query_number": self.queries_used,
                "queries_remaining": 0,
            }

        # Compute per-basis integrals
        x_q = self.solution.x
        u_q = self.solution.u
        u_x_q = self.solution.u_x()
        dx = x_q[1] - x_q[0]

        phi_q = tf(x_q)
        dphi_q = tf.derivative(x_q, 1)

        psi_q = self.basis.evaluate(x_q)  # (n_basis, N)
        n_b = self.basis.n_basis

        I = lambda y: _composite_simpson(y, dx)

        G_diff = [round(I(psi_q[j] * u_x_q * dphi_q), 12) for j in range(n_b)]
        G_adv = [round(I(psi_q[j] * u_x_q * phi_q), 12) for j in range(n_b)]
        G_react = [round(I(psi_q[j] * u_q * phi_q), 12) for j in range(n_b)]
        G_src = [round(I(psi_q[j] * phi_q), 12) for j in range(n_b)]

        # Also compute the RHS (∫ f φ dx) — needed by COMPUTE: solve
        assert tf is not None
        wf_result = compute_weak_form(self.solution, tf)
        integral_f_phi = round(wf_result.rhs, 12)

        self.queries_used += 1

        response = {
            "status": "ok",
            "type": "decompose",
            "spec": spec,
            "G_diff": G_diff,
            "G_adv": G_adv,
            "G_react": G_react,
            "G_src": G_src,
            "integral_f_phi": integral_f_phi,
            "query_number": self.queries_used,
            "queries_remaining": self.max_queries - self.queries_used,
        }

        self.history.append(QueryRecord(
            query_number=self.queries_used,
            spec=f"[DECOMPOSE] {spec}",
            status="ok",
            response=response,
            timestamp=time.time() - self.start_time,
        ))

        return response

    # ----- Free compute interface -----

    def compute(self, command: str) -> dict:
        """
        Free computation — does not cost any queries.

        Commands:
            "basis_info"                → basis function descriptions
            "eval_basis 0.1 0.3 0.7"   → evaluate all basis functions at given points
            "eval_solution 0.1 0.3 0.7" → interpolate u(x) at given points
        """
        parts = command.strip().split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "basis_info":
            return self._compute_basis_info()
        elif cmd == "eval_basis":
            return self._compute_eval_basis(args)
        elif cmd == "eval_solution":
            return self._compute_eval_solution(args)
        elif cmd == "solve":
            return self._compute_solve()
        else:
            return {
                "status": "error",
                "message": (
                    f"Unknown COMPUTE command: '{cmd}'. "
                    f"Available: basis_info, eval_basis, eval_solution, solve"
                ),
            }

    def _compute_basis_info(self) -> dict:
        n_b = self.basis.n_basis
        labels = self.basis.labels()
        info = []
        for j in range(n_b):
            info.append(f"P_{j}(x) = {labels[j]}  (Legendre polynomial of degree {j}, shifted to [0,1])")
        # Also provide values at a few reference points
        x_ref = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        vals = self.basis.evaluate(x_ref)
        ref_table = "x\t" + "\t".join(f"P_{j}" for j in range(n_b)) + "\n"
        for i, xi in enumerate(x_ref):
            ref_table += f"{xi:.2f}\t" + "\t".join(f"{vals[j][i]:.6f}" for j in range(n_b)) + "\n"
        return {
            "status": "ok",
            "type": "basis_info",
            "n_basis": n_b,
            "descriptions": info,
            "reference_values": ref_table,
        }

    def _compute_eval_basis(self, args: str) -> dict:
        try:
            x_vals = np.array([float(v.strip().rstrip(',')) for v in args.split() if v.strip()])
        except ValueError:
            return {"status": "error", "message": f"Cannot parse x-values from: '{args}'"}
        if len(x_vals) == 0:
            return {"status": "error", "message": "No x-values provided. Usage: COMPUTE: eval_basis 0.1 0.3 0.5"}
        vals = self.basis.evaluate(x_vals)
        n_b = self.basis.n_basis
        table = "x\t" + "\t".join(f"P_{j}" for j in range(n_b)) + "\n"
        for i, xi in enumerate(x_vals):
            table += f"{xi:.6f}\t" + "\t".join(f"{vals[j][i]:.10f}" for j in range(n_b)) + "\n"
        return {"status": "ok", "type": "eval_basis", "table": table}

    def _compute_eval_solution(self, args: str) -> dict:
        try:
            x_vals = np.array([float(v.strip().rstrip(',')) for v in args.split() if v.strip()])
        except ValueError:
            return {"status": "error", "message": f"Cannot parse x-values from: '{args}'"}
        if len(x_vals) == 0:
            return {"status": "error", "message": "No x-values provided. Usage: COMPUTE: eval_solution 0.1 0.3 0.5"}
        u_vals = np.interp(x_vals, self.solution.x, self.solution.u)
        ux_vals = np.interp(x_vals, self.solution.x, self.solution.u_x())
        table = "x\tu(x)\tu'(x)\n"
        for i, xi in enumerate(x_vals):
            table += f"{xi:.6f}\t{u_vals[i]:.10f}\t{ux_vals[i]:.10f}\n"
        return {"status": "ok", "type": "eval_solution", "table": table}

    def _compute_solve(self) -> dict:
        """
        Solve the linear system. Uses DECOMPOSE history if available, otherwise
        falls back to building the system from QUERY history.
        Free — no budget cost.
        """
        # Check for DECOMPOSE history first
        decomp_records = [
            r for r in self.history
            if r.status == "ok" and r.response.get("type") == "decompose"
        ]
        if not decomp_records:
            return self._compute_solve_from_queries()

        n_b = self.basis.n_basis
        n_unknowns = 4 * n_b
        n_rows = len(decomp_records)

        if n_rows < n_unknowns:
            return {
                "status": "error",
                "type": "solve",
                "message": (
                    f"System is underdetermined: {n_rows} equations for {n_unknowns} unknowns. "
                    f"Submit {n_unknowns - n_rows} more DECOMPOSE queries."
                ),
                "n_rows": n_rows,
                "n_unknowns": n_unknowns,
            }

        # Assemble matrices from stored response dicts
        G_diff = np.array([r.response["G_diff"] for r in decomp_records])    # (n_rows, n_b)
        G_adv = np.array([r.response["G_adv"] for r in decomp_records])
        G_react = np.array([r.response["G_react"] for r in decomp_records])
        G_src = np.array([r.response["G_src"] for r in decomp_records])
        rhs_vec = np.array([r.response["integral_f_phi"] for r in decomp_records])  # (n_rows,)

        # Step 1: recover f from G_src @ f_coeffs = rhs_vec
        f_rec, _, _, _ = np.linalg.lstsq(G_src, rhs_vec, rcond=None)

        # Step 2: recover a, b, c from [G_diff | G_adv | G_react] @ [a,b,c] = rhs_vec
        A_lhs = np.hstack([G_diff, G_adv, G_react])
        theta, _, _, _ = np.linalg.lstsq(A_lhs, rhs_vec, rcond=None)
        a_rec = theta[:n_b]
        b_rec = theta[n_b:2 * n_b]
        c_rec = theta[2 * n_b:3 * n_b]

        # Condition number of the full system
        A_full = np.hstack([G_diff, G_adv, G_react, -G_src])
        sv_full = np.linalg.svd(A_full, compute_uv=False)
        cond = float(sv_full[0] / sv_full[-1]) if sv_full[-1] > 0 else float("inf")

        def _fmt(arr):
            return "[" + ", ".join(f"{v:.6g}" for v in arr) + "]"

        predict_block = (
            "PREDICT:\n"
            f"a_coeffs = {_fmt(a_rec)}\n"
            f"b_coeffs = {_fmt(b_rec)}\n"
            f"c_coeffs = {_fmt(c_rec)}\n"
            f"f_coeffs = {_fmt(f_rec)}"
        )

        return {
            "status": "ok",
            "type": "solve",
            "n_rows": n_rows,
            "n_unknowns": n_unknowns,
            "a_coeffs": a_rec.tolist(),
            "b_coeffs": b_rec.tolist(),
            "c_coeffs": c_rec.tolist(),
            "f_coeffs": f_rec.tolist(),
            "f_residual_norm": float(np.linalg.norm(rhs_vec - G_src @ f_rec)),
            "lhs_residual_norm": float(np.linalg.norm(rhs_vec - A_lhs @ theta)),
            "condition_number": cond,
            "predict_block": predict_block,
        }

    def _compute_solve_from_queries(self) -> dict:
        """
        Build and solve the linear system from QUERY history.
        Used when DECOMPOSE is disabled or no DECOMPOSE history exists.
        """
        n_b = self.basis.n_basis
        n_unknowns = 4 * n_b

        # Collect stored G matrices from QUERY history
        seen_specs = set()
        rows_G_diff = []
        rows_G_adv = []
        rows_G_react = []
        rows_G_src = []
        rhs_values = []

        for r in self.history:
            if r.status == "ok" and "integral_f_phi" in r.response and "_G_diff" in r.response:
                spec = r.response.get("spec", "")
                if spec not in seen_specs:
                    seen_specs.add(spec)
                    rows_G_diff.append(r.response["_G_diff"])
                    rows_G_adv.append(r.response["_G_adv"])
                    rows_G_react.append(r.response["_G_react"])
                    rows_G_src.append(r.response["_G_src"])
                    rhs_values.append(r.response["integral_f_phi"])

        n_rows = len(rhs_values)

        if n_rows < n_unknowns:
            return {
                "status": "error",
                "type": "solve",
                "message": (
                    f"Not enough queries yet: {n_rows} unique test functions, "
                    f"need at least {n_unknowns}."
                ),
                "n_rows": n_rows,
                "n_unknowns": n_unknowns,
            }

        G_diff  = np.array(rows_G_diff)
        G_adv   = np.array(rows_G_adv)
        G_react = np.array(rows_G_react)
        G_src   = np.array(rows_G_src)
        rhs_vec = np.array(rhs_values)

        # Solve (same logic as DECOMPOSE-based solve)
        f_rec, _, _, _ = np.linalg.lstsq(G_src, rhs_vec, rcond=None)
        A_lhs = np.hstack([G_diff, G_adv, G_react])
        theta, _, _, _ = np.linalg.lstsq(A_lhs, rhs_vec, rcond=None)
        a_rec = theta[:n_b]
        b_rec = theta[n_b:2 * n_b]
        c_rec = theta[2 * n_b:3 * n_b]

        A_full = np.hstack([G_diff, G_adv, G_react, -G_src])
        sv_full = np.linalg.svd(A_full, compute_uv=False)
        cond = float(sv_full[0] / sv_full[-1]) if sv_full[-1] > 0 else float("inf")

        def _fmt(arr):
            return "[" + ", ".join(f"{v:.6g}" for v in arr) + "]"

        predict_block = (
            "PREDICT:\n"
            f"a_coeffs = {_fmt(a_rec)}\n"
            f"b_coeffs = {_fmt(b_rec)}\n"
            f"c_coeffs = {_fmt(c_rec)}\n"
            f"f_coeffs = {_fmt(f_rec)}"
        )

        return {
            "status": "ok",
            "type": "solve",
            "n_rows": n_rows,
            "n_unknowns": n_unknowns,
            "a_coeffs": a_rec.tolist(),
            "b_coeffs": b_rec.tolist(),
            "c_coeffs": c_rec.tolist(),
            "f_coeffs": f_rec.tolist(),
            "f_residual_norm": float(np.linalg.norm(rhs_vec - G_src @ f_rec)),
            "lhs_residual_norm": float(np.linalg.norm(rhs_vec - A_lhs @ theta)),
            "condition_number": cond,
            "predict_block": predict_block,
        }

    # ----- Prediction submission -----

    def submit_prediction(self, prediction: dict) -> dict:
        """
        Submit final prediction and get scored.

        prediction: {"a": [...], "b": [...], "c": [...], "f": [...]}

        Returns a detailed score report.
        """
        if self.prediction_submitted:
            return {"status": "error", "message": "Prediction already submitted."}

        # Validate prediction format
        n_b = self.basis.n_basis
        for key in ["a", "b", "c", "f"]:
            if key not in prediction:
                return {"status": "error", "message": f"Missing key '{key}' in prediction."}
            coeffs = prediction[key]
            if not isinstance(coeffs, (list, np.ndarray)):
                return {"status": "error", "message": f"'{key}' must be a list of {n_b} numbers."}
            if len(coeffs) != n_b:
                return {"status": "error",
                        "message": f"'{key}' has {len(coeffs)} coefficients, expected {n_b}."}

        self.prediction_submitted = True

        # Build a BenchmarkTask-like object for scoring
        ground_truth = {
            "a": self.pde.a.coeffs.tolist(),
            "b": self.pde.b.coeffs.tolist(),
            "c": self.pde.c.coeffs.tolist(),
            "f": self.pde.f.coeffs.tolist(),
        }

        task = BenchmarkTask(
            task_id=f"interactive_{self.difficulty}_{self.seed}",
            difficulty=self.difficulty,
            prompt="",
            solution_data=self.solution_data,
            basis_info={
                "type": "LegendreBasis",
                "n_basis": n_b,
                "domain": [0.0, 1.0],
            },
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
            "queries_used": self.queries_used,
            "pde_description": self.pde.describe(),
        }

    # ----- Session info -----

    def status(self) -> dict:
        """Current session status."""
        return {
            "difficulty": self.difficulty,
            "n_basis": self.basis.n_basis,
            "n_unknowns": 4 * self.basis.n_basis,
            "queries_used": self.queries_used,
            "queries_remaining": self.max_queries - self.queries_used,
            "n_errors": sum(1 for r in self.history if r.status == "error"),
            "prediction_submitted": self.prediction_submitted,
            "elapsed_seconds": round(time.time() - self.start_time, 1),
        }

    def query_history_table(self) -> str:
        """Format query history as a readable table."""
        lines = [f"{'#':<4} {'Status':<7} {'Type':<11} {'Spec':<40} {'Info':>20}"]
        lines.append("-" * 85)
        for r in self.history:
            spec = r.spec
            if r.status == "ok":
                resp = r.response
                if resp.get("type") == "decompose":
                    info = f"4x{len(resp.get('G_diff', []))} matrix row"
                    spec = spec.replace("[DECOMPOSE] ", "")
                else:
                    ifp = resp.get('integral_f_phi', 0)
                    info = f"∫fφ={ifp:+.6f}"
            else:
                info = r.response["message"][:20]
            lines.append(f"{r.query_number:<4} {r.status:<7} "
                         f"{'DECOMP' if '[DECOMPOSE]' in r.spec else 'QUERY':<11} "
                         f"{spec.replace('[DECOMPOSE] ', ''):<40} {info:>20}")
        return "\n".join(lines)

    def to_transcript(self) -> str:
        """Full session transcript for analysis."""
        parts = [
            "=" * 70,
            "SESSION TRANSCRIPT",
            "=" * 70,
            f"Difficulty: {self.difficulty}",
            f"Seed: {self.seed}",
            f"n_basis: {self.basis.n_basis}",
            f"Queries used: {self.queries_used} / {self.max_queries}",
            "",
            self.pde.describe(),
            "",
            "--- Query History ---",
            self.query_history_table(),
        ]
        if self.final_score:
            parts.extend([
                "",
                "--- Final Score ---",
                self.final_score.summary(),
            ])
        return "\n".join(parts)