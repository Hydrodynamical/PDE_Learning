#!/usr/bin/env python3
"""Run predefined, offline query controls on the 30 archived PDE instances.

    python scripts/control_baselines.py --archive /tmp/pde-runs-v1.zip

The archive SHA-256 is checked before reading it. No network or model calls are
made. Query lists depend only on the public budget and the control RNG seed;
they never depend on PDE coefficients, solutions, errors, or archived queries.
All strategies use the unchanged ProbeSession.query/solve implementation.

These are new runs on reconstructed instances, not historical model runs. The
generator is checked against all archived coefficient vectors. The original
finite-difference solution is checked against a banded implementation of the
same equations; the latter makes the grid-refinement experiment affordable.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import mean, median
import sys
import time

import numpy as np
from scipy.linalg import solve_banded


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basis import LegendreBasis
from benchmark import DIFFICULTY_CONFIGS
from main_loop import ProbeSession
from pde import EllipticSolution, make_random_elliptic_pde, solve_elliptic
from scripts.reproduce_paper_figures import load_and_validate


COEFFICIENTS = ("a", "b", "c", "f")
DIFFICULTIES = ("easy", "medium", "hard")
STRATEGIES = ("fixed_sine", "boundary_legendre", "random_mixed")
DEFAULT_CONTROL_SEED = 1729


def polynomial_expression(degree: int) -> str:
    """x(1-x) P_degree(2x-1), factored to avoid monomial cancellation.

    The roots and leading coefficient are properties of the public Legendre
    family, independent of the coefficient basis of an individual hidden PDE.
    """
    if degree < 0:
        raise ValueError("Polynomial degree must be nonnegative.")
    roots = np.real_if_close(np.polynomial.legendre.Legendre.basis(degree).roots())
    if np.iscomplexobj(roots):
        raise ValueError("Legendre roots unexpectedly have nonzero imaginary parts.")
    leading = math.comb(2 * degree, degree) / 2**degree
    factors = "".join(f"*(2*x-1-({root:.17g}))" for root in roots)
    return f"{leading:.17g}*x*(1-x){factors}"


def query_list(strategy: str, budget: int, replicate: int = 0,
               control_seed: int = DEFAULT_CONTROL_SEED) -> list[str]:
    """Generate a complete query policy before any session is constructed.

    Random policies use separate SeedSequence streams by budget and replicate.
    Families are sampled uniformly: sine, boundary-weighted Legendre,
    exponential, Gaussian, and boundary-weighted sine. Discrete functions are
    sampled without replacement within each family; continuous parameters make
    exponential/Gaussian repeats probability zero. All streams are reused across
    the ten instances of a difficulty, so seeds are paired when aggregating.
    """
    if budget < 1 or replicate < 0:
        raise ValueError("Budget must be positive and replicate nonnegative.")
    if strategy == "fixed_sine":
        return [f"sin({k}*pi*x)" for k in range(1, budget + 1)]
    if strategy == "boundary_legendre":
        return [polynomial_expression(k) for k in range(budget)]
    if strategy != "random_mixed":
        raise ValueError(f"Unknown strategy: {strategy}")
    rng = np.random.default_rng(np.random.SeedSequence([control_seed, budget, replicate]))
    discrete = {name: list(rng.permutation(budget)) for name in ("sine", "polynomial", "weighted_sine")}
    result = []
    for _ in range(budget):
        family = ("sine", "polynomial", "exponential", "gaussian", "weighted_sine")[int(rng.integers(5))]
        if family == "sine":
            result.append(f"sin({discrete[family].pop() + 1}*pi*x)")
        elif family == "polynomial":
            result.append(polynomial_expression(discrete[family].pop()))
        elif family == "exponential":
            rate = float(rng.uniform(-8, 8))
            result.append(f"x*(1-x)*exp(({rate:.17g})*(x-0.5))")
        elif family == "gaussian":
            center = float(rng.uniform(0.05, 0.95))
            precision = float(np.exp(rng.uniform(np.log(8), np.log(128))))
            result.append(f"x*(1-x)*exp(-({precision:.17g})*(x-({center:.17g}))**2)")
        else:
            result.append(f"x*(1-x)*sin({discrete[family].pop() + 1}*pi*x)")
    return result


def reconstruct_instance(run: dict):
    """Regenerate from public configuration; hidden truth only audits equality."""
    config = run["config"]
    settings = DIFFICULTY_CONFIGS[config["difficulty"]]
    basis = LegendreBasis(settings["n_basis"], domain=(0.0, 1.0))
    pde = make_random_elliptic_pde(
        basis, rng=np.random.default_rng(config["seed"]),
        a_min=settings["a_min"], sparsity=settings["sparsity"], scale=settings["scale"],
    )
    exact = all(np.array_equal(getattr(pde, name).coeffs,
                               run["ground_truth"][name + "_coeffs"])
                for name in COEFFICIENTS)
    if not exact:
        raise ValueError("Current generator does not exactly regenerate archived coefficients; cannot silently substitute instances.")
    return basis, pde


def solve_banded_equivalent(pde, n_grid: int) -> EllipticSolution:
    """Solve the exact second-order difference equations used in pde.py.

    This changes only the linear algebra storage/solver, not stencil, data,
    differentiation, quadrature, rounding, or the identification solver.
    """
    if n_grid < 3:
        raise ValueError("At least three grid points are required.")
    x = np.linspace(*pde.domain, n_grid)
    h = x[1] - x[0]
    a_half = pde.a((x[:-1] + x[1:]) / 2)
    b, c, f = pde.b(x)[1:-1], pde.c(x)[1:-1], pde.f(x)[1:-1]
    lower = -a_half[:-1] / h**2 - b / (2 * h)
    diagonal = (a_half[:-1] + a_half[1:]) / h**2 + c
    upper = -a_half[1:] / h**2 + b / (2 * h)
    bands = np.zeros((3, n_grid - 2))
    bands[0, 1:] = upper[:-1]
    bands[1] = diagonal
    bands[2, :-1] = lower[1:]
    rhs = f.copy()
    rhs[0] -= lower[0] * pde.bc_left
    rhs[-1] -= upper[-1] * pde.bc_right
    u = np.empty(n_grid)
    u[0], u[-1] = pde.bc_left, pde.bc_right
    u[1:-1] = solve_banded((1, 1), bands, rhs, check_finite=True)
    residual = (lower * u[:-2] + diagonal * u[1:-1] + upper * u[2:]) - f
    return EllipticSolution(pde=pde, x=x, u=u, residual=residual)


def design_diagnostics(history: list[dict]) -> dict:
    """Joint LHS sensitivity, unlike separate block condition numbers."""
    matrices = [np.asarray([row[key] for row in history])
                for key in ("_G_diff", "_G_adv", "_G_react")]
    matrix = np.hstack(matrices)

    def summarize(array):
        singular = np.linalg.svd(array, compute_uv=False)
        tolerance = max(array.shape) * np.finfo(float).eps * singular[0]
        rank = int(np.sum(singular > tolerance))
        return {
            "singular_values": singular.tolist(), "rank": rank,
            "n_columns": int(array.shape[1]), "rank_tolerance": float(tolerance),
            "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
            "smallest_singular_value": float(singular[-1]),
        }

    norms = np.linalg.norm(matrix, axis=0)
    return {"raw": summarize(matrix),
            "column_normalized": summarize(matrix / np.where(norms > 0, norms, 1)),
            "column_l2_norms": norms.tolist()}


def evaluate_policy(session: ProbeSession, specs: list[str], truth: dict) -> list[dict]:
    """Use existing query/solve; hidden truth enters only post-solve scoring."""
    checkpoints = []
    first_solve = 4 * session.basis.n_basis
    for spec in specs:
        response = session.query(spec)
        if response["status"] != "ok":
            raise ValueError(f"Control query failed: {spec}: {response}")
        if session.queries_used < first_solve:
            continue
        result = session.solve()
        if result["status"] != "ok":
            raise ValueError(f"Control solve failed: {result}")
        errors = {name: float(np.max(np.abs(np.asarray(session.last_solve_coeffs[name])
                                            - truth[name + "_coeffs"])))
                  for name in COEFFICIENTS}
        displayed_errors = {name: float(np.max(np.abs(np.asarray(result[name + "_coeffs"])
                                                      - truth[name + "_coeffs"])))
                            for name in COEFFICIENTS}
        checkpoints.append({
            "queries": session.queries_used,
            "coefficient_errors": {**errors, "total": sum(errors.values())},
            "displayed_8dp_coefficient_errors": {**displayed_errors, "total": sum(displayed_errors.values())},
            "coefficient_estimates": {name: list(session.last_solve_coeffs[name]) for name in COEFFICIENTS},
            "joint_design": design_diagnostics(session.history),
        })
    return checkpoints


def error_description(values):
    values = [float(v) for v in values]
    return {"n": len(values), "mean": mean(values), "median": median(values),
            "min": min(values), "max": max(values),
            "success_e_lt_0.01": sum(v < 0.01 for v in values),
            "success_e_lt_0.1": sum(v < 0.1 for v in values)}


def aggregate(records: list[dict]) -> list[dict]:
    """Keep instance and randomization variation distinct in the output.

    Deterministic policies have one run per PDE. Random policies report pooled
    runs descriptively, plus each replicate's ten-instance summary and each
    instance's randomized success fraction. No pooled run is called a new PDE.
    """
    groups = defaultdict(list)
    for record in records:
        groups[(record["difficulty"], record["strategy"], record["grid_factor"])].append(record)
    output = []
    for (difficulty, strategy, factor), group in sorted(groups.items()):
        checkpoints = []
        for index in range(len(group[0]["checkpoints"])):
            rows = [record["checkpoints"][index] for record in group]
            checkpoints.append({"queries": rows[0]["queries"],
                                "total_error": error_description(row["coefficient_errors"]["total"] for row in rows),
                                "joint_rank_deficient_runs": sum(row["joint_design"]["raw"]["rank"]
                                                                 < row["joint_design"]["raw"]["n_columns"] for row in rows)})
        per_replicate = []
        for replicate in sorted({record["replicate"] for record in group}):
            subset = [record for record in group if record["replicate"] == replicate]
            per_replicate.append({"replicate": replicate,
                                  "minimum_queries": error_description(r["checkpoints"][0]["coefficient_errors"]["total"] for r in subset),
                                  "full_budget": error_description(r["checkpoints"][-1]["coefficient_errors"]["total"] for r in subset)})
        per_instance = []
        for seed in sorted({record["seed"] for record in group}):
            subset = [record for record in group if record["seed"] == seed]
            per_instance.append({"seed": seed,
                                 "full_budget": error_description(r["checkpoints"][-1]["coefficient_errors"]["total"] for r in subset)})
        output.append({"difficulty": difficulty, "strategy": strategy, "grid_factor": factor,
                       "n_instances": len(per_instance), "n_replicates": len(per_replicate),
                       "checkpoints": checkpoints, "per_replicate": per_replicate,
                       "per_instance": per_instance})
    return output


def sensitivity_summary(sensitivities):
    output = []
    for difficulty in DIFFICULTIES:
        for strategy in STRATEGIES:
            subset = [row for row in sensitivities if row["difficulty"] == difficulty and row["strategy"] == strategy]
            if not subset:
                continue
            all_checks = [check for row in subset for check in row["checkpoints"]]
            final_checks = [row["checkpoints"][-1] for row in subset]
            def describe(checks):
                return {"n_comparisons": len(checks),
                        "max_absolute_total_error_change": max(row["absolute_total_error_change"] for row in checks),
                        "threshold_0.01_changes": sum(row["threshold_0.01_changed"] for row in checks),
                        "threshold_0.1_changes": sum(row["threshold_0.1_changed"] for row in checks)}
            output.append({"difficulty": difficulty, "strategy": strategy,
                           "all_checkpoints": describe(all_checks), "full_budget": describe(final_checks)})
    return output


def write_json(path: Path, value):
    """Write readable JSON or deterministic gzip according to the file suffix."""
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def compact_summary(result):
    excluded = {"runs", "query_policies", "native_vs_banded_score_sensitivity"}
    return {key: value for key, value in result.items() if key not in excluded}


def run_controls(archive: Path, manifest: dict, replicates: int, grid_factors: list[int],
                 control_seed: int = DEFAULT_CONTROL_SEED) -> dict:
    if replicates < 1 or not grid_factors or any(factor < 1 for factor in grid_factors):
        raise ValueError("Positive replicate count and grid factors are required.")
    if len(set(grid_factors)) != len(grid_factors):
        raise ValueError("Grid factors must be distinct.")
    runs = load_and_validate(archive, manifest)
    by_instance = {}
    for run in runs:
        by_instance.setdefault((run["config"]["difficulty"], run["config"]["seed"]), run)
    # Policies are fixed before constructing any session or generating its PDE.
    policies = {(strategy, budget, replicate): query_list(strategy, budget, replicate, control_seed)
                for budget in sorted({run["config"]["max_queries"] for run in runs})
                for strategy in STRATEGIES
                for replicate in range(replicates if strategy == "random_mixed" else 1)}
    records, audits, sensitivities = [], [], []
    start_time = time.perf_counter()
    for (difficulty, seed), run in sorted(by_instance.items()):
        basis, pde = reconstruct_instance(run)
        default_grid = DIFFICULTY_CONFIGS[difficulty]["n_grid"]
        native = solve_elliptic(pde, n_grid=default_grid)
        banded = solve_banded_equivalent(pde, default_grid)
        discrepancy = float(np.max(np.abs(native.u - banded.u)))
        if not np.allclose(native.u, banded.u, rtol=2e-8, atol=1e-10):
            raise ValueError("Banded solution failed agreement with the native dense solver.")
        replay = ProbeSession(pde, native, basis, difficulty, seed, max_queries=run["config"]["max_queries"])
        first_query = run["queries"][0]
        replay_response = replay.query(first_query["test_function"])
        rhs_discrepancy = abs(replay_response["integral_f_phi"] - first_query["rhs_value"])
        if rhs_discrepancy != 0:
            raise ValueError("First archived query RHS did not replay exactly.")
        audits.append({"difficulty": difficulty, "seed": seed,
                       "coefficient_regeneration_exact": True,
                       "native_grid_points": default_grid,
                       "native_banded_max_solution_difference": discrepancy,
                       "first_archived_rhs_exact": True})
        budget = run["config"]["max_queries"]
        for factor in grid_factors:
            # Default-resolution controls use the native implementation itself.
            solution = native if factor == 1 else solve_banded_equivalent(pde, (default_grid - 1) * factor + 1)
            for strategy in STRATEGIES:
                for replicate in range(replicates if strategy == "random_mixed" else 1):
                    session = ProbeSession(pde, solution, basis, difficulty, seed, max_queries=budget)
                    checkpoints = evaluate_policy(session, policies[(strategy, budget, replicate)], run["ground_truth"])
                    records.append({"difficulty": difficulty, "seed": seed, "n_basis": basis.n_basis,
                                    "strategy": strategy, "replicate": replicate,
                                    "grid_factor": factor, "n_grid": len(solution.x), "budget": budget,
                                    "checkpoints": checkpoints})
                    if factor == 1:
                        alternative = ProbeSession(pde, banded, basis, difficulty, seed, max_queries=budget)
                        banded_checks = evaluate_policy(alternative, policies[(strategy, budget, replicate)], run["ground_truth"])
                        sensitivities.append({
                            "difficulty": difficulty, "seed": seed, "strategy": strategy,
                            "replicate": replicate,
                            "checkpoints": [{"queries": left["queries"],
                                             "native_total_error": left["coefficient_errors"]["total"],
                                             "banded_total_error": right["coefficient_errors"]["total"],
                                             "absolute_total_error_change": abs(left["coefficient_errors"]["total"] - right["coefficient_errors"]["total"]),
                                             "threshold_0.01_changed": (left["coefficient_errors"]["total"] < .01) != (right["coefficient_errors"]["total"] < .01),
                                             "threshold_0.1_changed": (left["coefficient_errors"]["total"] < .1) != (right["coefficient_errors"]["total"] < .1)}
                                            for left, right in zip(checkpoints, banded_checks)],
                        })
        print(f"Completed {difficulty} seed {seed}: {len(records)} policy/grid runs", file=sys.stderr, flush=True)
    import scipy
    source_paths = ("scripts/control_baselines.py", "main_loop.py", "pde.py", "basis.py", "benchmark.py",
                    "expression_parser.py", "weak_form.py", "test_functions.py", "requirements.txt")
    all_checkpoints = [check for record in records for check in record["checkpoints"]]
    rounding_summary = {
        "n_checkpoints": len(all_checkpoints),
        "max_absolute_total_error_change": max(abs(row["coefficient_errors"]["total"]
                                                   - row["displayed_8dp_coefficient_errors"]["total"]) for row in all_checkpoints),
        "threshold_agreement": {str(threshold): all((row["coefficient_errors"]["total"] < threshold)
                                                    == (row["displayed_8dp_coefficient_errors"]["total"] < threshold)
                                                    for row in all_checkpoints) for threshold in (.01, .1)},
    }
    return {
        "schema_version": 1, "archive_sha256": manifest["archive_sha256"],
        "dataset_url": manifest["dataset_url"],
        "experiment_type": "new offline controls on exactly regenerated archived PDE instances",
        "historical_replay_limitation": "Exact ground-truth regeneration and first-query RHS replay do not establish identical historical solution grids or numerical implementations for all runs; historical logs lack per-run implementation hashes.",
        "protocol": {
            "control_seed": control_seed, "random_replicates": replicates, "grid_factors": grid_factors,
            "unique_instances": len(by_instance), "query_and_identification_implementation": "unchanged ProbeSession.query and ProbeSession.solve",
            "minimum_solve_queries": "4N, enforced by the existing interface; fitted LHS has 3N columns and forcing is fitted separately",
            "scoring": "Sum over a,b,c,f of maximum absolute coefficient error, using unrounded internal estimates, matching final PREDICT scoring; strict E < threshold. Displayed 8dp scores, matching intermediate historical error curves, are also saved and threshold agreement audited.",
            "selection_information": "Only public query budget and independent control RNG. No solution, coefficient, oracle-response, error, rank, or condition information enters selection.",
            "random_sampling": "Uniform family choice among sine, boundary-weighted Legendre, exponential, localized Gaussian, boundary-weighted sine. Sine degrees 1..budget and polynomial degrees 0..budget-1 sampled without replacement within each family. Exponential rates uniform[-8,8]; Gaussian centers uniform[.05,.95], log precision uniform[log8,log128].",
            "scaling": "Use expressions as generated, with x(1-x) envelope for all families except pure sine; no post-hoc row normalization or tuning.",
            "grid_refinement": "Refine forward-solution grid and every query integral together; (n_grid-1)*factor+1 points. This is not a noise robustness test.",
            "forward_solver": "Native dense solve at original grid; independently assembled equivalent tridiagonal solve_banded at refined grids, validated against native dense at original grid on all 30 instances.",
            "aggregation": "Pooled randomized runs are descriptive. Per-replicate summaries and per-instance success fractions preserve distinct PDE and randomization units; 10 random replicates do not increase the 30 independent PDE instances.",
            "design_svd": "Joint [G_diff G_adv G_react] and independently column-normalized version; rank tolerance max(shape)*eps*sigma_max. Full singular spectra are saved.",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "elapsed_seconds": time.perf_counter() - start_time,
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths},
        "regeneration_audit": audits,
        "display_rounding_score_sensitivity": rounding_summary,
        "native_vs_banded_score_sensitivity_summary": sensitivity_summary(sensitivities),
        "native_vs_banded_score_sensitivity": sensitivities,
        "query_policies": [{"strategy": strategy, "budget": budget, "replicate": replicate, "queries": specs}
                           for (strategy, budget, replicate), specs in sorted(policies.items())],
        "summary": aggregate(records), "runs": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "paper_artifacts/data_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "paper_artifacts/control_baselines.json.gz",
                        help="Full output, compressed deterministically when the filename ends .gz")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "paper_artifacts/control_baselines_summary.json")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--grid-factors", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--control-seed", type=int, default=DEFAULT_CONTROL_SEED)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    result = run_controls(args.archive, manifest, args.replicates, args.grid_factors, args.control_seed)
    write_json(args.output, result)
    write_json(args.summary_output, compact_summary(result))
    print(f"Saved {len(result['runs'])} new control runs to {args.output}")


if __name__ == "__main__":
    main()
