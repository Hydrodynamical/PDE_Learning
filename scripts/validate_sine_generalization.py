#!/usr/bin/env python3
"""Fresh-seed validation after selecting fixed sine on the archived tasks.

    python scripts/validate_sine_generalization.py

The fixed plan contains seeds 1001--1100 in each difficulty, including zero
forcing and any failed tasks. No retries, exclusions, policy changes, archive
selection, network access, or model calls are made. Original-resolution native
forward solves and the existing query/identification interface are retained.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
from statistics import NormalDist, median
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basis import LegendreBasis
from benchmark import DIFFICULTY_CONFIGS
from main_loop import ProbeSession
from pde import make_random_elliptic_pde, solve_elliptic
from scripts.control_baselines import evaluate_policy, query_list


DIFFICULTIES = ("easy", "medium", "hard")
FIRST_SEED, LAST_SEED = 1001, 1100


def validation_plan():
    """Return the complete fixed plan, in deterministic execution order."""
    return [{"difficulty": difficulty, "seed": seed}
            for difficulty in DIFFICULTIES
            for seed in range(FIRST_SEED, LAST_SEED + 1)]


def wilson_interval(successes: int, n: int):
    if not 0 <= successes <= n or n < 1:
        raise ValueError("Wilson interval requires 0 <= successes <= positive n.")
    z = NormalDist().inv_cdf(.975)
    p = successes / n
    divisor = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / divisor
    radius = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / divisor
    return [float(max(0, center - radius)), float(min(1, center + radius))]


def concise_checkpoint(checkpoint):
    diagnostics = {}
    for kind in ("raw", "column_normalized"):
        full = checkpoint["joint_design"][kind]
        diagnostics[kind] = {key: value for key, value in full.items() if key != "singular_values"}
        diagnostics[kind]["largest_singular_value"] = full["singular_values"][0]
    return {"queries": checkpoint["queries"],
            "coefficient_errors": checkpoint["coefficient_errors"],
            "displayed_8dp_coefficient_errors": checkpoint["displayed_8dp_coefficient_errors"],
            "coefficient_estimates": checkpoint["coefficient_estimates"],
            "joint_design": diagnostics}


def evaluate_task(task):
    """Retain an outcome and failure stage for every generated instance."""
    difficulty, seed = task["difficulty"], task["seed"]
    settings = DIFFICULTY_CONFIGS[difficulty]
    n_basis, n_grid = settings["n_basis"], settings["n_grid"]
    result = {**task, "status": "pending", "stage": "generation",
              "config": {**settings, "domain": [0, 1], "bc_left": 0, "bc_right": 0,
                         "minimum_queries": 4 * n_basis, "budget": 6 * n_basis},
              "ground_truth": None, "zero_source": None, "zero_solution": None,
              "minimum_queries": None, "full_budget": None, "error": None}
    started = time.perf_counter()
    try:
        basis = LegendreBasis(n_basis, domain=(0.0, 1.0))
        pde = make_random_elliptic_pde(
            basis, rng=np.random.default_rng(seed), a_min=settings["a_min"],
            sparsity=settings["sparsity"], scale=settings["scale"],
        )
        result["ground_truth"] = {name + "_coeffs": getattr(pde, name).coeffs.tolist()
                                  for name in ("a", "b", "c", "f")}
        result["zero_source"] = bool(np.all(pde.f.coeffs == 0))
        result["stage"] = "native_forward_solve"
        solution = solve_elliptic(pde, n_grid=n_grid)
        if not np.all(np.isfinite(solution.u)):
            raise ValueError("Native forward solution contains nonfinite values.")
        result["zero_solution"] = bool(np.all(solution.u == 0))
        result["forward_max_abs_u"] = float(np.max(np.abs(solution.u)))
        result["stage"] = "query_and_identification"
        session = ProbeSession(pde, solution, basis, difficulty, seed, max_queries=6 * n_basis)
        checkpoints = evaluate_policy(session, query_list("fixed_sine", 6 * n_basis), result["ground_truth"])
        result["minimum_queries"] = concise_checkpoint(checkpoints[0])
        result["full_budget"] = concise_checkpoint(checkpoints[-1])
        if not all(np.isfinite(result[key]["coefficient_errors"]["total"])
                   for key in ("minimum_queries", "full_budget")):
            raise ValueError("Coefficient recovery contains a nonfinite score.")
        result["status"], result["stage"] = "ok", "complete"
    except Exception as error:
        result["status"] = "error"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def execute_plan(plan, runner=evaluate_task, progress=False):
    """Never drop or retry a planned task, even on an unexpected runner failure."""
    outcomes = []
    for index, task in enumerate(plan, 1):
        try:
            outcome = runner(task)
        except Exception as error:
            outcome = {**task, "status": "error", "stage": "unexpected_runner_failure",
                       "ground_truth": None, "zero_source": None, "zero_solution": None,
                       "minimum_queries": None, "full_budget": None,
                       "error": {"type": type(error).__name__, "message": str(error)}}
        outcomes.append(outcome)
        if progress and index % 10 == 0:
            print(f"Completed {index}/{len(plan)} fresh tasks: {task['difficulty']} seed {task['seed']}",
                  file=sys.stderr, flush=True)
    return outcomes


def summarize(outcomes):
    summary = []
    for difficulty in DIFFICULTIES:
        rows = [row for row in outcomes if row["difficulty"] == difficulty]
        if not rows:
            continue
        cell = {"difficulty": difficulty, "n_planned_tasks": len(rows),
                "status_counts": dict(Counter(row["status"] for row in rows)),
                "zero_source_tasks": sum(row.get("zero_source") is True for row in rows),
                "zero_solution_tasks": sum(row.get("zero_solution") is True for row in rows),
                "zero_source_seeds": [row["seed"] for row in rows if row.get("zero_source") is True],
                "failed_tasks": [{"seed": row["seed"], "stage": row["stage"], "error": row["error"]}
                                 for row in rows if row["status"] != "ok"]}
        for checkpoint in ("minimum_queries", "full_budget"):
            good = [row for row in rows if row["status"] == "ok"]
            successes = sum(row[checkpoint]["coefficient_errors"]["total"] < .01 for row in good)
            cell[checkpoint] = {
                "successes_e_lt_0.01": successes, "denominator_all_planned_tasks": len(rows),
                "success_fraction": successes / len(rows), "wilson_95_interval": wilson_interval(successes, len(rows)),
                "successes_e_lt_0.1": sum(row[checkpoint]["coefficient_errors"]["total"] < .1 for row in good),
                "median_error_completed_tasks": median(row[checkpoint]["coefficient_errors"]["total"] for row in good) if good else None,
                "failure_seeds_including_errors": [row["seed"] for row in rows if row["status"] != "ok"
                                                   or row[checkpoint]["coefficient_errors"]["total"] >= .01],
                "zero_source_subset_descriptive": {
                    "n": cell["zero_source_tasks"],
                    "successes": sum(row.get("zero_source") is True and row[checkpoint]["coefficient_errors"]["total"] < .01 for row in good)},
                "nonzero_source_subset_descriptive": {
                    "n": sum(row.get("zero_source") is False for row in rows),
                    "successes": sum(row.get("zero_source") is False and row[checkpoint]["coefficient_errors"]["total"] < .01 for row in good)},
            }
        summary.append(cell)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "paper_artifacts/sine_generalization.json")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    plan = validation_plan()
    outcomes = execute_plan(plan, progress=True)
    environment = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                   "machine": platform.machine(), "platform": platform.platform()}
    sources = ("scripts/validate_sine_generalization.py", "scripts/control_baselines.py", "basis.py", "benchmark.py",
               "pde.py", "main_loop.py", "expression_parser.py", "weak_form.py", "test_functions.py", "requirements.txt")
    hashes = {source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest() for source in sources}
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = {
        "schema_version": 1,
        "experiment_type": "fresh-seed validation after design selection",
        "protocol": {
            "strategy": "fixed_sine", "first_seed": FIRST_SEED, "last_seed": LAST_SEED,
            "n_seeds_per_difficulty": 100, "n_planned_tasks": len(plan), "query_expression": "sin(k*pi*x), k=1..6N",
            "selection_timing": "Fixed sine was selected after its success on the 30 archived instances; this is subsequent validation, not a historical held-out evaluation or formal preregistration.",
            "inclusion": "Every seed-difficulty task in the fixed plan, including all-zero forcing, all-zero solutions, generation/solver/identification errors. No retries or exclusions.",
            "environment": "Original-resolution native solve_elliptic and unchanged ProbeSession.query/solve; no banded substitution or grid refinement.",
            "scoring": "Sum of per-field maximum absolute coefficient errors from unrounded internal estimates; E<0.01 primary. Displayed 8dp errors also retained.",
            "units": "100 tasks per difficulty. The same seed labels are reused across difficulties, with related generator RNG streams; no pooled 300-task independent-binomial interval is claimed.",
            "intervals": "Wilson 95% marginal intervals per difficulty over all 100 planned tasks; execution failures count as non-successes. Assumes seed-generated task sampling and conditions on this generator and fixed design.",
            "zero_source_subsets": "Descriptive counts only; zero-source tasks remain in all primary denominators.",
            "provider_calls": 0,
        },
        "environment": environment, "runtime_sha256": digest(environment),
        "source_sha256": hashes, "source_bundle_sha256": digest(hashes),
        "elapsed_seconds": time.perf_counter() - started,
        "plan": plan, "summary": summarize(outcomes), "tasks": outcomes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
