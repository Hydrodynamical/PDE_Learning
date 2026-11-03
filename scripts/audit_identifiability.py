#!/usr/bin/env python3
"""Offline audit of structural ambiguity, joint design spectra, and precision.

This diagnoses the existing protocol; it neither certifies identifiability of a
sampled continuum problem nor supplies a new model experiment. Saved ground
truth is used only to reconstruct each numerical solution and audit its design.
No API calls are made. Run from the repository root, for example::

  .venv/bin/python scripts/audit_identifiability.py \
      --archive /path/to/pinned-reasoning-v1.zip

The exact polynomial calculation is independent of a numerical rank tolerance.
The replay uses the repository's current, dense finite-difference solver, the
archived query expressions, and the protocol's absolute 12-decimal rounding.
Historical solver/library differences can affect ill-conditioned estimates;
replay discrepancies are therefore reported rather than silently tolerated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basis import CoefficientFunction, LegendreBasis
from benchmark import DIFFICULTY_CONFIGS
from expression_parser import make_test_function_from_string
from pde import EllipticPDE, solve_elliptic
from scripts.reproduce_paper_figures import load_and_validate
from weak_form import _composite_simpson

COEFFICIENTS = ("a", "b", "c", "f")
SCALES = (0.001, 1.0, 1000.0, 1_000_000.0)


def protocol_round(array, decimals=12):
    """Match the protocol's scalar Python round, not np.round's fast path."""
    values = np.asarray(array)
    return np.array([round(float(v), decimals) for v in values.flat]).reshape(values.shape)


def exact_rank(matrix):
    """Gaussian elimination over Q, for small integer proof certificates."""
    rows = [[Fraction(int(v)) for v in row] for row in matrix]
    pivot = 0
    for col in range(len(rows[0])):
        choice = next((i for i in range(pivot, len(rows)) if rows[i][col]), None)
        if choice is None:
            continue
        rows[pivot], rows[choice] = rows[choice], rows[pivot]
        divisor = rows[pivot][col]
        rows[pivot] = [v / divisor for v in rows[pivot]]
        for i in range(pivot + 1, len(rows)):
            multiplier = rows[i][col]
            rows[i] = [v - multiplier * p for v, p in zip(rows[i], rows[pivot])]
        pivot += 1
        if pivot == len(rows):
            break
    return pivot


def polynomial_operator_matrix(n_basis):
    """Integer monomial matrix for -(a u')' + b u' + c u, u=x(1-x).

    Rows encode powers x^0,...,x^(N+1); columns encode a,b,c coefficients.
    The same rank applies in any invertible polynomial basis, including shifted
    Legendre. Surjectivity also follows directly: c*u spans u*P_(N-1), while
    constant a and b supply span{1,1-2*x}, the remainder upon division by u.
    """
    if n_basis < 2:
        raise ValueError("The dimension statement requires N >= 2.")
    matrix = np.zeros((n_basis + 2, 3 * n_basis), dtype=int)
    for j in range(n_basis):
        matrix[j, j] = 2 * (j + 1)
        if j:
            matrix[j - 1, j] = -j
        matrix[j, n_basis + j] = 1
        matrix[j + 1, n_basis + j] = -2
        matrix[j + 1, 2 * n_basis + j] = 1
        matrix[j + 2, 2 * n_basis + j] = -1
    return matrix


def structural_certificate():
    records = []
    for n in range(2, 9):
        matrix = polynomial_operator_matrix(n)
        rank = exact_rank(matrix)
        assert rank == n + 2
        records.append({"n_basis": n, "exact_rank_over_rationals": rank,
                        "exact_nullity": 3 * n - rank,
                        "integer_matrix": matrix.tolist()})
    # The two elementary N=2 directions from the manuscript family.
    directions = [[1, 0, -2, 4, -8, 0], [-1, 2, 4, 0, 0, 0]]
    matrix = polynomial_operator_matrix(2)
    assert all(np.array_equal(matrix @ d, np.zeros(4)) for d in directions)
    return {
        "solution": "u=x(1-x)",
        "coefficient_space": "P_(N-1)^3, any invertible polynomial basis",
        "theorem": "For N>=2 the strong operator map has rank N+2 and nullity 2N-2.",
        "proof": "The image contains u*P_(N-1) through c, and span{1,u'} through constant a,b; polynomial division by u proves these span P_(N+1).",
        "interpretation": "Every null direction preserves u and f for all continuum test functions. Sufficiently small perturbations preserve positive diffusion. For N>=3, base a=1,b=0,c=1,f=2+u also allows an open neighborhood preserving positive reaction.",
        "certificates": records,
        "n2_monomial_null_directions": directions,
    }


def pde_from_truth(truth, n_basis):
    basis = LegendreBasis(n_basis)
    pde = EllipticPDE(**{k: CoefficientFunction(basis, truth[f"{k}_coeffs"])
                         for k in COEFFICIENTS})
    return basis, pde


def design_row(solution, basis, spec):
    tf, error = make_test_function_from_string(spec, domain=solution.pde.domain)
    if error:
        raise ValueError(f"Cannot replay archived expression {spec!r}: {error}")
    x, u, ux = solution.x, solution.u, solution.u_x()
    phi, dphi = tf(x), tf.derivative(x, 1)
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(dphi)):
        raise ValueError(f"Non-finite archived expression: {spec!r}")
    psi = basis.evaluate(x)
    integrate = lambda v: _composite_simpson(v, x[1] - x[0])
    a = [integrate(p * ux * dphi) for p in psi]
    b = [integrate(p * ux * phi) for p in psi]
    c = [integrate(p * u * phi) for p in psi]
    f = [integrate(p * phi) for p in psi]
    rhs = integrate(solution.pde.f(x) * phi)
    return np.array(a + b + c), np.array(f), float(rhs)


def spectra(matrix):
    """Numerical diagnostics only: never interpreted as continuum rank proof."""
    n_cols = matrix.shape[1]
    singular = np.linalg.svd(matrix, compute_uv=False)
    if len(singular) < n_cols:
        singular = np.pad(singular, (0, n_cols - len(singular)))
    scale = singular[0] if singular.size and singular[0] else 1.0
    return {
        "singular_values": singular.tolist(),
        "min_over_max_singular_value": float(singular[-1] / scale),
        "rank_at_numpy_default_tolerance": int(np.sum(singular > max(matrix.shape) * np.finfo(float).eps * scale)),
        "rank_at_relative_tolerance": {str(t): int(np.sum(singular > t * scale))
                                        for t in (1e-8, 1e-10, 1e-12)},
    }


def joint_diagnostics(matrix, rhs, theta):
    norms = np.linalg.norm(matrix, axis=0)
    balanced = matrix / np.where(norms > 0, norms, 1)
    residual = matrix @ theta - rhs
    return {"rows": len(rhs), "columns": matrix.shape[1],
            "raw": spectra(matrix), "unit_column_norm": spectra(balanced),
            "truth_residual_linf": float(np.max(np.abs(residual))),
            "truth_residual_l2_over_rhs_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).tiny))}


def solve_blocks(matrix, source, rhs):
    op = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    forcing = np.linalg.lstsq(source, rhs, rcond=None)[0]
    return np.r_[op, forcing]


def total_error(estimate, truth):
    n = len(truth) // 4
    return float(sum(np.max(np.abs(estimate[i*n:(i+1)*n] - truth[i*n:(i+1)*n]))
                     for i in range(4)))


def load_inputs(archive, manifest, runs_dir=None):
    """Validate the pinned ZIP, without extracting or executing its members.

    Optional previously extracted inputs must match the pinned member bytes,
    including the complete set of run filenames. Computation always consumes
    the validated ZIP content, so directory state cannot alter the audit.
    """
    validated_runs = load_and_validate(archive, manifest)
    with zipfile.ZipFile(archive) as source:
        names = sorted(n for n in source.namelist() if n.startswith("run_") and n.endswith(".json"))
        raw_members = {n: source.read(n) for n in names}
    if runs_dir is not None:
        extracted = {p.name: p.read_bytes() for p in runs_dir.glob("run_*.json")}
        if extracted != raw_members:
            raise ValueError("Extracted run-file names/bytes do not match the pinned archive.")
    return list(zip(names, validated_runs)), {
        n: hashlib.sha256(contents).hexdigest() for n, contents in raw_members.items()
    }


def archive_audit(runs):
    solution_cache, row_cache = {}, {}
    replay, scale_audit, check_counts = [], [], Counter()
    for name, run in runs:
        config, truth = run["config"], run["ground_truth"]
        difficulty, seed, n = config["difficulty"], config["seed"], config["n_basis"]
        key = difficulty, seed
        truth_vector = np.concatenate([truth[f"{k}_coeffs"] for k in COEFFICIENTS])
        if key not in solution_cache:
            basis, pde = pde_from_truth(truth, n)
            grid = DIFFICULTY_CONFIGS[difficulty]["n_grid"]
            solution_cache[key] = basis, solve_elliptic(pde, n_grid=grid), truth_vector
            print(f"Reconstructed {difficulty} seed {seed}, grid {grid}", flush=True)
        basis, solution, saved_truth = solution_cache[key]
        if not np.array_equal(truth_vector, saved_truth):
            raise ValueError("Archived tasks sharing seed/difficulty disagree on truth.")

        def row(spec):
            cache_key = key, spec
            if cache_key not in row_cache:
                row_cache[cache_key] = design_row(solution, basis, spec)
            return row_cache[cache_key]

        rows = [row(q["test_function"]) for q in run["queries"]]
        raw_matrix = np.array([r[0] for r in rows])
        raw_source = np.array([r[1] for r in rows])
        raw_rhs = np.array([r[2] for r in rows])
        matrix, source, rhs = (protocol_round(v) for v in (raw_matrix, raw_source, raw_rhs))
        logged_rhs = np.array([q["rhs_value"] for q in run["queries"]])
        solve_records = []
        for solve in run["solves"]:
            k = solve.get("queries_at_solve") or solve["n_equations"]
            estimate = solve_blocks(matrix[:k], source[:k], rhs[:k])
            logged = np.concatenate([solve["coefficients"][j] for j in COEFFICIENTS])
            solve_records.append({
                "solve_number": solve["solve_number"], "queries_at_solve": k,
                "max_coefficient_difference_from_logged_8dp_estimate": float(np.max(np.abs(protocol_round(estimate, 8) - logged))),
                "replayed_total_error": total_error(protocol_round(estimate, 8), truth_vector),
                "logged_total_error": solve["coeff_errors"]["total"],
                "joint": joint_diagnostics(matrix[:k], rhs[:k], truth_vector[:3*n]),
            })
        replay.append({
            "run_file": name, "model": config["model"], "difficulty": difficulty,
            "seed": seed, "n_basis": n, "n_grid": solution.n_grid,
            "query_rhs_max_difference_from_archive": float(np.max(np.abs(rhs - logged_rhs))),
            "stored_queries": len(rows), "stored_query_budget": config["max_queries"],
            "free_checks": len(run["verifications"]),
            "solves": solve_records,
            "all_stored_queries": joint_diagnostics(matrix, rhs, truth_vector[:3*n]),
        })
        for check in run["verifications"]:
            check_counts["free_checks"] += 1
            previous = [q["test_function"] for q in run["queries"] if q["turn"] < check["turn"]]
            if check["test_function"] not in previous:
                check_counts["expression_absent_from_earlier_stored_queries"] += 1
        # Exactly the same recorded functions and quadrature values, multiplied
        # before protocol rounding. This isolates quantization/row weighting;
        # it intentionally excludes extra expression-parser roundoff at scale.
        scaled_records = []
        for amplitude in SCALES:
            estimate = solve_blocks(protocol_round(amplitude * raw_matrix),
                                    protocol_round(amplitude * raw_source),
                                    protocol_round(amplitude * raw_rhs))
            scaled_records.append({"common_amplitude": amplitude,
                                   "total_error": total_error(protocol_round(estimate, 8), truth_vector)})
        scale_audit.append({"run_file": name, "difficulty": difficulty,
                            "model": config["model"], "seed": seed,
                            "all_stored_queries": scaled_records})
    return replay, scale_audit, dict(check_counts)


def summarize(records, scales, checks):
    groups = []
    for difficulty in ("easy", "medium", "hard"):
        cell = [r for r in records if r["difficulty"] == difficulty]
        s = [r for r in scales if r["difficulty"] == difficulty]
        columns = cell[0]["all_stored_queries"]["columns"]
        groups.append({
            "difficulty": difficulty, "n_runs": len(cell), "operator_columns": columns,
            "all_query_joint_full_rank_at_numpy_tolerance": sum(r["all_stored_queries"]["raw"]["rank_at_numpy_default_tolerance"] == columns for r in cell),
            "all_query_joint_rank_deficient_at_relative_1e8": sum(r["all_stored_queries"]["unit_column_norm"]["rank_at_relative_tolerance"]["1e-08"] < columns for r in cell),
            "median_unit_column_norm_smallest_over_largest_singular_value": float(np.median([r["all_stored_queries"]["unit_column_norm"]["min_over_max_singular_value"] for r in cell])),
            "median_truth_weak_residual_l2_over_rhs_l2": float(np.median([r["all_stored_queries"]["truth_residual_l2_over_rhs_l2"] for r in cell])),
            "runs_with_free_checks": sum(r["free_checks"] > 0 for r in cell),
            "stored_plus_check_response_count_exceeds_stored_budget": sum(r["stored_queries"] + r["free_checks"] > r["stored_query_budget"] for r in cell),
            "median_free_checks": float(np.median([r["free_checks"] for r in cell])),
            "amplitude_sensitivity": [{
                "common_amplitude": amplitude,
                "median_total_error": float(np.median([r["all_stored_queries"][i]["total_error"] for r in s])),
                "successes_at_001": sum(r["all_stored_queries"][i]["total_error"] < .01 for r in s),
                "n": len(s),
            } for i, amplitude in enumerate(SCALES)],
        })
    solves = [s for r in records for s in r["solves"]]
    return {
        "runs": len(records), "unique_reconstructed_solutions": 30,
        "stored_query_rhs_responses": sum(r["stored_queries"] for r in records),
        "max_query_rhs_difference_from_archive": max(r["query_rhs_max_difference_from_archive"] for r in records),
        "solves_replayed": len(solves),
        "max_coefficient_difference_from_logged_estimate": max(s["max_coefficient_difference_from_logged_8dp_estimate"] for s in solves),
        "median_coefficient_difference_from_logged_estimate": float(np.median([s["max_coefficient_difference_from_logged_8dp_estimate"] for s in solves])),
        "free_measurements": checks, "difficulty_groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Unmodified dataset version 1 ZIP; checked against the manifest SHA-256")
    parser.add_argument("--manifest", type=Path, default=ROOT / "paper_artifacts/data_manifest.json")
    parser.add_argument("--runs-dir", type=Path, help="Optional extracted copy to check byte-for-byte against the pinned ZIP; not used as the audit input")
    parser.add_argument("--output", type=Path, default=ROOT / "paper_artifacts/identifiability_audit.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    runs, input_hashes = load_inputs(args.archive, manifest, args.runs_dir)
    records, scales, checks = archive_audit(runs)
    payload = {
        "schema_version": 1,
        "archive_sha256": manifest["archive_sha256"],
        "dataset_url": manifest["dataset_url"],
        "method": {
            "input_validation": "ZIP bytes and complete scored-run corpus validated by reproduce_paper_figures.load_and_validate against the pinned dataset manifest.",
            "solution_reconstruction": "Current repository pde.solve_elliptic from saved ground truth; no fitting to archived estimates.",
            "grids": {d: DIFFICULTY_CONFIGS[d]["n_grid"] for d in ("easy", "medium", "hard")},
            "matrix_quantization": "Each matrix entry and RHS rounded to 12 decimal places, exactly as ProbeSession.query.",
            "column_normalization": "Each nonzero column divided by its Euclidean norm, solely for reported diagnostic spectra.",
            "rank_warning": "Floating-point rank of a discretized matrix cannot prove structural identifiability of the continuum problem.",
            "amplitude_warning": "Sensitivity audit scales precomputed unrounded rows/RHS before 12dp rounding, across all stored rows, then scores coefficient estimates rounded to 8dp. These diagnostic refits differ from the primary final score computed from unrounded internal estimates. It is not a fresh model run, an optimized strategy, or parser-roundoff experiment; archived query sets are held fixed.",
            "free_check_warning": "Expression novelty is literal-string novelty, not proof of independent information. Free check returns actual RHS without storing a solve row.",
            "source_code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                   for name in ("basis.py", "benchmark.py", "pde.py", "main_loop.py", "expression_parser.py", "weak_form.py", "requirements.txt", "scripts/audit_identifiability.py", "scripts/reproduce_paper_figures.py")},
            "input_run_files_sha256": input_hashes,
            "runtime": {"python": platform.python_version(), "python_implementation": platform.python_implementation(),
                        "platform": platform.platform(), "packages": {p: version(p) for p in ("numpy", "scipy", "matplotlib")}},
        },
        "structural": structural_certificate(),
        "summary": summarize(records, scales, checks),
        "replayed_runs": records, "amplitude_sensitivity": scales,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
