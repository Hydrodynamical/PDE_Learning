"""
Benchmark end-to-end demo.

1. Generate a suite of tasks (easy/medium/hard)
2. Show what an LLM would receive
3. Run an "oracle" solver (least-squares coefficient recovery)
4. Score the oracle predictions
5. Save the suite to JSON
"""

import numpy as np
import json

from basis import LegendreBasis
from pde import solve_elliptic
from test_functions import standard_library
from weak_form import assemble_linear_system
from benchmark import (
    generate_task,
    generate_suite,
    score_prediction,
    score_suite,
    suite_summary,
    save_suite,
)

OUT = "/Users/jkmiller/PDE_Learning"


def oracle_solve(task) -> dict:
    """
    Oracle solver: uses the weak-form linear system to recover coefficients.
    This is the best-case baseline — it knows the math and has access to
    the full solution object.

    An LLM would need to figure out this approach from the data.
    """
    sol = task._solution
    n_basis = task.basis_info["n_basis"]
    basis = LegendreBasis(n_basis=n_basis, domain=(0.0, 1.0))
    test_fns = standard_library()

    system = assemble_linear_system(sol, basis, test_fns)
    A = system.full_matrix()  # (n_tests, 4*n_basis)

    # The system is A @ [a, b, c, f] = 0
    # Use SVD null space to find the solution (up to scale)
    # But we need to fix the scale. One approach: solve the full
    # system A @ theta = 0 with a normalization constraint.
    #
    # Alternatively, if we treat this as two separate problems:
    #   - Recover f from the rhs responses (∫ f φ dx = known)
    #   - Then recover a, b, c given f

    # Step 1: Recover f(x) from ∫ f(x) φ_k(x) dx
    # We know ∫ f φ_k dx for each test function. If f = Σ f_j ψ_j, then:
    #   Σ_j f_j ∫ ψ_j φ_k dx = ∫ f φ_k dx
    # This is: G_source @ f_coeffs = rhs_observed
    rhs_observed = np.array([
        task.precomputed_responses[tf.spec]["integral_f_phi"]
        for tf in test_fns
    ])

    G_src = system.G_source  # (n_tests, n_basis)
    f_coeffs, _, _, _ = np.linalg.lstsq(G_src, rhs_observed, rcond=None)

    # Step 2: Recover a, b, c given f
    A_lhs, A_rhs = system.split_matrix()
    rhs_vec = A_rhs @ f_coeffs
    theta_lhs, _, _, _ = np.linalg.lstsq(A_lhs, rhs_vec, rcond=None)

    n_b = n_basis
    return {
        "a": theta_lhs[:n_b].tolist(),
        "b": theta_lhs[n_b:2*n_b].tolist(),
        "c": theta_lhs[2*n_b:3*n_b].tolist(),
        "f": f_coeffs.tolist(),
    }


def main():
    print("=" * 70)
    print("BENCHMARK SUITE DEMO")
    print("=" * 70)

    # --- 1. Generate a single task and show the LLM prompt ---
    print("\n--- Sample task (medium difficulty) ---\n")
    task = generate_task("demo_001", difficulty="medium", seed=42)
    print(f"PDE: {task.pde_description}\n")

    llm_prompt = task.to_llm_prompt()
    # Show first 40 lines
    prompt_lines = llm_prompt.split("\n")
    for line in prompt_lines[:40]:
        print(f"  {line}")
    if len(prompt_lines) > 40:
        print(f"  ... ({len(prompt_lines) - 40} more lines)")
    print()

    # --- 2. Oracle solve this task ---
    prediction = oracle_solve(task)
    print("--- Oracle prediction ---")
    for name in ["a", "b", "c", "f"]:
        true = task.ground_truth[name]
        pred = prediction[name]
        print(f"  {name}: true={[round(v, 4) for v in true]}, "
              f"pred={[round(v, 4) for v in pred]}")

    basis = LegendreBasis(n_basis=task.basis_info["n_basis"], domain=(0.0, 1.0))
    score = score_prediction(task, prediction, basis)
    print(f"\n{score.summary()}\n")

    # --- 3. Generate full suite ---
    print("--- Generating benchmark suite ---")
    suite = generate_suite(n_easy=3, n_medium=4, n_hard=3, base_seed=2026)
    print(f"  Generated {len(suite)} tasks: "
          f"{sum(1 for t in suite if t.difficulty == 'easy')} easy, "
          f"{sum(1 for t in suite if t.difficulty == 'medium')} medium, "
          f"{sum(1 for t in suite if t.difficulty == 'hard')} hard")

    # --- 4. Oracle solve all tasks ---
    print("\n--- Running oracle solver on all tasks ---")
    predictions = {}
    for task in suite:
        predictions[task.task_id] = oracle_solve(task)

    scores = score_suite(suite, predictions)
    print(f"\n{suite_summary(scores)}")

    # --- 5. Save suite ---
    suite_path = f"{OUT}/benchmark_suite.json"
    save_suite(suite, suite_path)
    print(f"\nSaved benchmark suite to {suite_path}")

    # Show file size
    with open(suite_path) as f:
        data = f.read()
    print(f"  File size: {len(data) / 1024:.1f} KB")
    print(f"  {len(suite)} tasks")

    print("\nDone.")


if __name__ == "__main__":
    main()
