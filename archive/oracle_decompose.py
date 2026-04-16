"""
Oracle strategy using the DECOMPOSE scratchpad.

This script demonstrates what a perfect LLM would do:
    1. COMPUTE: basis_info (free)
    2. DECOMPOSE with n_unknowns well-chosen test functions
    3. Assemble the linear system from the returned matrix rows
    4. Solve via least squares
    5. PREDICT

This gives the best-case baseline for comparing real LLM runs.

Usage:
    python oracle_decompose.py
    python oracle_decompose.py --difficulty hard --seed 42
"""

import argparse
import numpy as np
from interactive import InteractiveSession


def pick_test_functions(n_unknowns: int) -> list[str]:
    """
    Choose a good set of test functions for the linear system.

    Strategy: mix of Fourier sine modes (global) and polynomial
    bubbles (local structure), with some localized Gaussians.
    Aim for ~1.5x the number of unknowns for overdetermination.
    """
    n_target = int(n_unknowns * 1.5)
    specs = []

    # Sine modes: sin(k*pi*x) for k=1..ceil(n_target/3)
    n_sin = max(4, n_target // 3)
    for k in range(1, n_sin + 1):
        specs.append(f"sin({k}*pi*x)")

    # Polynomial bubbles: x*(1-x)*P_j(x)-like
    polys = [
        "x*(1-x)",
        "x*(1-x)*(2*x-1)",
        "x*(1-x)*(6*x**2-6*x+1)",
        "x*(1-x)*(20*x**3-30*x**2+12*x-1)",
        "x**2*(1-x)**2",
        "x**2*(1-x)**2*(2*x-1)",
    ]
    for p in polys[:n_target // 3]:
        specs.append(p)

    # Localized Gaussians: probe different regions
    centers = [0.2, 0.35, 0.5, 0.65, 0.8]
    for c in centers[:n_target // 4]:
        specs.append(f"x*(1-x)*exp(-30*(x-{c})**2)")

    # Product test functions for better conditioning
    extras = [
        "x*(1-x)*sin(pi*x)",
        "x*(1-x)*sin(2*pi*x)",
        "x*(1-x)*cos(2*pi*x)*(2*x-1)",
    ]
    for e in extras:
        if len(specs) < n_target:
            specs.append(e)

    return specs[:n_target]


def run_oracle(difficulty: str = "medium", seed: int = 42, max_queries: int = 50):
    """Run the oracle DECOMPOSE strategy and print results."""

    session = InteractiveSession.from_difficulty(
        difficulty=difficulty, seed=seed, max_queries=max_queries,
    )
    n_b = session.basis.n_basis
    n_unknowns = 4 * n_b

    print(f"Difficulty: {difficulty}, n_basis: {n_b}, n_unknowns: {n_unknowns}")
    print(f"Budget: {max_queries} queries\n")

    # Step 1: Get basis info (free)
    info = session.compute("basis_info")
    print("Step 1: COMPUTE basis_info (free)")
    for desc in info["descriptions"]:
        print(f"  {desc}")
    print()

    # Step 2: Choose test functions
    specs = pick_test_functions(n_unknowns)
    print(f"Step 2: Selected {len(specs)} test functions")
    for s in specs:
        print(f"  {s}")
    print()

    # Step 3: DECOMPOSE each one
    print(f"Step 3: DECOMPOSE all test functions")
    rows_diff = []
    rows_adv = []
    rows_react = []
    rows_src = []

    for spec in specs:
        resp = session.decompose(spec)
        if resp["status"] == "ok":
            rows_diff.append(resp["G_diff"])
            rows_adv.append(resp["G_adv"])
            rows_react.append(resp["G_react"])
            rows_src.append(resp["G_src"])
            print(f"  [{resp['query_number']:2d}] {spec:<45} ok "
                  f"({resp['queries_remaining']} remaining)")
        else:
            print(f"  [--] {spec:<45} ERROR: {resp['message'][:40]}")

    n_rows = len(rows_diff)
    print(f"\n  Collected {n_rows} equations for {n_unknowns} unknowns")

    # Step 4: Assemble and solve the linear system
    print(f"\nStep 4: Solve the linear system")

    G_diff = np.array(rows_diff)    # (n_rows, n_b)
    G_adv = np.array(rows_adv)
    G_react = np.array(rows_react)
    G_src = np.array(rows_src)

    # Full system: [G_diff | G_adv | G_react | -G_src] @ [a, b, c, f] = 0
    # But we need a constraint. Strategy: first recover f from the RHS,
    # then solve for a, b, c.

    # The weak form says: Σ a_j G_diff[j] + Σ b_j G_adv[j] + Σ c_j G_react[j] = Σ f_j G_src[j]
    # For each test function, the RHS = ∫ f φ dx. We can get this from a QUERY.

    # Actually, we can recover f directly from G_src and the DECOMPOSE data.
    # We know that: A_lhs @ [a,b,c] = G_src @ f
    # where A_lhs = [G_diff | G_adv | G_react]
    #
    # But we don't know the RHS without knowing f. However, we can also
    # use QUERY to get ∫fφ dx directly for each test function, and that
    # equals G_src @ f_coeffs. So let's query a few to get ∫fφ.

    # Alternative: use the full null-space approach
    # A_full @ θ = 0  where A_full = [G_diff | G_adv | G_react | -G_src]
    # Find the null space, normalize.

    # Let's try both approaches and see which works better.

    # Approach 1: Query to get ∫fφ for each test function, then solve
    print("  Getting ∫fφ for each test function via QUERY...")
    rhs_vec = []
    for spec in specs[:n_rows]:
        resp = session.query(spec)
        if resp["status"] == "ok":
            rhs_vec.append(resp["integral_f_phi"])

    rhs_vec = np.array(rhs_vec)

    # Now: G_src @ f_coeffs = rhs_vec  →  solve for f
    f_rec, _, _, _ = np.linalg.lstsq(G_src, rhs_vec, rcond=None)
    print(f"  Recovered f_coeffs = {np.round(f_rec, 6).tolist()}")

    # Now: [G_diff | G_adv | G_react] @ [a,b,c] = G_src @ f_rec = rhs_vec
    A_lhs = np.hstack([G_diff, G_adv, G_react])
    theta_lhs, _, _, _ = np.linalg.lstsq(A_lhs, rhs_vec, rcond=None)

    a_rec = theta_lhs[:n_b]
    b_rec = theta_lhs[n_b:2*n_b]
    c_rec = theta_lhs[2*n_b:3*n_b]

    print(f"  Recovered a_coeffs = {np.round(a_rec, 6).tolist()}")
    print(f"  Recovered b_coeffs = {np.round(b_rec, 6).tolist()}")
    print(f"  Recovered c_coeffs = {np.round(c_rec, 6).tolist()}")

    # Step 5: Submit prediction
    prediction = {
        "a": a_rec.tolist(),
        "b": b_rec.tolist(),
        "c": c_rec.tolist(),
        "f": f_rec.tolist(),
    }

    print(f"\nStep 5: Submit prediction")
    result = session.submit_prediction(prediction)
    print(f"\n{result['summary']}")

    print(f"\nGround truth:")
    gt = result["ground_truth"]
    for name in ["a", "b", "c", "f"]:
        print(f"  {name}: {[round(v, 6) for v in gt[name]]}")

    st = session.status()
    print(f"\nTotal queries used: {st['queries_used']} "
          f"(DECOMPOSE: {n_rows}, QUERY: {len(rhs_vec)})")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard", "extreme"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=50)
    args = parser.parse_args()

    print("=" * 70)
    print("ORACLE DECOMPOSE STRATEGY")
    print("=" * 70)
    print()

    result = run_oracle(args.difficulty, args.seed, args.max_queries)

    # Summary
    score = result["score"]
    print(f"\n{'=' * 70}")
    print(f"ORACLE RESULT: total coeff error = {score['total_coeff_error']:.6f}")
    print(f"               total pw error    = {score['total_pointwise_error']:.6f}")


if __name__ == "__main__":
    main()
