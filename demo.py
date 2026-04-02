"""
Demo: end-to-end PDE generation, solving, and diagnostic visualization.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from basis import LegendreBasis, CoefficientFunction, make_random_coefficient
from pde import EllipticPDE, solve_elliptic, make_random_elliptic_pde
from test_functions import parse_test_function, standard_library
from diagnostics import (
    plot_dashboard,
    plot_test_function_library,
    plot_test_function,
    solution_to_table,
)


def main():
    rng = np.random.default_rng(seed=42)

    # --- 1. Set up the basis ---
    basis = LegendreBasis(n_basis=4, domain=(0.0, 1.0))
    print(f"Basis: {basis}")
    print(f"Labels: {basis.labels()}")
    print()

    # --- 2. Generate a random PDE ---
    pde = make_random_elliptic_pde(basis, rng=rng, a_min=0.5, sparsity=0.3, scale=1.5)
    print(pde.describe())
    print()

    # --- 3. Solve ---
    sol = solve_elliptic(pde, n_grid=501)
    print(f"Solved on {sol.n_grid} grid points")
    print(f"Max residual: {sol.max_residual():.2e}")
    print(f"Energy norm:  {sol.energy_norm():.4f}")
    print()

    # --- 4. Test functions: parse from strings (LLM interface) ---
    specs = [
        "sin(1*pi*x)",
        "sin(3*pi*x)",
        "bump(0.2,0.8)",
        "bump(0.1,0.4)",
        "bump(0.6,0.9)",
        "poly_bubble(2)",
        "hat(0.1,0.5,0.9)",
    ]
    test_fns = []
    for s in specs:
        tf = parse_test_function(s)
        test_fns.append(tf)
        print(f"Parsed: '{s}' → {tf.description}, BCs OK: {tf.satisfies_bcs()}")

    print()

    # --- 5. Generate diagnostic plots ---

    # Dashboard
    fig_dash = plot_dashboard(sol, test_fns)
    fig_dash.savefig("/Users/jkmiller/PDE_Learning/dashboard.png",
                     dpi=150, bbox_inches="tight")
    print("Saved dashboard.png")

    # Test function library
    lib = standard_library()
    fig_lib = plot_test_function_library(lib)
    fig_lib.savefig("/Users/jkmiller/PDE_Learning/test_fn_library.png",
                    dpi=150, bbox_inches="tight")
    print("Saved test_fn_library.png")

    # Individual test function detail
    tf_detail = parse_test_function("bump(0.2,0.8)")
    fig_tf = plot_test_function(tf_detail)
    fig_tf.savefig("/Users/jkmiller/PDE_Learning/test_fn_detail.png",
                   dpi=150, bbox_inches="tight")
    print("Saved test_fn_detail.png")

    # --- 6. Show data export (what an LLM would see) ---
    print("\n--- Solution table (first 10 rows, stride=50) ---")
    table = solution_to_table(sol, stride=50)
    for line in table.split('\n')[:11]:
        print(line)

    # --- 7. Show PDE ground truth (what the benchmark scorer sees) ---
    print("\n--- Ground truth ---")
    gt = pde.to_dict()
    for key in ["a", "b", "c", "f"]:
        d = gt[key]
        print(f"  {key}(x): coeffs = {d['coeffs']}")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
