"""
Full diagnostic demo for the weak-form PDE benchmark.

Generates:
    1. PDE dashboard (coefficients, solution, residual, test functions)
    2. Weak-form decomposition bar chart
    3. Weak-form residuals
    4. Coefficient recovery comparison (true vs recovered)
    5. Convergence study (error vs grid refinement)
    6. Singular value spectrum of the linear system
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from basis import LegendreBasis
from pde import make_random_elliptic_pde, solve_elliptic
from test_functions import parse_test_function, standard_library
from weak_form import (
    compute_weak_form_battery,
    battery_to_table,
    assemble_linear_system,
)
from diagnostics import (
    plot_dashboard,
    plot_weak_form_battery,
    plot_weak_form_residuals,
    plot_recovery_comparison,
    plot_convergence_study,
    plot_singular_values,
)

OUT = "/Users/jkmiller/PDE_Learning"


def main():
    rng = np.random.default_rng(42)

    # Use 3 basis functions so the system is overdetermined
    # (15 test functions > 12 unknowns = 3 coeffs × 4 terms... wait,
    #  with 3 basis functions: 3×3 = 9 unknowns for a,b,c; 15 equations → ok)
    basis = LegendreBasis(n_basis=3, domain=(0.0, 1.0))
    pde = make_random_elliptic_pde(basis, rng=rng, a_min=0.5, sparsity=0.0, scale=1.0)
    sol = solve_elliptic(pde, n_grid=2001)

    print(pde.describe())
    print(f"\nSolved on {sol.n_grid} points, max residual = {sol.max_residual():.2e}")

    # Test functions
    test_fns = standard_library()

    # --- 1. Dashboard ---
    fig = plot_dashboard(sol, test_fns)
    fig.savefig(f"{OUT}/01_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 01_dashboard.png")

    # --- 2. Weak-form battery ---
    results = compute_weak_form_battery(sol, test_fns)
    print(f"\nWeak-form battery: {len(results)} test functions")
    print(f"  Max residual: {max(r.residual for r in results):.2e}")
    print(f"  Mean residual: {np.mean([r.residual for r in results]):.2e}")

    fig = plot_weak_form_battery(results)
    fig.savefig(f"{OUT}/02_weak_form_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 02_weak_form_decomposition.png")

    fig = plot_weak_form_residuals(results)
    fig.savefig(f"{OUT}/03_weak_form_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 03_weak_form_residuals.png")

    # --- 3. Linear system & coefficient recovery ---
    system = assemble_linear_system(sol, basis, test_fns)
    print(f"\nLinear system: {system.full_matrix().shape}")
    print(f"  Condition number: {system.condition_number():.2e}")

    fig = plot_singular_values(system)
    fig.savefig(f"{OUT}/04_singular_values.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 04_singular_values.png")

    # Recover coefficients
    A_lhs, A_rhs = system.split_matrix()
    rhs_vec = A_rhs @ pde.f.coeffs
    theta_rec, _, _, _ = np.linalg.lstsq(A_lhs, rhs_vec, rcond=None)
    n_b = basis.n_basis

    recovered = {
        "a": theta_rec[:n_b],
        "b": theta_rec[n_b:2*n_b],
        "c": theta_rec[2*n_b:3*n_b],
    }

    print("\nCoefficient recovery:")
    for name in ["a", "b", "c"]:
        true_c = getattr(pde, name).coeffs
        rec_c = recovered[name]
        err = np.max(np.abs(true_c - rec_c))
        print(f"  {name}(x): true={np.round(true_c, 4)}, "
              f"rec={np.round(rec_c, 4)}, err={err:.4f}")

    fig = plot_recovery_comparison(pde, recovered, basis)
    fig.savefig(f"{OUT}/05_recovery_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 05_recovery_comparison.png")

    # --- 4. Convergence study ---
    print("\nRunning convergence study...")
    fig = plot_convergence_study(pde, basis)
    fig.savefig(f"{OUT}/06_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 06_convergence.png")

    print("\nAll diagnostics complete.")


if __name__ == "__main__":
    main()
