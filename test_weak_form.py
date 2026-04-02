"""
Tests and demo for the weak-form integrator.

Key checks:
    1. For a correctly solved PDE, lhs ≈ rhs for every test function.
    2. The linear system Aθ = 0 is satisfied by the true coefficients.
    3. Coefficient recovery from weak-form data works.

Run: python test_weak_form.py
"""

import unittest
import numpy as np

from basis import LegendreBasis, CoefficientFunction, constant_coefficient
from pde import EllipticPDE, solve_elliptic, make_random_elliptic_pde
from test_functions import parse_test_function, standard_library
from weak_form import (
    compute_weak_form,
    compute_weak_form_battery,
    battery_to_table,
    weak_form_from_string,
    assemble_linear_system,
)


class TestWeakFormConsistency(unittest.TestCase):
    """The weak form identity lhs = rhs should hold for a correct solution."""

    def setUp(self):
        self.basis = LegendreBasis(n_basis=4, domain=(0.0, 1.0))
        self.rng = np.random.default_rng(123)
        self.pde = make_random_elliptic_pde(
            self.basis, rng=self.rng, a_min=0.5, sparsity=0.3
        )
        self.sol = solve_elliptic(self.pde, n_grid=501)

    def test_single_test_function(self):
        tf = parse_test_function("sin(1*pi*x)")
        result = compute_weak_form(self.sol, tf)
        self.assertAlmostEqual(result.lhs, result.rhs, places=5,
            msg=f"lhs={result.lhs}, rhs={result.rhs}, residual={result.residual}")

    def test_battery_all_consistent(self):
        """Weak-form residuals bounded by O(h⁴) from 4th-order FD derivatives."""
        results = compute_weak_form_battery(self.sol)
        for r in results:
            self.assertLess(r.residual, 5e-4,
                msg=f"Test fn '{r.spec}': residual={r.residual:.2e}")

    def test_bump_localized(self):
        tf = parse_test_function("bump(0.3,0.7)")
        result = compute_weak_form(self.sol, tf)
        self.assertAlmostEqual(result.lhs, result.rhs, places=5)

    def test_string_interface(self):
        response = weak_form_from_string(self.sol, "bump(0.2,0.5)")
        self.assertIn("integral_f_phi", response)
        self.assertIn("integral_u_phi", response)
        self.assertIn("spec", response)
        # Should be finite numbers
        self.assertTrue(np.isfinite(response["integral_f_phi"]))
        self.assertTrue(np.isfinite(response["integral_u_phi"]))


class TestLinearSystem(unittest.TestCase):
    """The assembled linear system should be consistent with true coefficients."""

    def setUp(self):
        self.basis = LegendreBasis(n_basis=3, domain=(0.0, 1.0))
        self.rng = np.random.default_rng(456)
        self.pde = make_random_elliptic_pde(
            self.basis, rng=self.rng, a_min=1.0, sparsity=0.0
        )
        self.sol = solve_elliptic(self.pde, n_grid=2001)
        self.test_fns = standard_library()
        self.system = assemble_linear_system(self.sol, self.basis, self.test_fns)

    def test_true_coefficients_satisfy_system(self):
        """A @ θ_true ≈ 0 for the true coefficient vector."""
        A = self.system.full_matrix()
        # full_matrix = [G_diff | G_adv | G_react | -G_src]
        # so A @ [α, β, γ, f] = G_diff α + G_adv β + G_react γ - G_src f = 0
        theta = np.concatenate([
            self.pde.a.coeffs,
            self.pde.b.coeffs,
            self.pde.c.coeffs,
            self.pde.f.coeffs,
        ])
        residual = A @ theta
        max_res = np.max(np.abs(residual))
        self.assertLess(max_res, 1e-4,
            msg=f"A @ θ_true max residual = {max_res:.2e}")

    def test_split_system_consistency(self):
        """A_lhs @ θ_lhs = A_rhs @ θ_rhs for true coefficients."""
        A_lhs, A_rhs = self.system.split_matrix()
        theta_lhs = np.concatenate([
            self.pde.a.coeffs,
            self.pde.b.coeffs,
            self.pde.c.coeffs,
        ])
        theta_rhs = self.pde.f.coeffs
        lhs = A_lhs @ theta_lhs
        rhs = A_rhs @ theta_rhs
        max_diff = np.max(np.abs(lhs - rhs))
        self.assertLess(max_diff, 1e-4,
            msg=f"Split system max difference = {max_diff:.2e}")

    def test_matrix_dimensions(self):
        n_tests = len(self.test_fns)
        n_b = self.basis.n_basis
        self.assertEqual(self.system.G_diffusion.shape, (n_tests, n_b))
        self.assertEqual(self.system.full_matrix().shape, (n_tests, 4 * n_b))

    def test_singular_values_finite(self):
        sv = self.system.singular_values()
        self.assertTrue(np.all(np.isfinite(sv)))
        self.assertGreater(sv[0], 0)


class TestCoefficientRecovery(unittest.TestCase):
    """Given enough test functions, can we recover the PDE coefficients?"""

    def test_recovery_known_f(self):
        """
        If f(x) is known, recover a, b, c from the weak form.
        A_lhs @ θ_lhs = A_rhs @ f_coeffs  →  solve for θ_lhs.
        """
        basis = LegendreBasis(n_basis=3, domain=(0.0, 1.0))
        rng = np.random.default_rng(789)
        pde = make_random_elliptic_pde(basis, rng=rng, a_min=1.0, sparsity=0.0)
        sol = solve_elliptic(pde, n_grid=2001)  # fine grid for accurate derivatives

        test_fns = standard_library()
        system = assemble_linear_system(sol, basis, test_fns)

        A_lhs, A_rhs = system.split_matrix()
        rhs_vec = A_rhs @ pde.f.coeffs

        # Solve the overdetermined system
        theta_recovered, residuals, rank, sv = np.linalg.lstsq(A_lhs, rhs_vec, rcond=None)

        n_b = basis.n_basis
        a_rec = theta_recovered[:n_b]
        b_rec = theta_recovered[n_b:2*n_b]
        c_rec = theta_recovered[2*n_b:3*n_b]

        # Check recovery (4th-order FD gives much better derivatives)
        np.testing.assert_allclose(a_rec, pde.a.coeffs, atol=0.02,
            err_msg=f"a: true={pde.a.coeffs}, recovered={a_rec}")
        np.testing.assert_allclose(b_rec, pde.b.coeffs, atol=0.02,
            err_msg=f"b: true={pde.b.coeffs}, recovered={b_rec}")
        np.testing.assert_allclose(c_rec, pde.c.coeffs, atol=0.02,
            err_msg=f"c: true={pde.c.coeffs}, recovered={c_rec}")


def demo():
    """Run a full demo showing weak-form diagnostics."""
    print("=" * 70)
    print("WEAK-FORM INTEGRATOR DEMO")
    print("=" * 70)

    # Setup
    basis = LegendreBasis(n_basis=4, domain=(0.0, 1.0))
    rng = np.random.default_rng(42)
    pde = make_random_elliptic_pde(basis, rng=rng, a_min=0.5, sparsity=0.3)
    sol = solve_elliptic(pde, n_grid=501)

    print(f"\n{pde.describe()}\n")

    # --- 1. Single test function ---
    print("--- Single test function: sin(1*pi*x) ---")
    r = compute_weak_form(sol, parse_test_function("sin(1*pi*x)"))
    print(f"  Diffusion ∫ a u' φ' dx = {r.diffusion:+.8f}")
    print(f"  Advection ∫ b u' φ dx  = {r.advection:+.8f}")
    print(f"  Reaction  ∫ c u  φ dx  = {r.reaction:+.8f}")
    print(f"  LHS total              = {r.lhs:+.8f}")
    print(f"  RHS  ∫ f φ dx          = {r.rhs:+.8f}")
    print(f"  |LHS - RHS|            = {r.residual:.2e}")
    print()

    # --- 2. Battery (standard library) ---
    print("--- Battery: standard library ---")
    results = compute_weak_form_battery(sol)
    print(f"  {len(results)} test functions evaluated")
    residuals = [r.residual for r in results]
    print(f"  Max weak-form residual: {max(residuals):.2e}")
    print(f"  Mean weak-form residual: {np.mean(residuals):.2e}")
    print()

    # --- 3. Full diagnostic table ---
    print("--- Full diagnostic table ---")
    table = battery_to_table(results, mode="full")
    for line in table.split('\n')[:8]:
        print(f"  {line}")
    print("  ...")
    print()

    # --- 4. Blind table (what the LLM sees) ---
    print("--- Blind table (LLM view) ---")
    blind_table = battery_to_table(results, mode="blind")
    for line in blind_table.split('\n')[:8]:
        print(f"  {line}")
    print("  ...")
    print()

    # --- 5. LLM string interface ---
    print("--- LLM string interface ---")
    for spec in ["bump(0.2,0.8)", "sin(3*pi*x)", "bump(0.45,0.55)"]:
        resp = weak_form_from_string(sol, spec)
        print(f"  '{spec}' → ∫fφ = {resp['integral_f_phi']:+.8f}, "
              f"∫uφ = {resp['integral_u_phi']:+.8f}")
    print()

    # --- 6. Linear system and coefficient recovery ---
    print("--- Linear system assembly ---")
    test_fns = standard_library()
    system = assemble_linear_system(sol, basis, test_fns)
    print(f"  System matrix: {system.full_matrix().shape} "
          f"({system.n_tests} equations, {system.n_unknowns} unknowns)")
    print(f"  Condition number: {system.condition_number():.1f}")
    sv = system.singular_values()
    print(f"  Singular values: {sv[:6].round(4)} ...")
    print()

    # Verify true coefficients satisfy the system
    A = system.full_matrix()
    theta_true = np.concatenate([
        pde.a.coeffs, pde.b.coeffs, pde.c.coeffs, pde.f.coeffs
    ])
    res = A @ theta_true
    print(f"  A @ θ_true: max |residual| = {np.max(np.abs(res)):.2e}")

    # Recover coefficients (assuming f is known)
    A_lhs, A_rhs = system.split_matrix()
    rhs_vec = A_rhs @ pde.f.coeffs
    theta_rec, _, _, _ = np.linalg.lstsq(A_lhs, rhs_vec, rcond=None)
    n_b = basis.n_basis

    print(f"\n--- Coefficient recovery (f known) ---")
    for name, true_c, rec_c in [
        ("a", pde.a.coeffs, theta_rec[:n_b]),
        ("b", pde.b.coeffs, theta_rec[n_b:2*n_b]),
        ("c", pde.c.coeffs, theta_rec[2*n_b:3*n_b]),
    ]:
        err = np.max(np.abs(true_c - rec_c))
        print(f"  {name}(x): true = {np.round(true_c, 4)}, "
              f"recovered = {np.round(rec_c, 4)}, max err = {err:.2e}")

    print("\nDone.")


if __name__ == "__main__":
    print("\n>>> Running unit tests...\n")
    # Run tests first
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestWeakFormConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestLinearSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestCoefficientRecovery))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n\n")
        demo()
