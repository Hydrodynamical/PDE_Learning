"""
Tests for basis function framework.
Run: python test_basis.py
"""

import unittest
import numpy as np
from basis import (
    LegendreBasis,
    FourierBasis,
    MonomialBasis,
    CoefficientFunction,
    make_random_coefficient,
)


def finite_diff_derivative(f, x, order=1, h=1e-6):
    if order == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    elif order == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / h**2
    else:
        raise NotImplementedError


class TestMonomialBasis(unittest.TestCase):
    def test_evaluate(self):
        basis = MonomialBasis(n_basis=4, domain=(0, 1))
        x = np.array([0.0, 0.5, 1.0])
        vals = basis.evaluate(x)
        np.testing.assert_allclose(vals[0], [1, 1, 1])
        np.testing.assert_allclose(vals[1], [0, 0.5, 1])
        np.testing.assert_allclose(vals[2], [0, 0.25, 1])
        np.testing.assert_allclose(vals[3], [0, 0.125, 1])

    def test_derivative_order1(self):
        basis = MonomialBasis(n_basis=4, domain=(0, 1))
        x = np.array([0.5, 1.0, 2.0])
        dvals = basis.evaluate_derivative(x, order=1)
        np.testing.assert_allclose(dvals[0], [0, 0, 0])
        np.testing.assert_allclose(dvals[1], [1, 1, 1])
        np.testing.assert_allclose(dvals[2], 2 * x)
        np.testing.assert_allclose(dvals[3], 3 * x**2)

    def test_derivative_order2(self):
        basis = MonomialBasis(n_basis=4, domain=(0, 1))
        x = np.array([0.5, 1.0, 2.0])
        d2vals = basis.evaluate_derivative(x, order=2)
        np.testing.assert_allclose(d2vals[0], [0, 0, 0])
        np.testing.assert_allclose(d2vals[1], [0, 0, 0])
        np.testing.assert_allclose(d2vals[2], [2, 2, 2])
        np.testing.assert_allclose(d2vals[3], 6 * x)


class TestLegendreBasis(unittest.TestCase):
    def test_P0_P1(self):
        basis = LegendreBasis(n_basis=3, domain=(0, 1))
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        vals = basis.evaluate(x)
        np.testing.assert_allclose(vals[0], 1.0)
        np.testing.assert_allclose(vals[1], 2 * x - 1, atol=1e-14)

    def test_derivative_vs_finite_diff(self):
        basis = LegendreBasis(n_basis=5, domain=(0, 1))
        x = np.linspace(0.05, 0.95, 50)

        for order in [1, 2]:
            analytic = basis.evaluate_derivative(x, order=order)
            for j in range(basis.n_basis):
                def psi_j(xx, _j=j):
                    return basis.evaluate(np.atleast_1d(xx))[_j]
                numerical = finite_diff_derivative(psi_j, x, order=order, h=1e-5)
                np.testing.assert_allclose(
                    analytic[j], numerical, atol=1e-4,
                    err_msg=f"Legendre P_{j}, derivative order {order}"
                )

    def test_orthogonality_approx(self):
        basis = LegendreBasis(n_basis=4, domain=(-1, 1))
        x, w = np.polynomial.legendre.leggauss(20)
        vals = basis.evaluate(x)
        gram = vals @ np.diag(w) @ vals.T
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertAlmostEqual(gram[i, j], 0.0, places=10,
                        msg=f"<P_{i}, P_{j}> = {gram[i, j]}")


class TestFourierBasis(unittest.TestCase):
    def test_evaluate_basic(self):
        basis = FourierBasis(n_basis=3, domain=(0, 1))
        x = np.array([0.0, 0.25, 0.5])
        vals = basis.evaluate(x)
        np.testing.assert_allclose(vals[0], 1.0)
        np.testing.assert_allclose(vals[1], np.cos(2 * np.pi * x), atol=1e-14)
        np.testing.assert_allclose(vals[2], np.sin(2 * np.pi * x), atol=1e-14)

    def test_derivative_vs_finite_diff(self):
        basis = FourierBasis(n_basis=5, domain=(0, 1))
        x = np.linspace(0.05, 0.95, 50)

        for order in [1, 2]:
            analytic = basis.evaluate_derivative(x, order=order)
            for j in range(basis.n_basis):
                def psi_j(xx, _j=j):
                    return basis.evaluate(np.atleast_1d(xx))[_j]
                numerical = finite_diff_derivative(psi_j, x, order=order, h=1e-5)
                np.testing.assert_allclose(
                    analytic[j], numerical, atol=1e-3,
                    err_msg=f"Fourier basis {j}, derivative order {order}"
                )


class TestCoefficientFunction(unittest.TestCase):
    def test_evaluation(self):
        basis = MonomialBasis(n_basis=3, domain=(0, 1))
        f = CoefficientFunction(basis, np.array([2.0, 3.0, -1.0]), name="test")
        x = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(f(x), [2.0, 3.25, 4.0])

    def test_derivative(self):
        basis = MonomialBasis(n_basis=3, domain=(0, 1))
        f = CoefficientFunction(basis, np.array([2.0, 3.0, -1.0]))
        x = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(f.derivative(x, order=1), [3.0, 2.0, 1.0])

    def test_random_generation(self):
        basis = LegendreBasis(n_basis=4, domain=(0, 1))
        rng = np.random.default_rng(42)
        f = make_random_coefficient(basis, rng=rng, sparsity=0.5, name="a(x)")
        x = np.linspace(0, 1, 10)
        vals = f(x)
        self.assertEqual(vals.shape, (10,))

    def test_serialization(self):
        basis = LegendreBasis(n_basis=3, domain=(0, 1))
        f = CoefficientFunction(basis, np.array([1.0, -0.5, 0.3]), name="a(x)")
        d = f.to_dict()
        self.assertEqual(d["basis_type"], "LegendreBasis")
        self.assertEqual(d["coeffs"], [1.0, -0.5, 0.3])
        self.assertEqual(d["name"], "a(x)")


class TestCrossBasis(unittest.TestCase):
    def test_legendre_vs_monomial_quadratic(self):
        x = np.linspace(0, 1, 100)
        mono = MonomialBasis(n_basis=3, domain=(0, 1))
        f_mono = CoefficientFunction(mono, np.array([1.0, 2.0, 3.0]))

        leg = LegendreBasis(n_basis=3, domain=(0, 1))
        basis_vals = leg.evaluate(x)
        target = f_mono(x)
        c_leg, _, _, _ = np.linalg.lstsq(basis_vals.T, target, rcond=None)

        f_leg = CoefficientFunction(leg, c_leg)
        np.testing.assert_allclose(f_leg(x), f_mono(x), atol=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
