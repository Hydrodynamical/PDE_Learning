"""
Weak-form integrator for elliptic PDE benchmark.

The PDE:  -(a(x) u')' + b(x) u' + c(x) u = f(x)

Multiply by test function φ(x) with φ(x₀) = φ(x₁) = 0, integrate over [x₀, x₁],
and integrate by parts on the diffusion term:

    ∫ a u' φ' dx  +  ∫ b u' φ dx  +  ∫ c u φ dx  =  ∫ f φ dx

Each test function φ produces one scalar equation. The LLM's job is to figure out
a, b, c, f from a collection of these scalar responses.

This module provides:
    - WeakFormResult: per-test-function breakdown of all integral terms
    - compute_weak_form: integrates one test function against a solution
    - compute_weak_form_battery: runs the full library, returns a table
    - weak_form_from_string: the LLM-facing API (string in → scalar out)
    - WeakFormLinearSystem: assembles the linear system for coefficient recovery
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from basis import BasisFamily, CoefficientFunction
from pde import EllipticPDE, EllipticSolution
from test_functions import TestFunction, parse_test_function, standard_library


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

def _composite_simpson(y: np.ndarray, dx: float) -> float:
    """
    Composite Simpson's rule for uniformly spaced data.

    Falls back to trapezoidal if n is even (number of intervals is odd).
    For our purposes n_grid is always odd (201, 501, etc.), so Simpson
    applies cleanly.
    """
    n = len(y)
    if n < 3:
        # Trapezoidal fallback
        return float(np.trapz(y, dx=dx))
    if (n - 1) % 2 == 0:
        # n-1 intervals, even number → Simpson works
        return float(dx / 3.0 * (y[0] + y[-1]
                                  + 4.0 * np.sum(y[1:-1:2])
                                  + 2.0 * np.sum(y[2:-2:2])))
    else:
        # Odd number of intervals: Simpson on first n-1 points, trap on last
        s = dx / 3.0 * (y[0] + y[-2]
                         + 4.0 * np.sum(y[1:-2:2])
                         + 2.0 * np.sum(y[2:-3:2]))
        s += 0.5 * dx * (y[-2] + y[-1])
        return float(s)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WeakFormResult:
    """
    Result of integrating one test function against a PDE solution.

    Attributes
    ----------
    spec : str
        Test function string specification.
    diffusion : float
        ∫ a(x) u'(x) φ'(x) dx
    advection : float
        ∫ b(x) u'(x) φ(x) dx
    reaction : float
        ∫ c(x) u(x) φ(x) dx
    lhs : float
        diffusion + advection + reaction (= left-hand side of weak form)
    rhs : float
        ∫ f(x) φ(x) dx (= right-hand side)
    residual : float
        |lhs - rhs| (should be small if the solution is accurate)
    u_phi : float
        ∫ u(x) φ(x) dx (useful diagnostic — how much does φ "see" of u?)
    """
    spec: str
    diffusion: float
    advection: float
    reaction: float
    lhs: float
    rhs: float
    residual: float
    u_phi: float

    def to_dict(self) -> dict:
        return {
            "spec": self.spec,
            "diffusion": self.diffusion,
            "advection": self.advection,
            "reaction": self.reaction,
            "lhs": self.lhs,
            "rhs": self.rhs,
            "residual": self.residual,
            "u_phi": self.u_phi,
        }

    def blind_response(self) -> dict:
        """
        What the LLM sees: only the rhs (observable) and u_phi.
        The LLM does NOT get the per-term breakdown — that would
        reveal the coefficient structure directly.
        """
        return {
            "spec": self.spec,
            "integral_f_phi": self.rhs,
            "integral_u_phi": self.u_phi,
        }


# ---------------------------------------------------------------------------
# Core integrator
# ---------------------------------------------------------------------------

def compute_weak_form(
    sol: EllipticSolution,
    tf: TestFunction,
    n_quad: Optional[int] = None,
) -> WeakFormResult:
    """
    Compute all weak-form integrals for one test function against a solution.

    Uses the solution grid for integration. If n_quad is specified, interpolates
    to a finer quadrature grid.

    The weak form identity (after integration by parts):
        ∫ a u' φ' dx  +  ∫ b u' φ dx  +  ∫ c u φ dx  =  ∫ f φ dx
    """
    pde = sol.pde
    x0, x1 = pde.domain

    if n_quad is not None and n_quad != sol.n_grid:
        # Interpolate solution to quadrature grid
        x_q = np.linspace(x0, x1, n_quad)
        u_q = np.interp(x_q, sol.x, sol.u)
        # Derivative by interpolating the FD derivative
        u_x_q = np.interp(x_q, sol.x, sol.u_x())
    else:
        x_q = sol.x
        u_q = sol.u
        u_x_q = sol.u_x()

    dx = x_q[1] - x_q[0]

    # Evaluate everything on the quadrature grid
    phi_q = tf(x_q)
    dphi_q = tf.derivative(x_q, 1)

    a_q = pde.a(x_q)
    b_q = pde.b(x_q)
    c_q = pde.c(x_q)
    f_q = pde.f(x_q)

    # Integrands
    integrand_diffusion = a_q * u_x_q * dphi_q     # a u' φ'
    integrand_advection = b_q * u_x_q * phi_q       # b u' φ
    integrand_reaction = c_q * u_q * phi_q           # c u φ
    integrand_rhs = f_q * phi_q                      # f φ
    integrand_u_phi = u_q * phi_q                    # u φ

    # Integrate
    I = lambda y: _composite_simpson(y, dx)

    diffusion = I(integrand_diffusion)
    advection = I(integrand_advection)
    reaction = I(integrand_reaction)
    rhs = I(integrand_rhs)
    u_phi = I(integrand_u_phi)

    lhs = diffusion + advection + reaction

    return WeakFormResult(
        spec=tf.spec,
        diffusion=diffusion,
        advection=advection,
        reaction=reaction,
        lhs=lhs,
        rhs=rhs,
        residual=abs(lhs - rhs),
        u_phi=u_phi,
    )


# ---------------------------------------------------------------------------
# Battery: run many test functions
# ---------------------------------------------------------------------------

def compute_weak_form_battery(
    sol: EllipticSolution,
    test_fns: Optional[list[TestFunction]] = None,
    n_quad: Optional[int] = None,
) -> list[WeakFormResult]:
    """
    Run weak-form integration for a list of test functions.
    Defaults to the standard library if none provided.
    """
    if test_fns is None:
        test_fns = standard_library(sol.pde.domain)
    return [compute_weak_form(sol, tf, n_quad) for tf in test_fns]


def battery_to_table(results: list[WeakFormResult], mode: str = "full") -> str:
    """
    Format battery results as a tab-separated table.

    mode="full":  all columns (for diagnostics / scoring)
    mode="blind": only what the LLM sees (integral_f_phi, integral_u_phi)
    """
    if mode == "blind":
        header = "spec\tintegral_f_phi\tintegral_u_phi"
        rows = []
        for r in results:
            rows.append(f"{r.spec}\t{r.rhs:.10f}\t{r.u_phi:.10f}")
        return header + "\n" + "\n".join(rows)
    else:
        header = "spec\tdiffusion\tadvection\treaction\tlhs\trhs\tresidual\tu_phi"
        rows = []
        for r in results:
            rows.append(
                f"{r.spec}\t{r.diffusion:.10f}\t{r.advection:.10f}\t"
                f"{r.reaction:.10f}\t{r.lhs:.10f}\t{r.rhs:.10f}\t"
                f"{r.residual:.2e}\t{r.u_phi:.10f}"
            )
        return header + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# LLM-facing API: string in → scalar out
# ---------------------------------------------------------------------------

def weak_form_from_string(
    sol: EllipticSolution,
    spec: str,
    n_quad: Optional[int] = None,
) -> dict:
    """
    The LLM calls this with a test function string and gets back
    the observable weak-form response.

    Parameters
    ----------
    sol : EllipticSolution
        The solved PDE (solution data is visible; coefficients are hidden).
    spec : str
        Test function specification, e.g. "bump(0.2, 0.8)".

    Returns
    -------
    dict with keys:
        "spec": the input string
        "integral_f_phi": ∫ f φ dx (the weak-form response)
        "integral_u_phi": ∫ u φ dx (how much φ overlaps with u)

    Example
    -------
    >>> response = weak_form_from_string(sol, "bump(0.3, 0.7)")
    >>> print(response["integral_f_phi"])
    0.04217...
    """
    tf = parse_test_function(spec, domain=sol.pde.domain)
    result = compute_weak_form(sol, tf, n_quad)
    return result.blind_response()


# ---------------------------------------------------------------------------
# Linear system assembly (for coefficient recovery)
# ---------------------------------------------------------------------------

@dataclass
class WeakFormLinearSystem:
    """
    Assembles the linear system that connects unknown coefficients to
    weak-form observations.

    If a(x) = Σ αⱼ ψⱼ(x), b(x) = Σ βⱼ ψⱼ(x), etc., then for each
    test function φ_k:

        Σⱼ αⱼ ∫ ψⱼ u' φ_k' dx  +  Σⱼ βⱼ ∫ ψⱼ u' φ_k dx
        + Σⱼ γⱼ ∫ ψⱼ u φ_k dx  =  Σⱼ fⱼ ∫ ψⱼ φ_k dx

    This is a linear system  A @ θ = 0  (or A_lhs @ θ_lhs = A_rhs @ θ_rhs)
    where θ collects all unknown basis coefficients.

    Attributes
    ----------
    G_diffusion : (n_tests, n_basis) — ∫ ψⱼ u' φ_k' dx
    G_advection : (n_tests, n_basis) — ∫ ψⱼ u' φ_k dx
    G_reaction  : (n_tests, n_basis) — ∫ ψⱼ u  φ_k dx
    G_source    : (n_tests, n_basis) — ∫ ψⱼ    φ_k dx
    """
    G_diffusion: np.ndarray
    G_advection: np.ndarray
    G_reaction: np.ndarray
    G_source: np.ndarray
    test_specs: list[str]
    basis_labels: list[str]

    @property
    def n_tests(self) -> int:
        return self.G_diffusion.shape[0]

    @property
    def n_basis(self) -> int:
        return self.G_diffusion.shape[1]

    @property
    def n_unknowns(self) -> int:
        """Total number of scalar unknowns: 4 coefficient functions × n_basis each."""
        return 4 * self.n_basis

    def full_matrix(self) -> np.ndarray:
        """
        Assemble the full system matrix A such that A @ θ = 0, where
        θ = [α₀ ... α_{n-1}, β₀ ... β_{n-1}, γ₀ ... γ_{n-1}, -f₀ ... -f_{n-1}].

        Shape: (n_tests, 4 * n_basis)
        """
        return np.hstack([
            self.G_diffusion,
            self.G_advection,
            self.G_reaction,
            -self.G_source,
        ])

    def split_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Split into A_lhs @ θ_lhs = A_rhs @ θ_rhs where
        θ_lhs = [α, β, γ] and θ_rhs = [f_coeffs].

        Returns (A_lhs, A_rhs) with shapes (n_tests, 3*n_basis) and (n_tests, n_basis).
        """
        A_lhs = np.hstack([self.G_diffusion, self.G_advection, self.G_reaction])
        A_rhs = self.G_source
        return A_lhs, A_rhs

    def condition_number(self) -> float:
        """Condition number of the full system matrix."""
        A = self.full_matrix()
        s = np.linalg.svd(A, compute_uv=False)
        return float(s[0] / s[-1]) if s[-1] > 0 else float('inf')

    def singular_values(self) -> np.ndarray:
        """Singular values of the full matrix (diagnostic for rank / identifiability)."""
        A = self.full_matrix()
        return np.linalg.svd(A, compute_uv=False)


def assemble_linear_system(
    sol: EllipticSolution,
    basis: BasisFamily,
    test_fns: Optional[list[TestFunction]] = None,
    n_quad: Optional[int] = None,
) -> WeakFormLinearSystem:
    """
    Build the linear system connecting basis coefficients to weak-form integrals.

    For each test function φ_k and each basis function ψⱼ, computes:
        G_diffusion[k, j] = ∫ ψⱼ(x) u'(x) φ_k'(x) dx
        G_advection[k, j] = ∫ ψⱼ(x) u'(x) φ_k(x)  dx
        G_reaction[k, j]  = ∫ ψⱼ(x) u(x)  φ_k(x)  dx
        G_source[k, j]    = ∫ ψⱼ(x)        φ_k(x)  dx
    """
    if test_fns is None:
        test_fns = standard_library(sol.pde.domain)

    pde = sol.pde
    x0, x1 = pde.domain

    if n_quad is not None:
        x_q = np.linspace(x0, x1, n_quad)
        u_q = np.interp(x_q, sol.x, sol.u)
        u_x_q = np.interp(x_q, sol.x, sol.u_x())
    else:
        x_q = sol.x
        u_q = sol.u
        u_x_q = sol.u_x()

    dx = x_q[1] - x_q[0]
    n_tests = len(test_fns)
    n_b = basis.n_basis

    # Evaluate all basis functions on quadrature grid
    psi_q = basis.evaluate(x_q)  # (n_basis, N_quad)

    G_diff = np.zeros((n_tests, n_b))
    G_adv = np.zeros((n_tests, n_b))
    G_react = np.zeros((n_tests, n_b))
    G_src = np.zeros((n_tests, n_b))

    I = lambda y: _composite_simpson(y, dx)

    for k, tf in enumerate(test_fns):
        phi_k = tf(x_q)
        dphi_k = tf.derivative(x_q, 1)

        for j in range(n_b):
            G_diff[k, j] = I(psi_q[j] * u_x_q * dphi_k)
            G_adv[k, j] = I(psi_q[j] * u_x_q * phi_k)
            G_react[k, j] = I(psi_q[j] * u_q * phi_k)
            G_src[k, j] = I(psi_q[j] * phi_k)

    return WeakFormLinearSystem(
        G_diffusion=G_diff,
        G_advection=G_adv,
        G_reaction=G_react,
        G_source=G_src,
        test_specs=[tf.spec for tf in test_fns],
        basis_labels=basis.labels(),
    )
