"""
Elliptic PDE framework.

Canonical form (divergence form):
    -(a(x) u'(x))' + b(x) u'(x) + c(x) u(x) = f(x)    on [x0, x1]

with Dirichlet BCs:  u(x0) = u_left,  u(x1) = u_right.

Coefficients a, b, c, f are CoefficientFunction objects from basis.py.

The solver uses second-order centered finite differences.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from basis import (
    BasisFamily,
    CoefficientFunction,
    make_random_coefficient,
    constant_coefficient,
    zero_coefficient,
)


# ---------------------------------------------------------------------------
# PDE definition
# ---------------------------------------------------------------------------

@dataclass
class EllipticPDE:
    """
    Represents  -(a u')' + b u' + c u = f  on [x0, x1].

    All coefficient functions live on the same domain.
    """
    a: CoefficientFunction   # diffusion (must be > 0 for ellipticity)
    b: CoefficientFunction   # advection
    c: CoefficientFunction   # reaction
    f: CoefficientFunction   # source / forcing
    domain: tuple[float, float] = (0.0, 1.0)
    bc_left: float = 0.0
    bc_right: float = 0.0

    def __post_init__(self):
        # Label the coefficients for display
        self.a.name = self.a.name or "a(x)"
        self.b.name = self.b.name or "b(x)"
        self.c.name = self.c.name or "c(x)"
        self.f.name = self.f.name or "f(x)"

    def is_elliptic(self, x: np.ndarray, tol: float = 1e-10) -> bool:
        """Check that a(x) > tol everywhere on the grid."""
        return bool(np.all(self.a(x) > tol))

    def coefficients(self) -> dict[str, CoefficientFunction]:
        return {"a": self.a, "b": self.b, "c": self.c, "f": self.f}

    def to_dict(self) -> dict:
        """Serializable representation (for benchmark ground truth)."""
        return {
            "domain": list(self.domain),
            "bc_left": self.bc_left,
            "bc_right": self.bc_right,
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "c": self.c.to_dict(),
            "f": self.f.to_dict(),
        }

    def describe(self) -> str:
        """Human-readable PDE description."""
        lines = [
            f"Elliptic PDE on [{self.domain[0]}, {self.domain[1]}]:",
            f"  -(a(x) u')' + b(x) u' + c(x) u = f(x)",
            f"  a(x) = {self.a.to_latex()}",
            f"  b(x) = {self.b.to_latex()}",
            f"  c(x) = {self.c.to_latex()}",
            f"  f(x) = {self.f.to_latex()}",
            f"  u({self.domain[0]}) = {self.bc_left},  u({self.domain[1]}) = {self.bc_right}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class EllipticSolution:
    """
    Container for a solved elliptic PDE, with diagnostics.
    """
    pde: EllipticPDE
    x: np.ndarray          # grid points (N,)
    u: np.ndarray          # solution values (N,)
    residual: np.ndarray   # pointwise residual on interior (N-2,)
    n_grid: int = 0

    def __post_init__(self):
        self.n_grid = len(self.x)

    # -- Derived quantities for diagnostics --

    def u_x(self) -> np.ndarray:
        """First derivative via 4th-order centered differences.

        Interior: (-u_{i+2} + 8u_{i+1} - 8u_{i-1} + u_{i-2}) / (12h)
        Near boundaries: falls back to 2nd-order or one-sided stencils.
        """
        dx = self.x[1] - self.x[0]
        N = len(self.u)
        du = np.zeros(N)

        # 4th-order interior (indices 2 .. N-3)
        if N >= 5:
            du[2:-2] = (
                -self.u[4:] + 8 * self.u[3:-1]
                - 8 * self.u[1:-3] + self.u[:-4]
            ) / (12 * dx)

        # 2nd-order at indices 1 and N-2
        if N >= 3:
            du[1] = (self.u[2] - self.u[0]) / (2 * dx)
            du[-2] = (self.u[-1] - self.u[-3]) / (2 * dx)

        # 1st-order at boundaries
        du[0] = (-3 * self.u[0] + 4 * self.u[1] - self.u[2]) / (2 * dx) if N >= 3 else (self.u[1] - self.u[0]) / dx
        du[-1] = (3 * self.u[-1] - 4 * self.u[-2] + self.u[-3]) / (2 * dx) if N >= 3 else (self.u[-1] - self.u[-2]) / dx

        return du

    def u_xx(self) -> np.ndarray:
        """Second derivative via 4th-order centered differences.

        Interior: (-u_{i+2} + 16u_{i+1} - 30u_i + 16u_{i-1} - u_{i-2}) / (12h²)
        """
        dx = self.x[1] - self.x[0]
        N = len(self.u)
        d2u = np.zeros(N)

        # 4th-order interior
        if N >= 5:
            d2u[2:-2] = (
                -self.u[4:] + 16 * self.u[3:-1] - 30 * self.u[2:-2]
                + 16 * self.u[1:-3] - self.u[:-4]
            ) / (12 * dx**2)

        # 2nd-order at indices 1 and N-2
        if N >= 3:
            d2u[1] = (self.u[2] - 2 * self.u[1] + self.u[0]) / dx**2
            d2u[-2] = (self.u[-1] - 2 * self.u[-2] + self.u[-3]) / dx**2

        return d2u

    def max_residual(self) -> float:
        return float(np.max(np.abs(self.residual)))

    def energy_norm(self) -> float:
        """∫ a(x) |u'(x)|² dx  (approximate)."""
        dx = self.x[1] - self.x[0]
        du = self.u_x()
        a_vals = self.pde.a(self.x)
        return float(np.sum(a_vals * du**2) * dx)

    def to_dict(self) -> dict:
        """Export solution data (for benchmark / LLM consumption)."""
        return {
            "x": self.x.tolist(),
            "u": self.u.tolist(),
            "n_grid": self.n_grid,
            "max_residual": self.max_residual(),
            "energy_norm": self.energy_norm(),
        }

    def diagnostics_table(self) -> dict:
        """
        Returns a dict of arrays, each (N,), suitable for plotting or
        feeding to an LLM as a table.

        Keys: x, u, u_x, u_xx, a, b, c, f, a_u_x (flux = a(x)*u'(x))
        """
        x = self.x
        return {
            "x": x,
            "u": self.u,
            "u_x": self.u_x(),
            "u_xx": self.u_xx(),
            "a": self.pde.a(x),
            "b": self.pde.b(x),
            "c": self.pde.c(x),
            "f": self.pde.f(x),
            "flux": self.pde.a(x) * self.u_x(),
        }


def solve_elliptic(pde: EllipticPDE, n_grid: int = 201) -> EllipticSolution:
    """
    Solve  -(a u')' + b u' + c u = f  via second-order finite differences.

    Discretization (interior point i, uniform grid spacing h):
        -(a_{i+1/2} (u_{i+1} - u_i) - a_{i-1/2} (u_i - u_{i-1})) / h²
        + b_i (u_{i+1} - u_{i-1}) / (2h)
        + c_i u_i
        = f_i

    where a_{i±1/2} = a((x_i + x_{i±1})/2).
    """
    x0, x1 = pde.domain
    x = np.linspace(x0, x1, n_grid)
    h = x[1] - x[0]
    N = n_grid

    # Evaluate coefficients
    a_vals = pde.a(x)
    b_vals = pde.b(x)
    c_vals = pde.c(x)
    f_vals = pde.f(x)

    # Half-grid diffusion coefficients
    x_half_right = (x[:-1] + x[1:]) / 2.0
    a_half = pde.a(x_half_right)  # a at x_{i+1/2}, length N-1

    # Build tridiagonal system for interior points (indices 1 .. N-2)
    n_interior = N - 2
    diag_main = np.zeros(n_interior)
    diag_lower = np.zeros(n_interior - 1)
    diag_upper = np.zeros(n_interior - 1)
    rhs = np.zeros(n_interior)

    for i in range(n_interior):
        idx = i + 1  # global index

        # Diffusion: -(a_{i+1/2}(u_{i+1}-u_i) - a_{i-1/2}(u_i-u_{i-1})) / h²
        a_right = a_half[idx]      # a_{i+1/2}
        a_left = a_half[idx - 1]   # a_{i-1/2}

        # Coefficient of u_{i-1}
        coeff_left = -a_left / h**2
        # Coefficient of u_i
        coeff_center = (a_left + a_right) / h**2 + c_vals[idx]
        # Coefficient of u_{i+1}
        coeff_right = -a_right / h**2

        # Advection: b_i (u_{i+1} - u_{i-1}) / (2h)
        coeff_left += -b_vals[idx] / (2 * h)
        coeff_right += b_vals[idx] / (2 * h)

        diag_main[i] = coeff_center
        if i > 0:
            diag_lower[i - 1] = coeff_left
        if i < n_interior - 1:
            diag_upper[i] = coeff_right

        rhs[i] = f_vals[idx]

        # Boundary contributions to RHS
        if idx == 1:
            rhs[i] -= coeff_left * pde.bc_left
        if idx == N - 2:
            rhs[i] -= coeff_right * pde.bc_right

    # Solve tridiagonal system (Thomas algorithm via numpy)
    A = np.zeros((n_interior, n_interior))
    np.fill_diagonal(A, diag_main)
    np.fill_diagonal(A[:-1, 1:], diag_upper)
    np.fill_diagonal(A[1:, :-1], diag_lower)
    u_interior = np.linalg.solve(A, rhs)

    # Assemble full solution
    u = np.zeros(N)
    u[0] = pde.bc_left
    u[-1] = pde.bc_right
    u[1:-1] = u_interior

    # Compute residual on interior for diagnostics
    residual = np.zeros(n_interior)
    for i in range(n_interior):
        idx = i + 1
        a_right = a_half[idx]
        a_left = a_half[idx - 1]
        diffusion = -(a_right * (u[idx + 1] - u[idx]) - a_left * (u[idx] - u[idx - 1])) / h**2
        advection = b_vals[idx] * (u[idx + 1] - u[idx - 1]) / (2 * h)
        reaction = c_vals[idx] * u[idx]
        residual[i] = diffusion + advection + reaction - f_vals[idx]

    return EllipticSolution(pde=pde, x=x, u=u, residual=residual)


# ---------------------------------------------------------------------------
# Random PDE generator
# ---------------------------------------------------------------------------

def make_random_elliptic_pde(
    basis: BasisFamily,
    rng: Optional[np.random.Generator] = None,
    a_min: float = 0.5,
    sparsity: float = 0.3,
    scale: float = 1.0,
    bc: tuple[float, float] = (0.0, 0.0),
) -> EllipticPDE:
    """
    Generate a random well-posed elliptic PDE.

    Ensures a(x) ≥ a_min > 0 for ellipticity by setting a(x) = a_min + |ã(x)|
    where ã is a random coefficient function.
    """
    if rng is None:
        rng = np.random.default_rng()

    domain = basis.domain

    # Diffusion: ensure positivity
    a_tilde = make_random_coefficient(basis, rng, sparsity=sparsity, scale=scale * 0.5, name="ã(x)")
    # We'll wrap this: a(x) = a_min + ã(x)^2 / (1 + ã(x)^2) * scale
    # Actually simpler: just make a constant + small perturbation
    a_coeffs = np.zeros(basis.n_basis)
    a_coeffs[0] = a_min + abs(rng.normal(0, scale * 0.3))
    # Add small higher-order terms
    for j in range(1, basis.n_basis):
        if rng.random() > sparsity:
            a_coeffs[j] = rng.normal(0, scale * 0.1)
    a = CoefficientFunction(basis, a_coeffs, name="a(x)")

    # Verify positivity on a fine grid
    x_check = np.linspace(domain[0], domain[1], 500)
    a_vals = a(x_check)
    if np.any(a_vals <= 0):
        # Fall back to constant
        a = constant_coefficient(basis, a_min, name="a(x)")

    b = make_random_coefficient(basis, rng, sparsity=sparsity, scale=scale, name="b(x)")
    c = make_random_coefficient(basis, rng, sparsity=sparsity, scale=scale, name="c(x)")
    f = make_random_coefficient(basis, rng, sparsity=sparsity, scale=scale, name="f(x)")

    return EllipticPDE(a=a, b=b, c=c, f=f, domain=domain, bc_left=bc[0], bc_right=bc[1])