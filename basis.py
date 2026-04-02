"""
Basis function framework for PDE coefficient representation.

Design principles:
    - Each BasisFamily defines a set of basis functions ψ_j(x) on a domain [a, b].
    - A CoefficientFunction represents f(x) = Σ c_j ψ_j(x) for some coefficient vector c.
    - Basis families are pluggable: swap Legendre for Fourier, wavelets, etc.
    - Everything supports batched numpy evaluation for integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Abstract basis family
# ---------------------------------------------------------------------------

class BasisFamily(ABC):
    """A family of basis functions {ψ_0, ψ_1, ..., ψ_{n-1}} on a domain [a, b]."""

    def __init__(self, n_basis: int, domain: tuple[float, float] = (0.0, 1.0)):
        self.n_basis = n_basis
        self.domain = domain

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate all basis functions at points x.

        Parameters
        ----------
        x : np.ndarray, shape (N,)
            Evaluation points.

        Returns
        -------
        np.ndarray, shape (n_basis, N)
            Row j = ψ_j(x) evaluated at all points.
        """
        ...

    @abstractmethod
    def evaluate_derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Evaluate derivatives of all basis functions at points x.

        Parameters
        ----------
        x : np.ndarray, shape (N,)
        order : int
            Derivative order (1, 2, ...).

        Returns
        -------
        np.ndarray, shape (n_basis, N)
        """
        ...

    @abstractmethod
    def label(self, j: int) -> str:
        """Human-readable label for basis function j (for benchmark display)."""
        ...

    def labels(self) -> list[str]:
        return [self.label(j) for j in range(self.n_basis)]

    def __repr__(self):
        return f"{self.__class__.__name__}(n_basis={self.n_basis}, domain={self.domain})"


# ---------------------------------------------------------------------------
# Concrete basis families
# ---------------------------------------------------------------------------

class LegendreBasis(BasisFamily):
    """
    Shifted Legendre polynomials on [a, b].

    Maps x ∈ [a, b] → ξ ∈ [-1, 1], then uses standard Legendre polynomials
    P_0, P_1, ..., P_{n-1}.
    """

    def _to_reference(self, x: np.ndarray) -> np.ndarray:
        a, b = self.domain
        return 2.0 * (x - a) / (b - a) - 1.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        xi = self._to_reference(x)
        # Build Legendre polynomials via three-term recurrence
        N = len(x)
        vals = np.zeros((self.n_basis, N))
        if self.n_basis >= 1:
            vals[0] = 1.0
        if self.n_basis >= 2:
            vals[1] = xi
        for j in range(2, self.n_basis):
            vals[j] = ((2 * j - 1) * xi * vals[j - 1] - (j - 1) * vals[j - 2]) / j
        return vals

    def evaluate_derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        a, b = self.domain
        xi = self._to_reference(x)
        scale = 2.0 / (b - a)  # dξ/dx

        N = len(x)
        # We'll compute derivatives via the identity:
        # P_j'(ξ) = j * P_{j-1}(ξ) + ξ * P_j'(ξ)  ... but it's cleaner to
        # use the recurrence for derivative polynomials.
        #
        # For order 1: P_j'(ξ) can be built from:
        #   P_0' = 0
        #   P_1' = 1
        #   P_j' = (2j-1)(P_{j-1} + ξ P_{j-1}') - (j-1) P_{j-2}') / j
        #        ... but this mixes values and derivatives.
        #
        # Cleanest: compute values, then use
        #   P_j'(ξ) = j/(ξ²-1) * (ξ P_j(ξ) - P_{j-1}(ξ))  for |ξ| ≠ 1
        # and boundary formulas at ξ = ±1.
        #
        # For higher order, iterate. For our purposes order ≤ 2 suffices.

        if order == 0:
            return self.evaluate(x)

        vals = self.evaluate(x)  # shape (n_basis, N)

        if order >= 1:
            dvals = np.zeros((self.n_basis, N))
            # P_0' = 0, P_1' = 1
            if self.n_basis >= 2:
                dvals[1] = 1.0
            # Recurrence: P_j'(ξ) = (2j-1) P_{j-1}(ξ) + P_{j-2}'(ξ)
            for j in range(2, self.n_basis):
                dvals[j] = (2 * j - 1) * vals[j - 1] + dvals[j - 2]
            dvals *= scale  # chain rule

        if order == 1:
            return dvals

        if order >= 2:
            d2vals = np.zeros((self.n_basis, N))
            # P_0'' = 0, P_1'' = 0, P_2'' = 3 (on [-1,1])
            # Recurrence: P_j''(ξ) = (2j-1)(2 P_{j-1}'(ξ) + ... ) -- let's
            # just differentiate the dvals recurrence:
            # d/dξ [P_j'(ξ)] = (2j-1) P_{j-1}'(ξ) + P_{j-2}''(ξ)
            # But dvals already has the chain rule applied. Let's work in ξ-space.

            # Redo in reference coordinates:
            dvals_ref = np.zeros((self.n_basis, N))
            if self.n_basis >= 2:
                dvals_ref[1] = 1.0
            for j in range(2, self.n_basis):
                dvals_ref[j] = (2 * j - 1) * vals[j - 1] + dvals_ref[j - 2]

            if self.n_basis >= 3:
                d2vals_ref = np.zeros((self.n_basis, N))
                # From recurrence: P_j''(ξ) = (2j-1) P_{j-1}'(ξ) + P_{j-2}''(ξ)
                for j in range(2, self.n_basis):
                    d2vals_ref[j] = (2 * j - 1) * dvals_ref[j - 1] + d2vals_ref[j - 2]

                d2vals = d2vals_ref * scale**2
            return d2vals

        raise NotImplementedError(f"Derivatives of order {order} > 2 not implemented.")

    def label(self, j: int) -> str:
        return f"P_{j}(x)"


class FourierBasis(BasisFamily):
    """
    Fourier basis on [a, b]: {1, cos(πx/L), sin(πx/L), cos(2πx/L), sin(2πx/L), ...}

    Ordering: ψ_0 = 1, ψ_{2k-1} = cos(kπx/L), ψ_{2k} = sin(kπx/L).
    Here L = (b - a) / 2.
    """

    def _freq_and_type(self, j: int) -> tuple[int, str]:
        """Return (frequency_index k, 'const' | 'cos' | 'sin') for basis index j."""
        if j == 0:
            return 0, "const"
        k = (j + 1) // 2
        return k, "cos" if j % 2 == 1 else "sin"

    def _phase(self, x: np.ndarray) -> np.ndarray:
        a, b = self.domain
        L = (b - a) / 2.0
        return np.pi * (x - a) / L  # maps [a, b] -> [0, 2π]... actually let's use
        # standard: map to [0, 2π] so period = (b-a)

    def _rescaled(self, x: np.ndarray) -> np.ndarray:
        a, b = self.domain
        return 2.0 * np.pi * (x - a) / (b - a)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        theta = self._rescaled(x)  # in [0, 2π]
        N = len(x)
        vals = np.zeros((self.n_basis, N))
        for j in range(self.n_basis):
            k, kind = self._freq_and_type(j)
            if kind == "const":
                vals[j] = 1.0
            elif kind == "cos":
                vals[j] = np.cos(k * theta)
            else:
                vals[j] = np.sin(k * theta)
        return vals

    def evaluate_derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        a, b = self.domain
        theta = self._rescaled(x)
        scale = 2.0 * np.pi / (b - a)
        N = len(x)
        vals = np.zeros((self.n_basis, N))
        for j in range(self.n_basis):
            k, kind = self._freq_and_type(j)
            if kind == "const":
                vals[j] = 0.0
            elif kind == "cos":
                # d^n/dx^n cos(kθ) where θ = scale*(x-a)
                phase_shift = order  # each derivative shifts cos -> -sin -> -cos -> sin -> ...
                coeff = (k * scale) ** order
                r = phase_shift % 4
                if r == 0:
                    vals[j] = coeff * np.cos(k * theta)
                elif r == 1:
                    vals[j] = -coeff * np.sin(k * theta)
                elif r == 2:
                    vals[j] = -coeff * np.cos(k * theta)
                else:
                    vals[j] = coeff * np.sin(k * theta)
            else:  # sin
                coeff = (k * scale) ** order
                r = order % 4
                if r == 0:
                    vals[j] = coeff * np.sin(k * theta)
                elif r == 1:
                    vals[j] = coeff * np.cos(k * theta)
                elif r == 2:
                    vals[j] = -coeff * np.sin(k * theta)
                else:
                    vals[j] = -coeff * np.cos(k * theta)
        return vals

    def label(self, j: int) -> str:
        k, kind = self._freq_and_type(j)
        if kind == "const":
            return "1"
        elif kind == "cos":
            return f"cos({k}θ)"
        else:
            return f"sin({k}θ)"


class MonomialBasis(BasisFamily):
    """
    Simple monomial basis: {1, x, x², ..., x^{n-1}} on [a, b].

    Included for simplicity / debugging. Ill-conditioned for large n,
    but perfectly fine for n ≤ 5 or so.
    """

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.vstack([x**j for j in range(self.n_basis)])

    def evaluate_derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        N = len(x)
        vals = np.zeros((self.n_basis, N))
        for j in range(order, self.n_basis):
            coeff = 1.0
            for k in range(order):
                coeff *= (j - k)
            vals[j] = coeff * x ** (j - order)
        return vals

    def label(self, j: int) -> str:
        if j == 0:
            return "1"
        if j == 1:
            return "x"
        return f"x^{j}"


# ---------------------------------------------------------------------------
# Coefficient function: linear combination of basis elements
# ---------------------------------------------------------------------------

@dataclass
class CoefficientFunction:
    """
    A function f(x) = Σ c_j ψ_j(x) represented by a basis and coefficient vector.

    This is the fundamental building block for PDE coefficients:
        a(x) = CoefficientFunction(basis, [1.0, 0.5, -0.3, ...])
    """
    basis: BasisFamily
    coeffs: np.ndarray  # shape (n_basis,)
    name: str = ""  # e.g., "a(x)", "b(x)" for display

    def __post_init__(self):
        self.coeffs = np.asarray(self.coeffs, dtype=float)
        assert len(self.coeffs) == self.basis.n_basis, (
            f"Coefficient length {len(self.coeffs)} != basis size {self.basis.n_basis}"
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate f(x) = Σ c_j ψ_j(x)."""
        basis_vals = self.basis.evaluate(x)  # (n_basis, N)
        return self.coeffs @ basis_vals  # (N,)

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        """Evaluate f^{(order)}(x)."""
        dbasis = self.basis.evaluate_derivative(x, order)
        return self.coeffs @ dbasis

    def to_latex(self) -> str:
        """Approximate LaTeX representation."""
        labels = self.basis.labels()
        terms = []
        for c, lab in zip(self.coeffs, labels):
            if abs(c) < 1e-12:
                continue
            if lab == "1":
                terms.append(f"{c:.4g}")
            else:
                terms.append(f"{c:.4g} \\cdot {lab}")
        if not terms:
            return "0"
        return " + ".join(terms).replace("+ -", "- ")

    def to_dict(self) -> dict:
        """Serializable representation for benchmark storage."""
        return {
            "basis_type": self.basis.__class__.__name__,
            "n_basis": self.basis.n_basis,
            "domain": list(self.basis.domain),
            "coeffs": self.coeffs.tolist(),
            "name": self.name,
        }

    def __repr__(self):
        name_str = f" ({self.name})" if self.name else ""
        return f"CoefficientFunction{name_str}: {self.to_latex()}"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_random_coefficient(
    basis: BasisFamily,
    rng: Optional[np.random.Generator] = None,
    sparsity: float = 0.0,
    scale: float = 1.0,
    name: str = "",
) -> CoefficientFunction:
    """
    Generate a random CoefficientFunction.

    Parameters
    ----------
    basis : BasisFamily
    rng : numpy random Generator (for reproducibility)
    sparsity : float in [0, 1)
        Probability of each coefficient being set to zero.
    scale : float
        Standard deviation of nonzero coefficients.
    name : str
        Label for display.
    """
    if rng is None:
        rng = np.random.default_rng()
    coeffs = rng.normal(0, scale, size=basis.n_basis)
    if sparsity > 0:
        mask = rng.random(basis.n_basis) > sparsity
        coeffs *= mask
    return CoefficientFunction(basis, coeffs, name=name)


def zero_coefficient(basis: BasisFamily, name: str = "") -> CoefficientFunction:
    """The zero function in a given basis."""
    return CoefficientFunction(basis, np.zeros(basis.n_basis), name=name)


def constant_coefficient(
    basis: BasisFamily, value: float, name: str = ""
) -> CoefficientFunction:
    """A constant function f(x) = value, assuming ψ_0 = 1."""
    coeffs = np.zeros(basis.n_basis)
    coeffs[0] = value
    return CoefficientFunction(basis, coeffs, name=name)
