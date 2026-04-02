"""
Test function framework for weak-form probing.

A test function φ(x) must satisfy φ(x0) = φ(x1) = 0 (for Dirichlet BCs)
and be sufficiently smooth.

The key design goal: an LLM can specify a test function via a human-readable
string, e.g.:
    "bump(0.2, 0.8)"
    "sin(2*pi*x)"
    "hat(0.3, 0.5, 0.7)"
    "bump(0.1, 0.4) + 0.5 * bump(0.6, 0.9)"

And get back a callable TestFunction with precomputed derivatives.

Architecture:
    - TestFunction: wraps a callable + its derivatives, with metadata
    - TestFunctionLibrary: a fixed catalog of test functions (for static benchmark)
    - parse_test_function(s): parses a string into a TestFunction (for LLM interaction)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import re


# ---------------------------------------------------------------------------
# Core test function class
# ---------------------------------------------------------------------------

@dataclass
class TestFunction:
    """
    A test function φ(x) with its derivatives, defined on a domain.

    Attributes
    ----------
    phi : callable (np.ndarray -> np.ndarray)
        The test function itself.
    dphi : callable (np.ndarray -> np.ndarray)
        First derivative φ'(x).
    d2phi : callable (np.ndarray -> np.ndarray)
        Second derivative φ''(x).
    domain : (float, float)
    spec : str
        The string specification that generated this test function.
        This is what the LLM sees / submits.
    description : str
        Human-readable description.
    """
    phi: Callable[[np.ndarray], np.ndarray]
    dphi: Callable[[np.ndarray], np.ndarray]
    d2phi: Callable[[np.ndarray], np.ndarray]
    domain: tuple[float, float] = (0.0, 1.0)
    spec: str = ""
    description: str = ""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.phi(x)

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        if order == 0:
            return self.phi(x)
        elif order == 1:
            return self.dphi(x)
        elif order == 2:
            return self.d2phi(x)
        else:
            raise ValueError(f"Derivatives up to order 2 supported, got {order}")

    def satisfies_bcs(self, tol: float = 1e-10) -> bool:
        """Check φ(x0) = φ(x1) = 0."""
        x0, x1 = self.domain
        pts = np.array([x0, x1])
        vals = self.phi(pts)
        return bool(np.all(np.abs(vals) < tol))

    def to_dict(self) -> dict:
        return {"spec": self.spec, "description": self.description,
                "domain": list(self.domain)}


# ---------------------------------------------------------------------------
# Primitive builders
# ---------------------------------------------------------------------------

def _bump(a: float, b: float, domain: tuple[float, float] = (0.0, 1.0)) -> TestFunction:
    """
    Smooth bump:  φ(x) = sin²(π (x-a)/(b-a))  for x ∈ [a, b], else 0.

    Satisfies φ(a) = φ(b) = 0, φ ∈ C¹, and is strictly positive on (a, b).
    Using sin² gives analytic derivatives everywhere.
    """
    L = b - a

    def phi(x):
        out = np.zeros_like(x, dtype=float)
        mask = (x > a) & (x < b)
        t = (x[mask] - a) / L
        out[mask] = np.sin(np.pi * t) ** 2
        return out

    def dphi(x):
        out = np.zeros_like(x, dtype=float)
        mask = (x > a) & (x < b)
        t = (x[mask] - a) / L
        out[mask] = 2 * np.pi / L * np.sin(np.pi * t) * np.cos(np.pi * t)
        return out

    def d2phi(x):
        out = np.zeros_like(x, dtype=float)
        mask = (x > a) & (x < b)
        t = (x[mask] - a) / L
        out[mask] = 2 * (np.pi / L) ** 2 * (np.cos(np.pi * t) ** 2 - np.sin(np.pi * t) ** 2)
        return out

    return TestFunction(
        phi=phi, dphi=dphi, d2phi=d2phi, domain=domain,
        spec=f"bump({a},{b})",
        description=f"sin² bump on [{a}, {b}]",
    )


def _sin_mode(k: int, domain: tuple[float, float] = (0.0, 1.0)) -> TestFunction:
    """
    Fourier-sine mode:  φ(x) = sin(k π (x - x0) / L)

    Automatically satisfies Dirichlet BCs at x0, x1.
    """
    x0, x1 = domain
    L = x1 - x0

    def phi(x):
        return np.sin(k * np.pi * (x - x0) / L)

    def dphi(x):
        return (k * np.pi / L) * np.cos(k * np.pi * (x - x0) / L)

    def d2phi(x):
        return -(k * np.pi / L) ** 2 * np.sin(k * np.pi * (x - x0) / L)

    return TestFunction(
        phi=phi, dphi=dphi, d2phi=d2phi, domain=domain,
        spec=f"sin({k}*pi*x)",
        description=f"sin({k}πx/L), Fourier-sine mode k={k}",
    )


def _hat(a: float, peak: float, b: float,
         domain: tuple[float, float] = (0.0, 1.0)) -> TestFunction:
    """
    Piecewise-linear hat function:
        φ(x) = (x-a)/(peak-a)   for x ∈ [a, peak]
        φ(x) = (b-x)/(b-peak)   for x ∈ [peak, b]
        φ(x) = 0                elsewhere

    C⁰ only — derivative has jumps at a, peak, b.
    The second derivative is zero a.e. (distributions, really).
    Useful as a simple probe, but less smooth than bump or sin.
    """
    def phi(x):
        out = np.zeros_like(x, dtype=float)
        left = (x >= a) & (x <= peak)
        right = (x > peak) & (x <= b)
        out[left] = (x[left] - a) / (peak - a)
        out[right] = (b - x[right]) / (b - peak)
        return out

    def dphi(x):
        out = np.zeros_like(x, dtype=float)
        left = (x > a) & (x < peak)
        right = (x > peak) & (x < b)
        out[left] = 1.0 / (peak - a)
        out[right] = -1.0 / (b - peak)
        return out

    def d2phi(x):
        # Zero a.e. — the distributional derivative has delta masses
        return np.zeros_like(x, dtype=float)

    return TestFunction(
        phi=phi, dphi=dphi, d2phi=d2phi, domain=domain,
        spec=f"hat({a},{peak},{b})",
        description=f"hat function peaking at {peak}, supported on [{a}, {b}]",
    )


def _poly_bubble(p: int, domain: tuple[float, float] = (0.0, 1.0)) -> TestFunction:
    """
    Polynomial bubble:  φ(x) = ((x - x0)(x1 - x))^p  (normalized to max = 1).

    Satisfies BCs, smooth, with increasing flatness near boundaries for larger p.
    """
    x0, x1 = domain
    L = x1 - x0
    # Maximum of (x-x0)(x1-x) is at midpoint, value (L/2)^2
    max_val = (L / 2) ** (2 * p)

    def phi(x):
        raw = ((x - x0) * (x1 - x)) ** p
        return raw / max_val

    def dphi(x):
        # d/dx [(x-x0)(x1-x)]^p = p [(x-x0)(x1-x)]^{p-1} * [(x1-x) - (x-x0)]
        if p == 0:
            return np.zeros_like(x, dtype=float)
        base = (x - x0) * (x1 - x)
        dbase = (x1 - x) - (x - x0)  # = x1 + x0 - 2x
        return p * base ** (p - 1) * dbase / max_val

    def d2phi(x):
        if p == 0:
            return np.zeros_like(x, dtype=float)
        base = (x - x0) * (x1 - x)
        dbase = (x1 + x0 - 2 * x)
        d2base = -2.0
        if p == 1:
            return np.full_like(x, d2base / max_val, dtype=float)
        # p(p-1) base^{p-2} (dbase)^2 + p base^{p-1} d2base
        return (p * (p - 1) * base ** (p - 2) * dbase ** 2
                + p * base ** (p - 1) * d2base) / max_val

    return TestFunction(
        phi=phi, dphi=dphi, d2phi=d2phi, domain=domain,
        spec=f"poly_bubble({p})",
        description=f"polynomial bubble ((x-x0)(x1-x))^{p}, normalized",
    )


# ---------------------------------------------------------------------------
# Arithmetic on test functions (for composition)
# ---------------------------------------------------------------------------

def _add_test_functions(tf1: TestFunction, tf2: TestFunction,
                        c1: float = 1.0, c2: float = 1.0) -> TestFunction:
    """Return c1 * tf1 + c2 * tf2."""
    return TestFunction(
        phi=lambda x, _a=tf1, _b=tf2, _c1=c1, _c2=c2: _c1 * _a(x) + _c2 * _b(x),
        dphi=lambda x, _a=tf1, _b=tf2, _c1=c1, _c2=c2: (
            _c1 * _a.derivative(x, 1) + _c2 * _b.derivative(x, 1)),
        d2phi=lambda x, _a=tf1, _b=tf2, _c1=c1, _c2=c2: (
            _c1 * _a.derivative(x, 2) + _c2 * _b.derivative(x, 2)),
        domain=tf1.domain,
        spec=f"{c1}*({tf1.spec}) + {c2}*({tf2.spec})",
        description=f"linear combination",
    )


def _scale_test_function(tf: TestFunction, c: float) -> TestFunction:
    """Return c * tf."""
    return TestFunction(
        phi=lambda x, _tf=tf, _c=c: _c * _tf(x),
        dphi=lambda x, _tf=tf, _c=c: _c * _tf.derivative(x, 1),
        d2phi=lambda x, _tf=tf, _c=c: _c * _tf.derivative(x, 2),
        domain=tf.domain,
        spec=f"{c}*({tf.spec})",
        description=f"{c} × {tf.description}",
    )


# ---------------------------------------------------------------------------
# String parser
# ---------------------------------------------------------------------------

def parse_test_function(
    spec: str,
    domain: tuple[float, float] = (0.0, 1.0),
) -> TestFunction:
    """
    Parse a string specification into a TestFunction.

    Supported syntax:
        "bump(a, b)"           → sin² bump on [a, b]
        "sin(k*pi*x)"         → k-th Fourier sine mode
        "hat(a, peak, b)"     → piecewise linear hat
        "poly_bubble(p)"      → polynomial bubble of degree p
        "c * expr"            → scaled test function
        "expr1 + expr2"       → sum of test functions

    The parser handles simple expressions. For the LLM use case,
    single primitives or simple sums are the expected input.

    Examples
    --------
    >>> tf = parse_test_function("bump(0.2, 0.8)")
    >>> tf = parse_test_function("sin(3*pi*x)")
    >>> tf = parse_test_function("0.5 * bump(0.1, 0.5) + bump(0.5, 0.9)")
    """
    spec = spec.strip()

    # Try to split on top-level '+' (not inside parentheses)
    parts = _split_top_level(spec, '+')
    if len(parts) > 1:
        tfs = [parse_test_function(p.strip(), domain) for p in parts]
        result = tfs[0]
        for tf in tfs[1:]:
            result = _add_test_functions(result, tf)
        result.spec = spec
        return result

    # Try to match "c * expr"
    m = re.match(r'^([0-9.eE+-]+)\s*\*\s*(.+)$', spec)
    if m:
        c = float(m.group(1))
        inner = parse_test_function(m.group(2), domain)
        result = _scale_test_function(inner, c)
        result.spec = spec
        return result

    # Match primitives
    # bump(a, b)
    m = re.match(r'^bump\(\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\)$', spec)
    if m:
        return _bump(float(m.group(1)), float(m.group(2)), domain)

    # sin(k*pi*x)
    m = re.match(r'^sin\(\s*([0-9.eE+-]*)\s*\*?\s*pi\s*\*\s*x\s*\)$', spec)
    if m:
        k_str = m.group(1).strip()
        k = int(float(k_str)) if k_str else 1
        return _sin_mode(k, domain)

    # hat(a, peak, b)
    m = re.match(r'^hat\(\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\)$', spec)
    if m:
        return _hat(float(m.group(1)), float(m.group(2)), float(m.group(3)), domain)

    # poly_bubble(p)
    m = re.match(r'^poly_bubble\(\s*([0-9]+)\s*\)$', spec)
    if m:
        return _poly_bubble(int(m.group(1)), domain)

    raise ValueError(
        f"Cannot parse test function spec: '{spec}'\n"
        f"Supported: bump(a,b), sin(k*pi*x), hat(a,peak,b), poly_bubble(p), "
        f"c*expr, expr1 + expr2"
    )


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split string on `sep` but only at top level (not inside parentheses)."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current))
    return parts


# ---------------------------------------------------------------------------
# Standard library (fixed catalog for static benchmark)
# ---------------------------------------------------------------------------

def standard_library(
    domain: tuple[float, float] = (0.0, 1.0),
    n_sin_modes: int = 5,
    n_bumps: int = 4,
) -> list[TestFunction]:
    """
    A fixed set of test functions for the static benchmark.

    Returns a diverse collection: sine modes, bumps at different locations,
    polynomial bubbles. The LLM gets the weak-form results for all of these.
    """
    library = []

    # Sine modes: sin(kπx) for k = 1, ..., n_sin_modes
    for k in range(1, n_sin_modes + 1):
        library.append(_sin_mode(k, domain))

    # Bumps at evenly spaced intervals
    x0, x1 = domain
    L = x1 - x0
    for i in range(n_bumps):
        a = x0 + i * L / n_bumps
        b = x0 + (i + 1) * L / n_bumps
        library.append(_bump(a, b, domain))

    # Polynomial bubbles
    for p in [1, 2, 3]:
        library.append(_poly_bubble(p, domain))

    # A few localized bumps (narrow)
    library.append(_bump(0.15, 0.35, domain))
    library.append(_bump(0.45, 0.55, domain))
    library.append(_bump(0.65, 0.85, domain))

    return library
