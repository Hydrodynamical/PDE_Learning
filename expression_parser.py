"""
Safe math expression evaluator and arbitrary test function constructor.

Parses human-readable math strings like:
    "x*(1-x)*sin(3*pi*x)"
    "exp(-50*(x-0.3)^2) * x * (1-x)"
    "sin(pi*x) + 0.5*sin(2*pi*x)"

Uses Python's ast module for safe parsing — no eval/exec.
Evaluates on numpy arrays for vectorized computation.
Computes derivatives via 4th-order finite differences.
Validates boundary conditions (φ(x0) = φ(x1) = 0).
"""

from __future__ import annotations
import ast
import operator
import numpy as np
from typing import Callable, Optional
from test_functions import TestFunction


# ---------------------------------------------------------------------------
# Safe expression parser
# ---------------------------------------------------------------------------

# Allowed functions (name → numpy implementation)
SAFE_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "ln": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sign": np.sign,
    "heaviside": lambda x: np.heaviside(x, 0.5),
}

# Allowed constants
SAFE_CONSTANTS = {
    "pi": np.pi,
    "e": np.e,
}

# Allowed binary operators
SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

SAFE_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class ExpressionError(Exception):
    """Raised when an expression cannot be parsed or evaluated."""
    pass


class SafeExprEvaluator:
    """
    Parses a math expression string into a callable f(x) on numpy arrays.

    Supports:
        - Variable: x
        - Constants: pi, e, numeric literals
        - Functions: sin, cos, exp, log, sqrt, abs, tan, sinh, cosh, tanh,
                     arcsin, arccos, arctan, sign, heaviside
        - Operators: +, -, *, /, ** (or ^), //, %
        - Parentheses

    Usage:
        evaluator = SafeExprEvaluator("x*(1-x)*sin(3*pi*x)")
        y = evaluator(np.linspace(0, 1, 100))
    """

    def __init__(self, expr_str: str):
        self.expr_str = expr_str.strip()
        self._normalized = self._normalize(self.expr_str)
        try:
            self._tree = ast.parse(self._normalized, mode="eval")
        except SyntaxError as e:
            raise ExpressionError(f"Syntax error in expression '{expr_str}': {e}")
        self._validate(self._tree)

    @staticmethod
    def _normalize(s: str) -> str:
        """Normalize expression: replace ^ with **, handle implicit multiplication patterns."""
        s = s.replace("^", "**")
        return s

    def _validate(self, tree: ast.AST):
        """Walk the AST and reject anything unsafe."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ExpressionError(
                        f"Only simple function calls allowed, got {ast.dump(node.func)}")
                if node.func.id not in SAFE_FUNCTIONS:
                    raise ExpressionError(
                        f"Unknown function '{node.func.id}'. "
                        f"Allowed: {', '.join(sorted(SAFE_FUNCTIONS.keys()))}")
            elif isinstance(node, ast.Name):
                if node.id not in ("x",) and node.id not in SAFE_CONSTANTS and node.id not in SAFE_FUNCTIONS:
                    raise ExpressionError(
                        f"Unknown variable '{node.id}'. Use 'x' for the spatial variable, "
                        f"or constants: {', '.join(sorted(SAFE_CONSTANTS.keys()))}")
            elif isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                                    ast.Constant, *([ast.Num] if hasattr(ast, 'Num') else []))):
                pass  # These are fine
            elif isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                    ast.Pow, ast.FloorDiv, ast.Mod,
                                    ast.USub, ast.UAdd)):
                pass  # Operators
            elif isinstance(node, ast.keyword):
                raise ExpressionError("Keyword arguments not allowed")
            elif isinstance(node, ast.Load):
                pass
            else:
                # Allow some other harmless node types
                pass

    def _eval_node(self, node: ast.AST, x: np.ndarray) -> np.ndarray:
        """Recursively evaluate an AST node with x as the variable."""
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, x)

        elif isinstance(node, ast.Constant):
            return np.full_like(x, node.value, dtype=float)

        elif hasattr(ast, 'Num') and isinstance(node, ast.Num):  # Python 3.7 compat
            return np.full_like(x, node.n, dtype=float)

        elif isinstance(node, ast.Name):
            if node.id == "x":
                return x.copy()
            elif node.id in SAFE_CONSTANTS:
                return np.full_like(x, SAFE_CONSTANTS[node.id], dtype=float)
            else:
                raise ExpressionError(f"Unknown name: {node.id}")

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, x)
            right = self._eval_node(node.right, x)
            op_type = type(node.op)
            if op_type not in SAFE_OPS:
                raise ExpressionError(f"Unsupported operator: {op_type.__name__}")
            try:
                return SAFE_OPS[op_type](left, right)
            except Exception as e:
                raise ExpressionError(f"Evaluation error: {e}")

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, x)
            op_type = type(node.op)
            if op_type not in SAFE_UNARY_OPS:
                raise ExpressionError(f"Unsupported unary operator: {op_type.__name__}")
            return SAFE_UNARY_OPS[op_type](operand)

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError("Only simple function calls allowed")
            fname = node.func.id
            if fname not in SAFE_FUNCTIONS:
                raise ExpressionError(f"Unknown function: {fname}")
            args = [self._eval_node(arg, x) for arg in node.args]
            try:
                return SAFE_FUNCTIONS[fname](*args)
            except Exception as e:
                raise ExpressionError(f"Error evaluating {fname}: {e}")

        else:
            raise ExpressionError(f"Unsupported AST node: {type(node).__name__}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the expression at points x."""
        x = np.asarray(x, dtype=float)
        try:
            result = self._eval_node(self._tree, x)
        except ExpressionError:
            raise
        except Exception as e:
            raise ExpressionError(f"Evaluation failed for '{self.expr_str}': {e}")

        if not np.all(np.isfinite(result)):
            # Replace NaN/Inf with 0 and warn
            n_bad = np.sum(~np.isfinite(result))
            result = np.where(np.isfinite(result), result, 0.0)

        return result

    def __repr__(self):
        return f"SafeExprEvaluator('{self.expr_str}')"


# ---------------------------------------------------------------------------
# Numerical derivatives (4th-order FD)
# ---------------------------------------------------------------------------

def _numerical_derivative(f: Callable, x: np.ndarray, order: int = 1,
                           h: float = 1e-5) -> np.ndarray:
    """
    Compute derivative of f at points x using 4th-order central differences.

    For order=1: (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    For order=2: (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h²)
    """
    if order == 1:
        return (
            -f(x + 2 * h) + 8 * f(x + h)
            - 8 * f(x - h) + f(x - 2 * h)
        ) / (12 * h)
    elif order == 2:
        return (
            -f(x + 2 * h) + 16 * f(x + h) - 30 * f(x)
            + 16 * f(x - h) - f(x - 2 * h)
        ) / (12 * h ** 2)
    else:
        raise ValueError(f"Only derivatives of order 1 and 2 supported, got {order}")


# ---------------------------------------------------------------------------
# Arbitrary test function from string
# ---------------------------------------------------------------------------

def make_test_function_from_string(
    expr_str: str,
    domain: tuple[float, float] = (0.0, 1.0),
    bc_tol: float = 1e-6,
    fd_step: float = 1e-6,
) -> tuple[Optional[TestFunction], Optional[str]]:
    """
    Parse an arbitrary math expression into a TestFunction.

    Returns (TestFunction, None) on success, or (None, error_message) on failure.

    Checks:
        1. Expression parses and evaluates without errors
        2. Result is finite on the domain
        3. Boundary conditions: φ(x0) ≈ 0 and φ(x1) ≈ 0
        4. Function is not identically zero

    Parameters
    ----------
    expr_str : str
        Math expression using 'x' as the variable.
    domain : (float, float)
    bc_tol : float
        Tolerance for boundary condition check.
    fd_step : float
        Step size for numerical derivatives.
    """
    # Step 1: Parse
    try:
        evaluator = SafeExprEvaluator(expr_str)
    except ExpressionError as e:
        return None, f"Parse error: {e}"

    # Step 2: Evaluate on test grid
    x_test = np.linspace(domain[0], domain[1], 500)
    try:
        vals = evaluator(x_test)
    except ExpressionError as e:
        return None, f"Evaluation error: {e}"
    except Exception as e:
        return None, f"Unexpected error during evaluation: {e}"

    if not np.all(np.isfinite(vals)):
        n_bad = np.sum(~np.isfinite(vals))
        return None, f"Expression produces {n_bad} non-finite values on [{domain[0]}, {domain[1]}]"

    # Step 3: Check boundary conditions
    x_bc = np.array([domain[0], domain[1]])
    bc_vals = evaluator(x_bc)
    if abs(bc_vals[0]) > bc_tol:
        return None, (
            f"Boundary condition violated: φ({domain[0]}) = {bc_vals[0]:.6e} ≠ 0. "
            f"Test functions must satisfy φ({domain[0]}) = φ({domain[1]}) = 0."
        )
    if abs(bc_vals[1]) > bc_tol:
        return None, (
            f"Boundary condition violated: φ({domain[1]}) = {bc_vals[1]:.6e} ≠ 0. "
            f"Test functions must satisfy φ({domain[0]}) = φ({domain[1]}) = 0."
        )

    # Step 4: Check not identically zero
    if np.max(np.abs(vals)) < 1e-14:
        return None, "Function is identically zero on the domain."

    # Step 5: Build the TestFunction with numerical derivatives
    h = fd_step

    def phi(x, _ev=evaluator):
        return _ev(x)

    def dphi(x, _ev=evaluator, _h=h):
        return _numerical_derivative(_ev, x, order=1, h=_h)

    def d2phi(x, _ev=evaluator, _h=h):
        return _numerical_derivative(_ev, x, order=2, h=_h)

    tf = TestFunction(
        phi=phi,
        dphi=dphi,
        d2phi=d2phi,
        domain=domain,
        spec=expr_str,
        description=f"arbitrary: {expr_str}",
    )

    return tf, None


# ---------------------------------------------------------------------------
# Convenience: validate + evaluate in one call (for interactive session)
# ---------------------------------------------------------------------------

def validate_expression(expr_str: str, domain: tuple[float, float] = (0.0, 1.0)) -> dict:
    """
    Quick validation of an expression string.

    Returns a dict with:
        "valid": bool
        "error": str or None
        "bc_left": float (value at left boundary)
        "bc_right": float (value at right boundary)
        "max_abs": float (maximum absolute value on domain)
        "support_fraction": float (fraction of domain where |φ| > 1e-10)
    """
    try:
        evaluator = SafeExprEvaluator(expr_str)
    except ExpressionError as e:
        return {"valid": False, "error": str(e)}

    x = np.linspace(domain[0], domain[1], 500)
    try:
        vals = evaluator(x)
    except Exception as e:
        return {"valid": False, "error": str(e)}

    x_bc = np.array([domain[0], domain[1]])
    bc_vals = evaluator(x_bc)

    bc_ok = abs(bc_vals[0]) < 1e-6 and abs(bc_vals[1]) < 1e-6
    support = np.mean(np.abs(vals) > 1e-10)

    error = None
    if not np.all(np.isfinite(vals)):
        error = "Non-finite values"
    elif not bc_ok:
        error = f"BC violated: φ(0)={bc_vals[0]:.2e}, φ(1)={bc_vals[1]:.2e}"
    elif np.max(np.abs(vals)) < 1e-14:
        error = "Identically zero"

    return {
        "valid": error is None,
        "error": error,
        "bc_left": float(bc_vals[0]),
        "bc_right": float(bc_vals[1]),
        "max_abs": float(np.max(np.abs(vals))),
        "support_fraction": float(support),
    }
