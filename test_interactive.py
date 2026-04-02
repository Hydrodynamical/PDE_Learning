"""
Test and demo of the interactive LLM session.

Exercises:
    - Expression parser (valid and invalid inputs)
    - Boundary condition enforcement
    - Successful queries and error handling
    - Oracle strategy using arbitrary expressions
    - Final scoring
"""

import unittest
import numpy as np
from expression_parser import (
    SafeExprEvaluator,
    ExpressionError,
    make_test_function_from_string,
    validate_expression,
)
from interactive import InteractiveSession


class TestExpressionParser(unittest.TestCase):

    def test_simple_expressions(self):
        f = SafeExprEvaluator("x")
        x = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(f(x), x)

    def test_polynomial(self):
        f = SafeExprEvaluator("x*(1-x)")
        x = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(f(x), [0.0, 0.25, 0.0])

    def test_trig(self):
        f = SafeExprEvaluator("sin(pi*x)")
        x = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(f(x), [0.0, 1.0, 0.0], atol=1e-14)

    def test_complex_expression(self):
        f = SafeExprEvaluator("x*(1-x)*sin(3*pi*x)")
        x = np.linspace(0, 1, 100)
        vals = f(x)
        self.assertTrue(np.all(np.isfinite(vals)))
        self.assertAlmostEqual(vals[0], 0.0, places=10)
        self.assertAlmostEqual(vals[-1], 0.0, places=10)

    def test_power_caret(self):
        f = SafeExprEvaluator("x^2*(1-x)^2")
        x = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(f(x), [0.0, 0.0625, 0.0])

    def test_exp_gaussian(self):
        f = SafeExprEvaluator("exp(-50*(x-0.5)**2) * x * (1-x)")
        x = np.linspace(0, 1, 100)
        vals = f(x)
        self.assertTrue(np.all(np.isfinite(vals)))

    def test_rejects_unknown_function(self):
        with self.assertRaises(ExpressionError):
            SafeExprEvaluator("eval(x)")

    def test_rejects_unknown_variable(self):
        with self.assertRaises(ExpressionError):
            SafeExprEvaluator("y + 1")

    def test_rejects_import(self):
        with self.assertRaises(ExpressionError):
            SafeExprEvaluator("__import__('os')")


class TestTestFunctionConstruction(unittest.TestCase):

    def test_valid_test_function(self):
        tf, err = make_test_function_from_string("sin(pi*x)")
        self.assertIsNone(err)
        self.assertIsNotNone(tf)
        self.assertTrue(tf.satisfies_bcs())

    def test_bc_violation_detected(self):
        tf, err = make_test_function_from_string("sin(x)")  # sin(1) ≠ 0
        self.assertIsNone(tf)
        self.assertIn("Boundary condition violated", err)

    def test_constant_rejected(self):
        tf, err = make_test_function_from_string("1")
        self.assertIsNone(tf)
        self.assertIn("Boundary condition", err)

    def test_zero_rejected(self):
        tf, err = make_test_function_from_string("0*x")
        self.assertIsNone(tf)
        self.assertIn("zero", err.lower())

    def test_complex_valid(self):
        tf, err = make_test_function_from_string("x^2*(1-x)^2*cos(2*pi*x)")
        self.assertIsNone(err)
        self.assertIsNotNone(tf)

    def test_numerical_derivatives(self):
        tf, err = make_test_function_from_string("sin(pi*x)")
        self.assertIsNone(err)
        x = np.array([0.25, 0.5, 0.75])
        dphi = tf.derivative(x, 1)
        expected = np.pi * np.cos(np.pi * x)
        np.testing.assert_allclose(dphi, expected, atol=1e-4)


class TestInteractiveSession(unittest.TestCase):

    def setUp(self):
        self.session = InteractiveSession.from_difficulty("easy", seed=99, max_queries=20)

    def test_valid_query(self):
        resp = self.session.query("sin(pi*x)")
        self.assertEqual(resp["status"], "ok")
        self.assertIn("integral_f_phi", resp)
        self.assertIn("integral_u_phi", resp)
        self.assertEqual(resp["queries_remaining"], 19)

    def test_invalid_expression_returns_error(self):
        resp = self.session.query("this is not math")
        self.assertEqual(resp["status"], "error")
        # Errors don't count against budget
        self.assertEqual(resp["queries_remaining"], 20)

    def test_bc_violation_returns_error(self):
        resp = self.session.query("x + 1")
        self.assertEqual(resp["status"], "error")
        self.assertIn("Boundary condition", resp["message"])
        self.assertEqual(resp["queries_remaining"], 20)

    def test_budget_enforcement(self):
        session = InteractiveSession.from_difficulty("easy", seed=99, max_queries=3)
        for i in range(3):
            resp = session.query(f"sin({i+1}*pi*x)")
            self.assertEqual(resp["status"], "ok")
        resp = session.query("sin(4*pi*x)")
        self.assertEqual(resp["status"], "error")
        self.assertIn("budget", resp["message"].lower())

    def test_submit_prediction(self):
        n_b = self.session.basis.n_basis
        prediction = {
            "a": [1.0] + [0.0] * (n_b - 1),
            "b": [0.0] * n_b,
            "c": [0.0] * n_b,
            "f": [1.0] + [0.0] * (n_b - 1),
        }
        result = self.session.submit_prediction(prediction)
        self.assertEqual(result["status"], "scored")
        self.assertIn("score", result)
        self.assertIn("ground_truth", result)

    def test_system_prompt_generated(self):
        prompt = self.session.system_prompt()
        self.assertIn("elliptic pde", prompt.lower())
        self.assertIn("Legendre", prompt)
        self.assertIn("x =", prompt)

    def test_status(self):
        st = self.session.status()
        self.assertEqual(st["difficulty"], "easy")
        self.assertEqual(st["queries_used"], 0)


def demo():
    """Simulate an LLM session with a medium-difficulty task."""
    print("=" * 70)
    print("INTERACTIVE SESSION DEMO")
    print("=" * 70)

    session = InteractiveSession.from_difficulty("medium", seed=42, max_queries=30)

    # Show the system prompt (first 20 lines)
    prompt = session.system_prompt()
    print("\n--- System prompt (first 20 lines) ---")
    for line in prompt.split("\n")[:20]:
        print(f"  {line}")
    print("  ...")

    # Simulate LLM queries
    print("\n--- Simulated LLM queries ---\n")

    # Query 1: LLM tries something invalid
    resp = session.query("hello world")
    print(f"  Query: 'hello world'")
    print(f"  → {resp['status']}: {resp.get('message', '')}")
    print()

    # Query 2: LLM tries a function that violates BCs
    resp = session.query("cos(pi*x)")
    print(f"  Query: 'cos(pi*x)'")
    print(f"  → {resp['status']}: {resp.get('message', '')[:80]}")
    print()

    # Query 3-8: LLM uses Fourier sine modes (smart strategy)
    for k in range(1, 7):
        resp = session.query(f"sin({k}*pi*x)")
        print(f"  Query: 'sin({k}*pi*x)'")
        print(f"  → ∫fφ = {resp['integral_f_phi']:+.8f}, ∫uφ = {resp['integral_u_phi']:+.8f}")

    print()

    # Query 9-14: LLM uses localized probes
    localized = [
        "x*(1-x)*sin(pi*x)",
        "x*(1-x)*sin(2*pi*x)",
        "x*(1-x)*sin(3*pi*x)",
        "x^2*(1-x)^2",
        "x^2*(1-x)^2*cos(2*pi*x)",
        "exp(-100*(x-0.5)^2)*x*(1-x)",
    ]
    for spec in localized:
        resp = session.query(spec)
        if resp["status"] == "ok":
            print(f"  Query: '{spec}'")
            print(f"  → ∫fφ = {resp['integral_f_phi']:+.8f}, ∫uφ = {resp['integral_u_phi']:+.8f}")
        else:
            print(f"  Query: '{spec}' → ERROR: {resp['message'][:60]}")

    print()

    # Show status
    st = session.status()
    print(f"--- Status: {st['queries_used']} queries used, "
          f"{st['queries_remaining']} remaining, {st['n_errors']} errors ---\n")

    # Submit a dummy prediction
    n_b = session.basis.n_basis
    prediction = {
        "a": [0.5] + [0.0] * (n_b - 1),
        "b": [0.0] * n_b,
        "c": [0.0] * n_b,
        "f": [0.1] + [0.0] * (n_b - 1),
    }
    result = session.submit_prediction(prediction)
    print("--- Submitted dummy prediction ---")
    print(result["summary"])

    print(f"\n--- Ground truth ---")
    gt = result["ground_truth"]
    for name in ["a", "b", "c", "f"]:
        print(f"  {name}: {[round(v, 4) for v in gt[name]]}")

    print(f"\n--- Query history ---")
    print(session.query_history_table())

    print("\nDone.")


if __name__ == "__main__":
    print(">>> Running tests...\n")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestExpressionParser))
    suite.addTests(loader.loadTestsFromTestCase(TestTestFunctionConstruction))
    suite.addTests(loader.loadTestsFromTestCase(TestInteractiveSession))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n\n")
        demo()
