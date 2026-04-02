"""Tests for the LLM response parser."""

import unittest
from llm_loop import parse_llm_response, _parse_prediction


class TestParser(unittest.TestCase):

    def test_simple_query(self):
        actions = parse_llm_response("QUERY: sin(pi*x)")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["action"], "query")
        self.assertEqual(actions[0]["spec"], "sin(pi*x)")

    def test_query_with_quotes(self):
        actions = parse_llm_response('QUERY: "x*(1-x)*sin(pi*x)"')
        self.assertEqual(actions[0]["spec"], "x*(1-x)*sin(pi*x)")

    def test_query_with_backticks(self):
        actions = parse_llm_response("QUERY: `sin(2*pi*x)`")
        self.assertEqual(actions[0]["spec"], "sin(2*pi*x)")

    def test_query_case_insensitive(self):
        actions = parse_llm_response("query: sin(pi*x)")
        self.assertEqual(actions[0]["action"], "query")

    def test_multiple_queries(self):
        text = "Let me try two.\n\nQUERY: sin(pi*x)\nQUERY: sin(2*pi*x)"
        actions = parse_llm_response(text)
        queries = [a for a in actions if a["action"] == "query"]
        self.assertEqual(len(queries), 2)
        self.assertEqual(queries[0]["spec"], "sin(pi*x)")
        self.assertEqual(queries[1]["spec"], "sin(2*pi*x)")

    def test_reasoning_plus_query(self):
        text = "I think sine modes will be informative.\n\nQUERY: sin(pi*x)"
        actions = parse_llm_response(text)
        reasoning = [a for a in actions if a["action"] == "reasoning"]
        queries = [a for a in actions if a["action"] == "query"]
        self.assertGreater(len(reasoning), 0)
        self.assertEqual(len(queries), 1)

    def test_prediction(self):
        text = (
            "PREDICT:\n"
            "a_coeffs = [1.0, 0.5, -0.3]\n"
            "b_coeffs = [0.0, 0.1, 0.2]\n"
            "c_coeffs = [-0.5, 0.0, 0.3]\n"
            "f_coeffs = [1.0, -0.5, 0.0]"
        )
        actions = parse_llm_response(text)
        preds = [a for a in actions if a["action"] == "predict"]
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["a"], [1.0, 0.5, -0.3])
        self.assertEqual(preds[0]["f"], [1.0, -0.5, 0.0])

    def test_prediction_short_form(self):
        text = (
            "PREDICT:\n"
            "a = [1.0, 0.5]\n"
            "b = [0.0, 0.1]\n"
            "c = [-0.5, 0.0]\n"
            "f = [1.0, -0.5]"
        )
        actions = parse_llm_response(text)
        preds = [a for a in actions if a["action"] == "predict"]
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["a"], [1.0, 0.5])

    def test_reasoning_only(self):
        text = "I need to think about this problem more carefully."
        actions = parse_llm_response(text)
        self.assertTrue(all(a["action"] == "reasoning" for a in actions))

    def test_empty_string(self):
        actions = parse_llm_response("")
        self.assertEqual(len(actions), 0)

    def test_prediction_with_reasoning(self):
        text = (
            "Based on my analysis, f is approximately constant and negative.\n"
            "The diffusion coefficient is roughly 0.5.\n\n"
            "PREDICT:\n"
            "a_coeffs = [0.5, 0.0, 0.0]\n"
            "b_coeffs = [0.0, 0.0, 0.0]\n"
            "c_coeffs = [0.0, 0.0, 0.0]\n"
            "f_coeffs = [-0.1, 0.0, 0.0]"
        )
        actions = parse_llm_response(text)
        reasoning = [a for a in actions if a["action"] == "reasoning"]
        preds = [a for a in actions if a["action"] == "predict"]
        self.assertGreater(len(reasoning), 0)
        self.assertEqual(len(preds), 1)

    def test_decompose(self):
        actions = parse_llm_response("DECOMPOSE: sin(pi*x)")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["action"], "decompose")
        self.assertEqual(actions[0]["spec"], "sin(pi*x)")

    def test_decompose_case_insensitive(self):
        actions = parse_llm_response("decompose: x*(1-x)")
        self.assertEqual(actions[0]["action"], "decompose")

    def test_compute(self):
        actions = parse_llm_response("COMPUTE: basis_info")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["action"], "compute")
        self.assertEqual(actions[0]["command"], "basis_info")

    def test_compute_with_args(self):
        actions = parse_llm_response("COMPUTE: eval_basis 0.1 0.5 0.9")
        self.assertEqual(actions[0]["action"], "compute")
        self.assertEqual(actions[0]["command"], "eval_basis 0.1 0.5 0.9")

    def test_compute_solve(self):
        actions = parse_llm_response("COMPUTE: solve")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["action"], "compute")
        self.assertEqual(actions[0]["command"], "solve")

    def test_mixed_actions(self):
        text = (
            "Let me get basis info and then decompose.\n\n"
            "COMPUTE: basis_info\n"
            "DECOMPOSE: sin(pi*x)\n"
            "QUERY: sin(2*pi*x)"
        )
        actions = parse_llm_response(text)
        computes = [a for a in actions if a["action"] == "compute"]
        decomps = [a for a in actions if a["action"] == "decompose"]
        queries = [a for a in actions if a["action"] == "query"]
        self.assertEqual(len(computes), 1)
        self.assertEqual(len(decomps), 1)
        self.assertEqual(len(queries), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)