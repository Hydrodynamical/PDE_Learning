"""Offline tests for the response protocol used by ``main_loop.py``.

PREDICT submits the session's most recent COMPUTE: solve result. Coefficient
arrays and the legacy DECOMPOSE action belong to the archived protocol.
"""

import unittest

from main_loop import parse_probe_response


class TestProbeResponseParser(unittest.TestCase):
    def test_simple_query(self):
        self.assertEqual(
            parse_probe_response("QUERY: sin(pi*x)"),
            [{"action": "query", "spec": "sin(pi*x)"}],
        )

    def test_quoted_queries(self):
        for quote in ('"', "'", "`"):
            with self.subTest(quote=quote):
                actions = parse_probe_response(f"QUERY: {quote}x*(1-x){quote}")
                self.assertEqual(actions, [{"action": "query", "spec": "x*(1-x)"}])

    def test_commands_are_case_insensitive(self):
        self.assertEqual(
            parse_probe_response("query: sin(pi*x)\ncompute: solve\npredict:"),
            [
                {"action": "query", "spec": "sin(pi*x)"},
                {"action": "compute", "command": "solve"},
                {"action": "predict"},
            ],
        )

    def test_query_sanitizes_latex_and_implicit_multiplication(self):
        self.assertEqual(
            parse_probe_response(r"QUERY: $\sin(2\pi x)$"),
            [{"action": "query", "spec": "sin(2*pi*x)"}],
        )

    def test_multiple_queries_preserve_order(self):
        self.assertEqual(
            parse_probe_response("QUERY: sin(pi*x)\nQUERY: sin(2*pi*x)"),
            [
                {"action": "query", "spec": "sin(pi*x)"},
                {"action": "query", "spec": "sin(2*pi*x)"},
            ],
        )

    def test_compute_commands_preserve_arguments(self):
        commands = [
            "eval_solution 0.1 0.5 0.9",
            "solve",
            "check x*(1-x)",
            "term_integrals sin(2*pi*x)",
        ]
        for command in commands:
            with self.subTest(command=command):
                self.assertEqual(
                    parse_probe_response(f"COMPUTE: {command}"),
                    [{"action": "compute", "command": command}],
                )

    def test_predict_does_not_require_coefficient_arrays(self):
        self.assertEqual(parse_probe_response("PREDICT:"), [{"action": "predict"}])

    def test_predict_preserves_confidence_report_as_reasoning(self):
        confidence = "Confidence: a=90%, b=80%, c=70%, f=95%"
        self.assertEqual(
            parse_probe_response(f"PREDICT:\n{confidence}"),
            [{"action": "predict"}, {"action": "reasoning", "text": confidence}],
        )

    def test_legacy_arrays_do_not_override_solved_coefficients(self):
        actions = parse_probe_response("PREDICT:\na_coeffs = [99.0, 99.0]")
        self.assertEqual(
            actions,
            [
                {"action": "predict"},
                {"action": "reasoning", "text": "a_coeffs = [99.0, 99.0]"},
            ],
        )

    def test_legacy_decompose_is_reasoning(self):
        self.assertEqual(
            parse_probe_response("DECOMPOSE: sin(pi*x)"),
            [{"action": "reasoning", "text": "DECOMPOSE: sin(pi*x)"}],
        )

    def test_mixed_actions_and_reasoning_preserve_order(self):
        self.assertEqual(
            parse_probe_response(
                "First inspect the solution.\n\n"
                "COMPUTE: eval_solution 0.25 0.75\n"
                "QUERY: sin(pi*x)\n"
                "I will examine the result next."
            ),
            [
                {"action": "reasoning", "text": "First inspect the solution."},
                {"action": "compute", "command": "eval_solution 0.25 0.75"},
                {"action": "query", "spec": "sin(pi*x)"},
                {"action": "reasoning", "text": "I will examine the result next."},
            ],
        )

    def test_empty_and_whitespace_responses(self):
        for text in ("", " \n\t\n "):
            with self.subTest(text=text):
                self.assertEqual(parse_probe_response(text), [])

    def test_empty_query_and_compute_are_not_dispatched(self):
        self.assertEqual(
            parse_probe_response("QUERY:\nCOMPUTE:"),
            [
                {"action": "reasoning", "text": "QUERY:"},
                {"action": "reasoning", "text": "COMPUTE:"},
            ],
        )

    def test_commands_must_start_the_line(self):
        text = "I considered QUERY: sin(pi*x), but have not selected a probe."
        self.assertEqual(
            parse_probe_response(text), [{"action": "reasoning", "text": text}]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
