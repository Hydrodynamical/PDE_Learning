"""Scientific integrity checks for offline, hidden-truth-independent controls."""

import unittest
import gzip
import json
from pathlib import Path
import tempfile

import numpy as np

from basis import CoefficientFunction, LegendreBasis
from expression_parser import make_test_function_from_string
from main_loop import ProbeSession
from pde import EllipticPDE, solve_elliptic
from scripts.control_baselines import (
    aggregate, design_diagnostics, evaluate_policy, polynomial_expression,
    query_list, reconstruct_instance, solve_banded_equivalent, write_json,
)


class TestControlPolicies(unittest.TestCase):
    def test_queries_are_valid_unique_and_reproducible(self):
        for strategy in ("fixed_sine", "boundary_legendre", "random_mixed"):
            specs = query_list(strategy, 24, replicate=3)
            self.assertEqual(specs, query_list(strategy, 24, replicate=3))
            self.assertEqual(len(specs), 24)
            self.assertEqual(len(set(specs)), 24)
            self.assertFalse(any("j" in spec for spec in specs))
            for spec in specs:
                function, error = make_test_function_from_string(spec)
                self.assertIsNone(error, (spec, error))
                self.assertIsNotNone(function)
        self.assertNotEqual(query_list("random_mixed", 24, 2), query_list("random_mixed", 24, 3))

    def test_factored_legendre_matches_stable_recurrence(self):
        x = np.linspace(0, 1, 257)
        for degree in (0, 1, 5, 15, 23):
            function, error = make_test_function_from_string(polynomial_expression(degree))
            self.assertIsNone(error)
            expected = x * (1 - x) * np.polynomial.legendre.Legendre.basis(degree)(2 * x - 1)
            np.testing.assert_allclose(function(x), expected, atol=5e-14, rtol=1e-10)

    def test_policy_does_not_accept_instance_or_solution_information(self):
        import inspect
        self.assertEqual(tuple(inspect.signature(query_list).parameters),
                         ("strategy", "budget", "replicate", "control_seed"))


class TestNumericalControls(unittest.TestCase):
    def make_pde(self):
        basis = LegendreBasis(2)
        functions = [CoefficientFunction(basis, coeff) for coeff in
                     ([1.3, .1], [.2, -.1], [.5, .2], [1, -.4])]
        return basis, EllipticPDE(*functions, bc_left=.3, bc_right=-.2)

    def test_banded_system_matches_native_including_nonzero_boundaries(self):
        _, pde = self.make_pde()
        for n_grid in (3, 17, 101):
            dense = solve_elliptic(pde, n_grid)
            banded = solve_banded_equivalent(pde, n_grid)
            np.testing.assert_allclose(banded.u, dense.u, rtol=1e-12, atol=1e-13)
            self.assertLess(np.max(np.abs(banded.residual)), 1e-10)

    def test_reconstruction_rejects_changed_archived_truth(self):
        session = ProbeSession.from_difficulty("easy", seed=1)
        run = {"config": {"difficulty": "easy", "seed": 1},
               "ground_truth": {name + "_coeffs": getattr(session.pde, name).coeffs.tolist()
                                for name in ("a", "b", "c", "f")}}
        reconstruct_instance(run)
        run["ground_truth"]["a_coeffs"][0] += .001
        with self.assertRaisesRegex(ValueError, "exactly regenerate"):
            reconstruct_instance(run)

    def test_evaluation_respects_native_minimum_and_full_budget(self):
        basis, pde = self.make_pde()
        solution = solve_elliptic(pde, 101)
        session = ProbeSession(pde, solution, basis, "easy", 1, max_queries=12)
        truth = {name + "_coeffs": getattr(pde, name).coeffs.tolist() for name in ("a", "b", "c", "f")}
        checkpoints = evaluate_policy(session, query_list("fixed_sine", 12), truth)
        self.assertEqual([row["queries"] for row in checkpoints], list(range(8, 13)))
        self.assertEqual(session.queries_used, 12)
        errors = checkpoints[-1]["coefficient_errors"]
        self.assertAlmostEqual(errors["total"], sum(errors[name] for name in ("a", "b", "c", "f")))
        displayed = checkpoints[-1]["displayed_8dp_coefficient_errors"]
        self.assertLessEqual(abs(errors["total"] - displayed["total"]), 2.01e-8)

    def test_joint_rank_reveals_cross_block_dependence(self):
        history = [{"_G_diff": [1, 0], "_G_adv": [1, 0], "_G_react": [0, 1]},
                   {"_G_diff": [0, 1], "_G_adv": [0, 1], "_G_react": [1, 0]}]
        diagnostic = design_diagnostics(history)
        self.assertEqual(diagnostic["raw"]["rank"], 2)
        self.assertEqual(diagnostic["raw"]["n_columns"], 6)
        self.assertEqual(diagnostic["column_normalized"]["rank"], 2)

    def test_random_replicates_are_not_counted_as_additional_pdes(self):
        records = []
        for seed in (1, 2):
            for replicate in (0, 1):
                checkpoint = {"queries": 8, "coefficient_errors": {"total": .001 if replicate == 0 else .1},
                              "joint_design": {"raw": {"rank": 6, "n_columns": 6}}}
                records.append({"difficulty": "easy", "strategy": "random_mixed", "grid_factor": 1,
                                "seed": seed, "replicate": replicate, "checkpoints": [checkpoint]})
        summary, = aggregate(records)
        self.assertEqual(summary["n_instances"], 2)
        self.assertEqual(summary["n_replicates"], 2)
        self.assertEqual(summary["checkpoints"][0]["total_error"]["n"], 4)
        self.assertEqual(summary["checkpoints"][0]["total_error"]["success_e_lt_0.01"], 2)
        for instance in summary["per_instance"]:
            self.assertEqual(instance["full_budget"]["n"], 2)
            self.assertEqual(instance["full_budget"]["success_e_lt_0.01"], 1)

    def test_compressed_output_is_readable_and_deterministic(self):
        value = {"score": .001, "nested": {"n": 30}}
        with tempfile.TemporaryDirectory() as directory:
            plain, zipped = Path(directory) / "out.json", Path(directory) / "out.json.gz"
            write_json(plain, value)
            write_json(zipped, value)
            first = zipped.read_bytes()
            write_json(zipped, value)
            self.assertEqual(zipped.read_bytes(), first)
            self.assertEqual(json.loads(gzip.decompress(first)), json.loads(plain.read_text()))


if __name__ == "__main__":
    unittest.main()
