"""Task-pairing and dependence checks for exploratory archive statistics."""

import unittest

import numpy as np

from scripts.analyze_archive_statistics import (
    exact_discordant_p, holm_adjust, paired_success_statistics,
    run_confidence_metrics, summarize_run_confidence, wilson_interval,
)


class TestPairedArchiveStatistics(unittest.TestCase):
    def test_pairs_are_joined_by_seed_not_input_order(self):
        result = paired_success_statistics(
            {2: 0, 1: 1}, {1: 1, 2: 1}, np.array([[0, 0], [1, 1]])
        )
        self.assertEqual(result["seeds"], [1, 2])
        self.assertEqual(result["first_only_successes"], 0)
        self.assertEqual(result["second_only_successes"], 1)
        self.assertEqual(result["success_rate_difference_first_minus_second"], -0.5)
        np.testing.assert_allclose(result["paired_task_bootstrap_95_interval"], [-0.975, -0.025])

    def test_unmatched_seeds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_success_statistics({1: 1, 2: 0}, {1: 1, 3: 0}, np.array([[0, 1]]))

    def test_exact_test_handles_zero_and_extreme_discordance(self):
        self.assertEqual(exact_discordant_p(0, 0), 1.0)
        self.assertEqual(exact_discordant_p(3, 0), 0.25)
        self.assertEqual(exact_discordant_p(0, 10), 2 / 1024)
        self.assertEqual(exact_discordant_p(3, 3), 1.0)

    def test_holm_is_monotone_in_sorted_p_and_preserves_input_order(self):
        np.testing.assert_allclose(holm_adjust([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])

    def test_wilson_retains_uncertainty_when_all_ten_tasks_succeed(self):
        lower, upper = wilson_interval(10, 10)
        self.assertAlmostEqual(lower, 0.7224672001371107)
        self.assertAlmostEqual(upper, 1.0)
        lower_zero, upper_zero = wilson_interval(0, 10)
        self.assertAlmostEqual(lower_zero, 0.0)
        self.assertAlmostEqual(upper_zero, 1 - lower)


class TestRunBalancedConfidence(unittest.TestCase):
    def test_replicating_reports_does_not_reweight_a_run(self):
        short = {1: [(0.9, 1)], 2: [(0.9, 0)]}
        duplicated = {1: [(0.9, 1)] * 100, 2: [(0.9, 0)]}
        a = summarize_run_confidence(short, np.random.default_rng(17), 200)
        b = summarize_run_confidence(duplicated, np.random.default_rng(17), 200)
        self.assertEqual(a["n_runs_with_paired_reports"], 2)
        self.assertEqual(b["coefficient_observations"], 101)
        for name in a["run_balanced"]:
            self.assertAlmostEqual(a["run_balanced"][name]["estimate"], b["run_balanced"][name]["estimate"])
            np.testing.assert_allclose(
                a["run_balanced"][name]["whole_run_bootstrap_95_interval"],
                b["run_balanced"][name]["whole_run_bootstrap_95_interval"],
            )
        self.assertAlmostEqual(a["run_balanced"]["brier_score"]["estimate"], 0.41)
        self.assertAlmostEqual(a["run_balanced"]["signed_calibration_in_the_large_gap"]["estimate"], 0.4)

    def test_bootstrap_draws_retain_all_blocks_within_a_run(self):
        # Both runs average to the same metrics despite opposite block outcomes.
        result = summarize_run_confidence(
            {1: [(0.8, 1), (0.8, 0)], 2: [(0.8, 0), (0.8, 1)]},
            np.random.default_rng(42), 200,
        )
        np.testing.assert_allclose(
            result["run_balanced"]["brier_score"]["whole_run_bootstrap_95_interval"],
            [0.34, 0.34],
        )

    def test_missing_reports_are_explicit_and_not_imputed(self):
        result = summarize_run_confidence({1: [], 2: [(1.0, 1)]}, np.random.default_rng(0), 10)
        self.assertEqual(result["seeds_without_paired_reports"], [1])
        self.assertEqual(result["n_runs_with_paired_reports"], 1)
        self.assertEqual(result["run_balanced"]["brier_score"]["estimate"], 0.0)
        empty = summarize_run_confidence({1: []}, np.random.default_rng(0), 10)
        self.assertEqual(empty["run_balanced"], {})
        self.assertIsNone(run_confidence_metrics([]))


if __name__ == "__main__":
    unittest.main()
