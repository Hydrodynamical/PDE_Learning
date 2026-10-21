"""Inclusion and fixed-plan checks for subsequent fresh-seed validation."""

import unittest

from scripts.validate_sine_generalization import execute_plan, summarize, validation_plan


class TestFreshSineValidation(unittest.TestCase):
    def test_plan_is_fixed_complete_and_has_no_archived_seed(self):
        plan = validation_plan()
        self.assertEqual(plan, validation_plan())
        self.assertEqual(len(plan), 300)
        self.assertEqual(len({(row["difficulty"], row["seed"]) for row in plan}), 300)
        for difficulty in ("easy", "medium", "hard"):
            self.assertEqual([row["seed"] for row in plan if row["difficulty"] == difficulty],
                             list(range(1001, 1101)))

    def test_zero_source_and_execution_failure_both_remain_in_primary_denominator(self):
        plan = [{"difficulty": "easy", "seed": seed} for seed in (1001, 1002, 1003)]
        called = []
        def runner(task):
            called.append(task["seed"])
            if task["seed"] == 1002:
                raise RuntimeError("Synthetic solver failure")
            error = 2.0 if task["seed"] == 1001 else .0001
            checkpoint = {"coefficient_errors": {"total": error}}
            return {**task, "status": "ok", "stage": "complete", "error": None,
                    "zero_source": task["seed"] == 1001, "zero_solution": task["seed"] == 1001,
                    "minimum_queries": checkpoint, "full_budget": checkpoint}
        outcomes = execute_plan(plan, runner=runner)
        self.assertEqual(called, [1001, 1002, 1003])
        self.assertEqual(len(outcomes), 3)
        cell, = summarize(outcomes)
        self.assertEqual(cell["n_planned_tasks"], 3)
        self.assertEqual(cell["status_counts"], {"ok": 2, "error": 1})
        self.assertEqual(cell["zero_source_tasks"], 1)
        for stage in ("minimum_queries", "full_budget"):
            self.assertEqual(cell[stage]["successes_e_lt_0.01"], 1)
            self.assertEqual(cell[stage]["denominator_all_planned_tasks"], 3)
            self.assertEqual(cell[stage]["failure_seeds_including_errors"], [1001, 1002])
            self.assertEqual(cell[stage]["zero_source_subset_descriptive"], {"n": 1, "successes": 0})


if __name__ == "__main__":
    unittest.main()
