"""End-to-end checks of the documented offline command; never call a provider."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent


class TestOfflineCLI(unittest.TestCase):
    def test_mock_produces_a_scored_log_for_each_difficulty(self):
        for difficulty, expected_queries in (("easy", 8), ("medium", 12), ("hard", 16), ("extreme", 32)):
            with self.subTest(difficulty=difficulty), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, str(ROOT / "main_loop.py"), "--mock", "--difficulty", difficulty,
                     "--seed", "42", "--no-plot", "--output-dir", directory],
                    cwd=ROOT, text=True, capture_output=True, timeout=60,
                    env={**os.environ, "MPLBACKEND": "Agg"},
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                logs = list(Path(directory).glob("*.json"))
                self.assertEqual(len(logs), 1)
                log = json.loads(logs[0].read_text())
                self.assertEqual(log["config"]["model"], "mock")
                self.assertEqual(log["config"]["provider"], "mock")
                self.assertEqual(log["config"]["seed"], 42)
                self.assertIn("temperature", log["config"])
                self.assertIn("min_queries", log["config"])
                self.assertEqual(len(log["queries"]), expected_queries)
                self.assertEqual(len(log["solves"]), 1)
                self.assertFalse(log["behavioral_metrics"]["auto_submitted"])
                self.assertTrue(math.isfinite(log["results"]["coefficient_errors"]["total"]))
                self.assertIn("predict", log["turns"][-1]["parsed_actions"])

    def test_mock_rejects_limits_that_cannot_complete_the_protocol(self):
        for limits in (("--max-queries", "7"), ("--max-turns", "10")):
            with self.subTest(limits=limits), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, str(ROOT / "main_loop.py"), "--mock", "--difficulty", "easy",
                     "--no-plot", "--output-dir", directory, *limits],
                    cwd=ROOT, text=True, capture_output=True, timeout=30,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("--mock needs at least", result.stderr)
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
