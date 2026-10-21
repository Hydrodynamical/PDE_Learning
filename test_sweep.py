"""Regression tests for sweep provenance and resumption; never call an API."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import generate_transcripts as sweep


def successful_record(**overrides):
    record = {
        "model": "gpt-4o", "provider": "openai", "difficulty": "easy", "seed": 42,
        "prompt_condition": "standard", "temperature": 0.0, "reasoning_effort": None,
        "total_coeff_error": 1.0, "queries_used": 8,
    }
    record.update(overrides)
    return record


class TestSweepResumption(unittest.TestCase):
    def run_sweep(self, existing, *options, fail=False):
        """Exercise the actual CLI's resume/merge logic with a fake single run."""
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "sweep_results.json"
            summary_path.write_text(json.dumps(existing))

            def fake_run(**kwargs):
                return successful_record(
                    model=kwargs["model"], provider=kwargs["provider"],
                    difficulty=kwargs["difficulty"], seed=kwargs["seed"],
                    prompt_condition="baseline" if kwargs["baseline"] else "standard",
                    temperature=kwargs["temperature"],
                    reasoning_effort=kwargs["reasoning_effort"], total_coeff_error=0.25,
                )

            argv = [
                "generate_transcripts.py", "--models", "gpt-4o",
                "--difficulties", "easy", "--seeds", "42", "--no-plot",
                "--output-dir", directory, *options,
            ]
            with patch("sys.argv", argv), patch.object(
                sweep, "run_single", side_effect=RuntimeError("fake failure") if fail else fake_run
            ) as runner, patch.object(sweep, "print_summary_table"), contextlib.redirect_stdout(io.StringIO()):
                sweep.main()

            return json.loads(summary_path.read_text()), runner.call_args_list

    def test_retry_skips_exact_success_including_zero_error(self):
        records, calls = self.run_sweep(
            [successful_record(total_coeff_error=0.0)], "--retry-failed"
        )
        self.assertEqual(calls, [])
        self.assertEqual(records[0]["total_coeff_error"], 0.0)

    def test_retry_does_not_reuse_other_prompt_temperature_or_reasoning(self):
        for options in (("--baseline",), ("--temperature", "0.5"), ("--reasoning-effort", "low")):
            with self.subTest(options=options):
                records, calls = self.run_sweep(
                    [successful_record()], "--retry-failed", *options
                )
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(records), 2)
                self.assertNotEqual(sweep.run_key(records[0]), sweep.run_key(records[1]))

    def test_legacy_summary_is_preserved_and_does_not_skip_current_run(self):
        for missing in ("prompt_condition", "temperature", "reasoning_effort"):
            with self.subTest(missing=missing):
                legacy = successful_record()
                del legacy[missing]
                records, calls = self.run_sweep([legacy], "--retry-failed")
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(records), 2)
                self.assertIn(legacy, records)

    def test_retry_replaces_matching_failure(self):
        failed = successful_record(total_coeff_error=None, error="previous failure")
        records, calls = self.run_sweep([failed], "--retry-failed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["total_coeff_error"], 0.25)
        self.assertNotIn("error", records[0])

    def test_merge_replaces_only_matching_configuration(self):
        old = [
            successful_record(),
            successful_record(prompt_condition="baseline"),
            successful_record(temperature=0.5),
            successful_record(reasoning_effort="low"),
        ]
        records, calls = self.run_sweep(old)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(records), 4)
        matching = [r for r in records if sweep.run_key(r) == sweep.run_key(old[0])]
        self.assertEqual(matching[0]["total_coeff_error"], 0.25)
        for unchanged in old[1:]:
            self.assertIn(unchanged, records)

    def test_failed_run_keeps_configuration_metadata(self):
        records, _ = self.run_sweep(
            [], "--baseline", "--temperature", "0.5", "--reasoning-effort", "low", fail=True
        )
        self.assertEqual(records[0]["prompt_condition"], "baseline")
        self.assertEqual(records[0]["temperature"], 0.5)
        self.assertEqual(records[0]["reasoning_effort"], "low")
        self.assertEqual(records[0]["error"], "fake failure")

    def test_unregistered_gemini_model_uses_google_backend(self):
        _, calls = self.run_sweep([], "--models", "gemini-example")
        self.assertEqual(calls[0].kwargs["provider"], "google")
        self.assertIsNone(calls[0].kwargs["reasoning_effort"])


class TestSingleRunMetadata(unittest.TestCase):
    def make_session(self):
        coefficient = SimpleNamespace(coeffs=SimpleNamespace(tolist=lambda: [1.0]))
        return SimpleNamespace(pde=SimpleNamespace(
            a=coefficient, b=coefficient, c=coefficient, f=coefficient,
            describe=lambda: "synthetic test PDE",
        ))

    def test_json_and_summary_record_requested_configuration(self):
        result = {"turns": 1, "queries_used": 0, "prediction_submitted": False}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sweep.ProbeSession, "from_difficulty", return_value=self.make_session()
        ), patch.object(sweep, "make_backend"), patch.object(
            sweep, "run_probe_session", return_value=result
        ), patch.object(sweep, "compute_extended_metrics", return_value={}):
            summary = sweep.run_single(
                "example-model", "openai", "easy", 42, directory,
                baseline=True, temperature=0.5, reasoning_effort="low", no_plot=True,
            )
            log = json.loads(Path(summary["log_path"]).read_text())

        for record in (summary, log["config"]):
            self.assertEqual(record["prompt_condition"], "baseline")
            self.assertEqual(record["temperature"], 0.5)
            self.assertEqual(record["reasoning_effort"], "low")

    def test_runs_in_same_second_keep_both_raw_logs(self):
        result = {"turns": 1, "queries_used": 0, "prediction_submitted": False}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sweep.ProbeSession, "from_difficulty", return_value=self.make_session()
        ), patch.object(sweep, "make_backend"), patch.object(
            sweep, "run_probe_session", return_value=result
        ), patch.object(sweep, "compute_extended_metrics", return_value={}), patch.object(
            sweep.time, "strftime", return_value="20261020_120000"
        ):
            standard = sweep.run_single(
                "example-model", "openai", "easy", 42, directory, no_plot=True,
            )
            baseline = sweep.run_single(
                "example-model", "openai", "easy", 42, directory, baseline=True,
                temperature=0.5, no_plot=True,
            )
            self.assertNotEqual(standard["log_path"], baseline["log_path"])
            self.assertEqual(len(list(Path(directory).glob("run_*.json"))), 2)
            standard_log = json.loads(Path(standard["log_path"]).read_text())
            baseline_log = json.loads(Path(baseline["log_path"]).read_text())

        self.assertEqual(standard_log["config"]["prompt_condition"], "standard")
        self.assertEqual(standard_log["config"]["temperature"], 0.0)
        self.assertEqual(baseline_log["config"]["prompt_condition"], "baseline")
        self.assertEqual(baseline_log["config"]["temperature"], 0.5)


if __name__ == "__main__":
    unittest.main()
