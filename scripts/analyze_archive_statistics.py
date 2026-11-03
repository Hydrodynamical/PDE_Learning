#!/usr/bin/env python3
"""Exploratory task-level uncertainty for the pinned, selected run archive.

No network or model calls are made. Example, from any working directory::

    python scripts/analyze_archive_statistics.py --archive /tmp/reasoning-v1.zip

This is uncertainty conditional on the scored archive. Unobserved attempts and
model sampling variability cannot be recovered by resampling saved outputs.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import platform
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

try:  # Support both direct execution and imports from the repository root.
    from .reproduce_paper_figures import (
        DIFFICULTIES, ROOT, causal_confidence_pairs, load_and_validate,
    )
except ImportError:
    from reproduce_paper_figures import (
        DIFFICULTIES, ROOT, causal_confidence_pairs, load_and_validate,
    )


BOOTSTRAP_RESAMPLES = 20_000
RANDOM_SEED = 20261020
CONFIDENCE_LEVEL = 0.95


def wilson_interval(successes, n, confidence_level=CONFIDENCE_LEVEL):
    """Wilson score interval without continuity correction; no pooled models."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Wilson intervals require 0 <= successes <= n and n > 0.")
    z = NormalDist().inv_cdf((1 + confidence_level) / 2)
    fraction = successes / n
    denominator = 1 + z * z / n
    center = (fraction + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def exact_discordant_p(first_only, second_only):
    """Two-sided exact paired binomial/McNemar test on discordant tasks."""
    if min(first_only, second_only) < 0 or any(
        int(value) != value for value in (first_only, second_only)
    ):
        raise ValueError("Discordant counts must be nonnegative integers.")
    n = int(first_only + second_only)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(int(min(first_only, second_only)) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm_adjust(p_values):
    """Holm step-down adjusted p-values in input order."""
    if any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("P-values must lie in [0, 1].")
    adjusted = [None] * len(p_values)
    previous = 0.0
    for order, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        previous = max(previous, (len(p_values) - order) * p_values[index])
        adjusted[index] = min(1.0, previous)
    return adjusted


def percentile_interval(values):
    return [float(value) for value in np.quantile(values, [0.025, 0.975], method="linear")]


def paired_success_statistics(first, second, bootstrap_indices):
    """Join successes by seed, retaining the pair in every bootstrap draw."""
    if not first or first.keys() != second.keys():
        raise ValueError("Paired comparisons require identical nonempty task-seed sets.")
    seeds = sorted(first)
    a = np.asarray([first[seed] for seed in seeds], dtype=float)
    b = np.asarray([second[seed] for seed in seeds], dtype=float)
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Success indicators must be binary.")
    difference = a - b
    first_only = int(np.sum((a == 1) & (b == 0)))
    second_only = int(np.sum((a == 0) & (b == 1)))
    bootstrap_means = difference[bootstrap_indices].mean(axis=1)
    return {
        "n_paired_tasks": len(seeds),
        "seeds": seeds,
        "first_successes": int(a.sum()),
        "second_successes": int(b.sum()),
        "first_only_successes": first_only,
        "second_only_successes": second_only,
        "both_successes": int(np.sum((a == 1) & (b == 1))),
        "neither_successes": int(np.sum((a == 0) & (b == 0))),
        "success_rate_difference_first_minus_second": float(difference.mean()),
        "paired_task_bootstrap_95_interval": percentile_interval(bootstrap_means),
        "exact_p_two_sided": exact_discordant_p(first_only, second_only),
    }


def run_confidence_metrics(pairs):
    """Reduce dependent reports/blocks to one equally weighted run summary."""
    if not pairs:
        return None
    values = np.asarray(pairs, dtype=float)
    confidence, outcome = values[:, 0], values[:, 1]
    if not ((0 <= confidence) & (confidence <= 1)).all() or not np.isin(outcome, [0, 1]).all():
        raise ValueError("Confidence must be in [0, 1] and outcomes must be binary.")
    return {
        "coefficient_observations": len(pairs),
        "brier_score": float(np.mean((confidence - outcome) ** 2)),
        "mean_confidence": float(confidence.mean()),
        "success_fraction": float(outcome.mean()),
        "signed_calibration_in_the_large_gap": float((confidence - outcome).mean()),
    }


def summarize_run_confidence(run_pairs, rng, resamples=BOOTSTRAP_RESAMPLES):
    """Resample whole supported runs, never reports or coefficient blocks.

    Each run contributes its within-run mean, so adding reports in one run does
    not increase that run's weight. Runs without causally paired reports are
    omitted; the estimand is conditional on reporting and archive inclusion.
    """
    supported = [(seed, run_confidence_metrics(pairs)) for seed, pairs in sorted(run_pairs.items()) if pairs]
    result = {
        "n_total_runs": len(run_pairs),
        "n_runs_with_paired_reports": len(supported),
        "seeds_with_paired_reports": [seed for seed, _ in supported],
        "seeds_without_paired_reports": sorted(seed for seed, pairs in run_pairs.items() if not pairs),
        "coefficient_observations": sum(row["coefficient_observations"] for _, row in supported),
        "runs": [{"seed": seed, **row} for seed, row in supported],
        "run_balanced": {},
    }
    if not supported:
        return result
    metrics = ["brier_score", "mean_confidence", "success_fraction", "signed_calibration_in_the_large_gap"]
    values = np.asarray([[row[name] for name in metrics] for _, row in supported])
    indices = rng.integers(0, len(supported), size=(resamples, len(supported)))
    bootstrap_means = values[indices].mean(axis=1)
    for index, name in enumerate(metrics):
        result["run_balanced"][name] = {
            "estimate": float(values[:, index].mean()),
            "whole_run_bootstrap_95_interval": percentile_interval(bootstrap_means[:, index]),
        }
    return result


def build_analysis(runs, manifest, resamples=BOOTSTRAP_RESAMPLES, seed=RANDOM_SEED):
    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive.")
    rng = np.random.default_rng(seed)
    analysis = {
        "schema_version": 1,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("scripts/analyze_archive_statistics.py",
                                       "scripts/reproduce_paper_figures.py", "requirements.txt")},
        "archive_sha256": manifest["archive_sha256"],
        "dataset_url": manifest["dataset_url"],
        "analysis_kind": "exploratory, conditional on inclusion in the scored archive",
        "settings": {
            "bootstrap_resamples": resamples,
            "random_seed": seed,
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_quantiles": [0.025, 0.975],
            "bootstrap_quantile_method": "linear",
            "success_thresholds": [0.01, 0.1],
            "paired_comparison_threshold": 0.01,
            "multiplicity": "Holm correction within all 15 model pairs separately for each difficulty; no correction across difficulties",
        },
        "uncertainty_units": {
            "success_intervals": "Wilson 95% score intervals over ten task seeds per model/difficulty, conditional on scored archive inclusion; illustrative task-sampling uncertainty under a binomial sampling model",
            "paired_comparisons": "Task seed within one difficulty. A draw retains the two models' outcomes on the same task; models are never treated as independent samples on that task.",
            "confidence": "One run on one task. First average all causally paired a/b/c observations within the run; then weight supported runs equally. Resample whole supported runs with their reports and coefficient blocks intact, conditional on having a paired report.",
            "calibration_gap": "Signed calibration-in-the-large gap: mean within-run confidence minus mean within-run correctness, then averaged equally across runs. Positive means overconfidence; zero alone does not imply calibration, and this is not ECE.",
            "confidence_event": "The preceding solve has block maximum absolute coefficient error E_q < 0.1; reports pair only to strictly earlier transcript turns; source f is excluded.",
            "scope": "Six model policies evaluated on 30 seed/difficulty tasks (ten per difficulty), with one retained scored record per model/task. No repeated model-sampling runs or complete attempt history are available.",
            "limitations": [
                "Intervals do not account for missing attempts, retries, archive selection, implementation drift, or model-sampling variability.",
                "Ten tasks per difficulty give coarse tests and potentially degenerate empirical bootstrap intervals; these intervals do not establish universal rates or model rankings.",
                "Seed labels are reused across difficulties. Analyses and resampling remain within difficulty and do not assume independence across difficulties.",
                "Paired percentile intervals are unadjusted exploratory intervals; only exact paired test p-values receive the stated Holm correction.",
                "These resampling distributions describe sensitivity to the ten observed tasks under exchangeable task-sampling assumptions; they do not prove the archive is representative of a target population.",
                "Report count is not an independent sample size. Runs without causally paired confidence reports are excluded from confidence estimands and their seeds are listed.",
            ],
        },
        "cells": [],
        "pairwise_success_comparisons": [],
    }
    for difficulty in DIFFICULTIES:
        by_model = {}
        for model in manifest["models"]:
            cell = [run for run in runs if run["config"]["difficulty"] == difficulty and run["config"]["model"] == model]
            by_seed = {run["config"]["seed"]: run for run in cell}
            if len(by_seed) != len(cell) or sorted(by_seed) != manifest["expected_seeds"]:
                raise ValueError("Cell has duplicate or missing task seeds.")
            by_model[model] = {
                task_seed: int(run["results"]["coefficient_errors"]["total"] < 0.01)
                for task_seed, run in by_seed.items()
            }
            success = {}
            for threshold in (0.01, 0.1):
                count = sum(run["results"]["coefficient_errors"]["total"] < threshold for run in cell)
                success[str(threshold)] = {
                    "count": count, "n": len(cell), "rate": count / len(cell),
                    "wilson_95_interval": wilson_interval(count, len(cell)),
                }
            confidence_pairs, support = {}, {"reports": 0, "reports_paired": 0, "reports_excluded_no_preceding_solve": 0}
            for task_seed, run in sorted(by_seed.items()):
                pairs, run_support = causal_confidence_pairs(run)
                confidence_pairs[task_seed] = pairs
                for name in support:
                    support[name] += run_support[name]
            analysis["cells"].append({
                "model": model, "difficulty": difficulty, "success": success,
                "confidence_report_support": support,
                "confidence": summarize_run_confidence(confidence_pairs, rng, resamples),
            })
        indices = rng.integers(0, len(manifest["expected_seeds"]), size=(resamples, len(manifest["expected_seeds"])))
        comparisons = [{
            "difficulty": difficulty,
            "first_model": first,
            "second_model": second,
            **paired_success_statistics(by_model[first], by_model[second], indices),
        } for first, second in combinations(manifest["models"], 2)]
        for comparison, adjusted in zip(comparisons, holm_adjust([row["exact_p_two_sided"] for row in comparisons])):
            comparison["holm_adjusted_p_within_difficulty"] = adjusted
        analysis["pairwise_success_comparisons"].extend(comparisons)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "paper_artifacts/data_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "paper_artifacts/statistical_analysis.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    runs = load_and_validate(args.archive, manifest)
    analysis = build_analysis(runs, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    print(f"Analyzed {len(runs)} scored runs; wrote {len(analysis['pairwise_success_comparisons'])} paired comparisons to {args.output}")


if __name__ == "__main__":
    main()
