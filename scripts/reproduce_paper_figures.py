#!/usr/bin/env python3
"""Audit the archived Kaggle study and regenerate its publication figures offline.

Download separately (the script never makes network or model API calls)::

    curl -L --fail 'https://www.kaggle.com/api/v1/datasets/download/jkmiller/elliptic-differential-equation-llm-reasoning?datasetVersionNumber=1' -o /tmp/pde-runs-v1.zip
    python scripts/reproduce_paper_figures.py --archive /tmp/pde-runs-v1.zip

Only the standard library, NumPy, and Matplotlib are required. Run from any
directory; default output paths resolve relative to this repository. The input
archive is checked against paper_artifacts/data_manifest.json before parsing.
No archive member is extracted or executed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, median, pstdev
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIFFICULTIES = ("easy", "medium", "hard")
COEFFICIENTS = ("a", "b", "c", "f")
MODEL_LABELS = {
    "gemini-3.1-pro-preview": "Gemini 3.1\nPro",
    "claude-opus-4-6": "Claude\nOpus 4.6",
    "gpt-5.4": "GPT-5.4",
    "gemini-3-flash-preview": "Gemini 3\nFlash",
    "claude-sonnet-4-20250514": "Claude\nSonnet 4",
    "gpt-5.4-mini": "GPT-5.4\nmini",
}
MODEL_COLORS = ("#3274A1", "#C47E16", "#238B72", "#81A9D6", "#D4A329", "#76BEA4")
CONFIDENCE = re.compile(
    r"Confidence:\s*a=(\d+)%,\s*b=(\d+)%,\s*c=(\d+)%,\s*f=(\d+)%", re.I
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def describe(values):
    """Population SD describes the archived runs, not estimation uncertainty."""
    values = [float(v) for v in values]
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "population_sd": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def load_and_validate(archive, manifest):
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    require(digest == manifest["archive_sha256"], "Archive SHA-256 does not match the pinned dataset version.")
    with zipfile.ZipFile(archive) as source:
        require(len(source.infolist()) == manifest["archive_members"], "Unexpected archive member count.")
        names = sorted(n for n in source.namelist() if n.startswith("run_") and n.endswith(".json"))
        runs = [json.loads(source.read(n)) for n in names]
        sweep = json.loads(source.read("sweep_results.json"))
    require(len(runs) == manifest["expected_runs"], "Unexpected run count.")
    require(len(sweep) == len(runs), "Sweep summary count differs from run-log count.")
    require({r["config"]["model"] for r in runs} == set(manifest["models"]), "Unexpected model set.")
    require({r["config"]["difficulty"] for r in runs} == set(DIFFICULTIES), "Unexpected difficulty set.")
    keys = [(r["config"]["model"], r["config"]["difficulty"], r["config"]["seed"]) for r in runs]
    require(len(keys) == len(set(keys)), "Duplicate model/difficulty/seed record.")
    summaries = {(r["model"], r["difficulty"], r["seed"]): r for r in sweep}
    for run in runs:
        cfg = run["config"]
        settings = manifest["difficulty_settings"][cfg["difficulty"]]
        require(all(cfg[k] == v for k, v in settings.items()), "Run settings differ from the manifest.")
        require(cfg["prompt_condition"] == manifest["prompt_condition"], "Unexpected prompt condition.")
        expected_effort = "medium" if cfg["provider"] == "openai" else None
        require(cfg["reasoning_effort"] == expected_effort, "Unexpected reasoning-effort setting.")
        errors = run["results"]["coefficient_errors"]
        require(all(isinstance(errors[k], (int, float)) and math.isfinite(errors[k]) and errors[k] >= 0
                    for k in (*COEFFICIENTS, "total")), "A run lacks a finite nonnegative score.")
        require(math.isclose(sum(errors[k] for k in COEFFICIENTS), errors["total"], rel_tol=1e-12, abs_tol=1e-12),
                "Component errors do not sum to the reported total.")
        key = (cfg["model"], cfg["difficulty"], cfg["seed"])
        require(key in summaries and summaries[key]["total_coeff_error"] == errors["total"],
                "Sweep summary score disagrees with its run log.")
    for difficulty in DIFFICULTIES:
        for model in manifest["models"]:
            cell = [r for r in runs if r["config"]["difficulty"] == difficulty and r["config"]["model"] == model]
            require(sorted(r["config"]["seed"] for r in cell) == manifest["expected_seeds"],
                    "A model/difficulty cell does not contain the ten expected seeds.")
        for seed in manifest["expected_seeds"]:
            truths = {json.dumps(r["ground_truth"], sort_keys=True) for r in runs
                      if r["config"]["difficulty"] == difficulty and r["config"]["seed"] == seed}
            require(len(truths) == 1, "Models do not share ground truth for a seed/difficulty pair.")
    return runs


def causal_confidence_pairs(run):
    """Pair reports to previously returned solve outputs using transcript turns.

    Archived reports omit turn numbers, but their exact confidence strings occur
    in the assistant transcript in the same order. A solve in the report's own
    response is not yet observable to the model and is therefore excluded.
    """
    reports = run.get("metacognitive", {}).get("confidence_reports", [])
    occurrences = [(turn["turn"], match) for turn in run["turns"]
                   if turn.get("role") == "assistant"
                   for match in CONFIDENCE.finditer(turn.get("content", ""))]
    require(len(occurrences) == len(reports), "Confidence reports do not match transcript occurrence counts.")
    pairs, stats = [], Counter(reports=len(reports))
    for (turn_number, occurrence), report in zip(occurrences, reports):
        require([int(v) / 100 for v in occurrence.groups()] == [report[k] for k in COEFFICIENTS],
                "Archived confidence values disagree with the transcript.")
        eligible = [s for s in run.get("solves", [])
                    if s["turn"] < turn_number and (s.get("queries_at_solve") or s["n_equations"]) <= report["query_count"]]
        if not eligible:
            stats["reports_excluded_no_preceding_solve"] += 1
            continue
        solve = max(eligible, key=lambda s: (s["turn"], s.get("solve_number", 0)))
        stats["reports_paired"] += 1
        # Audit the legacy forward-looking matching and overlapping boundaries.
        legacy = next((s for s in run["solves"]
                       if (s.get("queries_at_solve") or s["n_equations"]) >= report["query_count"]), run["solves"][-1])
        if legacy.get("solve_number") != solve.get("solve_number"):
            stats["reports_with_different_solve_from_legacy"] += 1
        for coefficient in ("a", "b", "c"):
            confidence = float(report[coefficient])
            require(0 <= confidence <= 1, "Confidence is outside [0, 1].")
            error = float(solve["coeff_errors"][coefficient])
            pairs.append((confidence, int(error < 0.1)))
            if 0 < confidence < 1 and math.isclose(confidence * 10, round(confidence * 10), abs_tol=1e-9):
                stats["observations_on_shared_legacy_bin_boundaries"] += 1
    stats["coefficient_observations"] = len(pairs)
    return pairs, stats


def calibration_summary(pairs):
    """Disjoint bins [lo, hi), with 1 included in the final bin."""
    bins = [[] for _ in range(10)]
    for confidence, success in pairs:
        index = min(int(confidence * 10), 9)
        bins[index].append((confidence, success))
    return [{
        "lower": i / 10,
        "upper": (i + 1) / 10,
        "upper_inclusive": i == 9,
        "n": len(values),
        "successes": sum(v[1] for v in values),
        "mean_confidence": mean(v[0] for v in values) if values else None,
        "success_fraction": mean(v[1] for v in values) if values else None,
    } for i, values in enumerate(bins)]


def build_summary(runs, manifest):
    summary = {
        "schema_version": 1,
        "archive_sha256": manifest["archive_sha256"],
        "dataset_url": manifest["dataset_url"],
        "validation": {
            "runs": len(runs), "scored_runs": len(runs), "models": len(manifest["models"]),
            "difficulties": list(DIFFICULTIES), "seeds_per_cell": manifest["expected_seeds"],
            "shared_ground_truth_per_seed_difficulty": True,
            "sweep_summary_matches_all_run_scores": True,
            "attempts_and_retries_before_archive": "not recorded",
        },
        "settings": {
            "difficulty_settings": manifest["difficulty_settings"],
            "prompt_condition": manifest["prompt_condition"],
            "reasoning_effort": {"openai": "medium", "anthropic": None, "google": None},
            "run_timestamp_range": [min(r["config"]["timestamp"] for r in runs), max(r["config"]["timestamp"] for r in runs)],
            "run_timestamp_timezone": "not recorded",
        },
        "definitions": {
            "coefficient_error": "E_j = max_l abs(predicted_theta[j,l] - true_theta[j,l]); absolute coefficient error, not relative error",
            "total_error": "E = E_a + E_b + E_c + E_f; strict threshold comparison",
            "improvement_ratio": "first solve total error / last solve total error; missing values assigned 1 to retain legacy plotting convention",
            "family_entropy": "-sum(p_i log2(p_i)) / log2(5), over the final solve taxonomy: polynomial/trigonometric/exponential/localized/other; later queries are omitted",
            "action_stack": "separate per-action medians, summed; not the median total action count",
            "population_sd": "ddof=0 across the ten archived runs; descriptive variation, not a confidence interval",
            "calibration": "pool a/b/c reports; pair with latest solve on an earlier transcript turn with queries_at_solve <= report query count; omit reports without prior solve; event is E_j < 0.1",
            "calibration_bins": "disjoint [lo,hi), with 1 in the final bin; plot mean stated confidence against empirical success fraction; marker area in points squared is 5 + 1.1 times bin count",
        },
        "cells": [],
        "calibration_totals": {},
        "changes_from_legacy_assets": [
            "GPT-5.4 mini hard success at E < 0.01 is 2/10 in both run logs and sweep_results.json; the checked-in PNG displays 0/10.",
            "Calibration uses causally available solve outputs and disjoint confidence bins, replacing forward-looking solve matching and overlapping bin boundaries.",
            "Improvement figure shows individual runs and medians on a logarithmic axis; it omits arithmetic-mean bars and standard-deviation whiskers."
        ],
    }
    total_calibration = Counter()
    for difficulty in DIFFICULTIES:
        for model in manifest["models"]:
            cell_runs = [r for r in runs if r["config"]["difficulty"] == difficulty and r["config"]["model"] == model]
            errors = [r["results"]["coefficient_errors"]["total"] for r in cell_runs]
            metrics = [r["behavioral_metrics"] for r in cell_runs]
            improvement = [m.get("improvement_ratio") if m.get("improvement_ratio") is not None else 1.0 for m in metrics]
            pairs, counts = [], Counter()
            for run in cell_runs:
                run_pairs, run_counts = causal_confidence_pairs(run)
                pairs.extend(run_pairs)
                counts.update(run_counts)
            total_calibration.update(counts)
            summary["cells"].append({
                "model": model, "difficulty": difficulty, "n": len(cell_runs),
                "success": {str(t): {"count": sum(e < t for e in errors), "n": len(errors), "rate": sum(e < t for e in errors) / len(errors)} for t in (0.01, 0.1)},
                "total_error": describe(errors),
                "coefficient_errors": {k: describe([r["results"]["coefficient_errors"][k] for r in cell_runs]) for k in COEFFICIENTS},
                "solves": describe([m.get("solve_count", 0) or 0 for m in metrics]),
                "checks": describe([m.get("check_count", 0) or 0 for m in metrics]),
                "term_integrals": describe([m.get("term_integral_count", 0) or 0 for m in metrics]),
                "budget_utilization": describe([m["budget_utilization"] for m in metrics]),
                "improvement_ratio": describe(improvement),
                "improvement_ratio_imputed_runs": sum(m.get("improvement_ratio") is None for m in metrics),
                "family_entropy": describe([m.get("family_entropy_normalized", 0) or 0 for m in metrics]),
                "calibration_support": dict(counts),
                "calibration_bins": calibration_summary(pairs),
            })
    summary["calibration_totals"] = dict(total_calibration)
    return summary


def plot_figures(runs, summary, manifest, output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42,
    })
    output.mkdir(parents=True, exist_ok=True)
    models = manifest["models"]
    cells = {(c["model"], c["difficulty"]): c for c in summary["cells"]}
    labels = [MODEL_LABELS[m] for m in models]
    pdf_metadata = {"Creator": "scripts/reproduce_paper_figures.py", "CreationDate": None, "ModDate": None}

    def save(fig, name):
        fig.savefig(output / name, metadata=pdf_metadata)
        plt.close(fig)

    def three_panels(ylabel, *, top=0.96):
        fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.3), sharex=True)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=top, hspace=0.34)
        for index, (ax, difficulty) in enumerate(zip(axes, DIFFICULTIES)):
            cfg = manifest["difficulty_settings"][difficulty]
            ax.set_title(f"({chr(97 + index)}) {difficulty.capitalize()} ($N={cfg['n_basis']}$; query budget {cfg['max_queries']})", loc="left", pad=4)
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(models)), labels)
            ax.grid(axis="y", color="0.91", linewidth=0.6, zorder=0)
        return fig, axes

    # Two threshold views share a color scale and annotate exact numerators.
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.1))
    fig.subplots_adjust(left=0.26, right=0.90, top=0.94, bottom=0.09, hspace=0.33)
    for ax, threshold in zip(axes, (0.01, 0.1)):
        counts = np.array([[cells[(m, d)]["success"][str(threshold)]["count"] for d in DIFFICULTIES] for m in models])
        im = ax.imshow(counts / 10, cmap="Blues", norm=Normalize(0, 1), aspect="auto")
        ax.set_xticks(range(3), [d.capitalize() for d in DIFFICULTIES])
        ax.set_yticks(range(6), [MODEL_LABELS[m].replace("\n", " ") for m in models])
        ax.set_title(f"{'(a) Main threshold' if threshold == .01 else '(b) Sensitivity threshold'}: total error $E < {threshold}$", loc="left", pad=5)
        for row in range(6):
            for col in range(3):
                ax.text(col, row, f"{counts[row, col]}/10", ha="center", va="center", fontsize=8,
                        color="white" if counts[row, col] >= 7 else "#183047")
        ax.tick_params(length=0)
    color_ax = fig.add_axes([0.93, 0.22, 0.018, 0.56])
    fig.colorbar(im, cax=color_ax, ticks=[0, .5, 1])
    color_ax.set_title("Rate", fontsize=8, pad=5)
    save(fig, "paper_success.pdf")

    fig, axes = three_panels("Action count", top=0.92)
    action_names = (("solves", "Solve"), ("checks", "Check"), ("term_integrals", "Term integral"))
    action_colors = ("#329367", "#D5A229", "#35A9BC")
    for ax, difficulty in zip(axes, DIFFICULTIES):
        bottom = np.zeros(len(models))
        maximum_stack = max(sum(cells[(m, difficulty)][key]["median"] for key, _ in action_names) for m in models)
        for (key, label), color in zip(action_names, action_colors):
            values = np.array([cells[(m, difficulty)][key]["median"] for m in models])
            ax.bar(range(len(models)), values, bottom=bottom, color=color, edgecolor="white", linewidth=.5, label=label, zorder=3)
            for x, value, start in zip(range(len(models)), values, bottom):
                if value >= maximum_stack * .10:
                    ax.text(x, start + value / 2, f"{value:g}", ha="center", va="center", fontsize=7.5)
            bottom += values
        ax.set_ylim(0, max(bottom) * 1.13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    save(fig, "paper_actions.pdf")

    fig, axes = three_panels("First / last solve error")
    rng = np.random.default_rng(2026)
    for ax, difficulty in zip(axes, DIFFICULTIES):
        all_values = []
        for x, (model, color) in enumerate(zip(models, MODEL_COLORS)):
            values = [r["behavioral_metrics"].get("improvement_ratio") for r in runs
                      if r["config"]["model"] == model and r["config"]["difficulty"] == difficulty]
            values = np.array([v if v is not None else 1 for v in values])
            require(np.all(values > 0), "Log-scale improvement plot requires positive ratios.")
            all_values.extend(values)
            ax.scatter(x + rng.uniform(-.17, .17, len(values)), values, s=18, color=color, alpha=.7, linewidths=.3, edgecolors="white", zorder=3)
            med = float(np.median(values))
            ax.plot([x - .28, x + .28], [med, med], color="#202020", linewidth=1.8, zorder=4)
        ax.axhline(1, color="0.5", linestyle="--", linewidth=.7)
        ax.set_yscale("log")
        ax.set_ylim(min(.5, min(all_values) / 1.7), max(all_values) * 2)
    save(fig, "paper_improvement.pdf")

    fig, axes = three_panels("Normalized family entropy")
    rng = np.random.default_rng(2026)
    for ax, difficulty in zip(axes, DIFFICULTIES):
        for x, (model, color) in enumerate(zip(models, MODEL_COLORS)):
            values = [r["behavioral_metrics"].get("family_entropy_normalized", 0) or 0 for r in runs
                      if r["config"]["model"] == model and r["config"]["difficulty"] == difficulty]
            stats = cells[(model, difficulty)]["family_entropy"]
            ax.bar(x, stats["mean"], color=color, alpha=.7, zorder=2)
            ax.errorbar(x, stats["mean"], yerr=stats["population_sd"], color="#444444", capsize=2, linewidth=.8, zorder=3)
            ax.scatter(x + rng.uniform(-.17, .17, len(values)), values, s=13, color=color, alpha=.9, linewidths=.3, edgecolors="white", zorder=4)
            ax.plot([x - .28, x + .28], [stats["median"], stats["median"]], color="#202020", linewidth=1.7, zorder=5)
        ax.set_ylim(0, 1)
    save(fig, "paper_diversity.pdf")

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.97, top=.88, bottom=.08, hspace=.29)
    for index, (ax, difficulty) in enumerate(zip(axes, DIFFICULTIES)):
        support = sum(cells[(m, difficulty)]["calibration_support"].get("coefficient_observations", 0) for m in models)
        ax.set_title(f"({chr(97 + index)}) {difficulty.capitalize()} ({support} coefficient reports)", loc="left", pad=4)
        ax.plot([0, 1], [0, 1], "--", color="0.6", linewidth=.8, zorder=1)
        for model, color in zip(models, MODEL_COLORS):
            bins = [b for b in cells[(model, difficulty)]["calibration_bins"] if b["n"]]
            x = [b["mean_confidence"] for b in bins]
            y = [b["success_fraction"] for b in bins]
            ax.plot(x, y, color=color, linewidth=.9, alpha=.9)
            ax.scatter(x, y, s=[5 + b["n"] * 1.1 for b in bins], color=color, alpha=.75, edgecolor="white", linewidth=.4, zorder=3,
                       label=MODEL_LABELS[model].replace("\n", " "))
        ax.set_xlim(-.02, 1.02)
        ax.set_ylim(-.06, 1.06)
        ax.set_ylabel("Fraction with $E_j < 0.1$")
        ax.grid(color="0.92", linewidth=.6)
    axes[-1].set_xlabel("Mean stated confidence within bin")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    # Uniform legend markers retain model identity; areas in panels show support.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker="o", markersize=4, linewidth=.9) for c in MODEL_COLORS]
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.text(.5, .905, "Marker area increases with bin support; exact counts are in summary.json", ha="center", fontsize=7)
    save(fig, "paper_calibration.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, required=True, help="Downloaded, unmodified Kaggle dataset version 1 ZIP")
    parser.add_argument("--manifest", type=Path, default=ROOT / "paper_artifacts/data_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "images")
    parser.add_argument("--summary", type=Path, default=ROOT / "paper_artifacts/summary.json")
    parser.add_argument("--summary-only", action="store_true", help="Audit and write JSON without importing NumPy/Matplotlib")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    runs = load_and_validate(args.archive, manifest)
    summary = build_summary(runs, manifest)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(runs)} scored runs against the pinned archive SHA-256; wrote {args.summary}")
    print("Calibration support:", json.dumps(summary["calibration_totals"], sort_keys=True))
    if not args.summary_only:
        plot_figures(runs, summary, manifest, args.output_dir)
        print(f"Wrote five publication figures to {args.output_dir}")


if __name__ == "__main__":
    main()
