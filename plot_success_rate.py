"""
Success-rate bar chart for the PDE benchmark hackathon submission.

Usage:
    python plot_success_rate.py runs/

Produces: runs/hackathon_success_rate.png

Loads ALL run JSONs (all difficulties), groups by model × difficulty,
and plots the fraction of seeds achieving error < threshold.
"""
import os
import sys
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Config (copy from plot_hackathon.py) ──────────────────────────────────

MODEL_ORDER = [
    "gemini-3.1-pro-preview",
    "claude-opus-4-6",
    "gpt-5.4",
    "gemini-3-flash-preview",
    "claude-sonnet-4-20250514",
    "gpt-5.4-mini",
    "gpt-4o",
]

MODEL_SHORT = {
    "gemini-3.1-pro-preview": "Gemini 3.1\nPro",
    "gemini-3-flash-preview": "Gemini 3\nFlash",
    "claude-opus-4-6": "Claude\nOpus 4.6",
    "claude-sonnet-4-20250514": "Claude\nSonnet 4",
    "claude-sonnet-4-6": "Claude\nSonnet 4",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4\nmini",
    "gpt-4o": "GPT-4o",
}

MODEL_COLORS = {
    "gemini-3.1-pro-preview": "#4285F4",
    "gemini-3-flash-preview": "#7BAAF7",
    "claude-opus-4-6": "#D97706",
    "claude-sonnet-4-20250514": "#F59E0B",
    "claude-sonnet-4-6": "#F59E0B",
    "gpt-5.4": "#10B981",
    "gpt-5.4-mini": "#6EE7B7",
    "gpt-4o": "#34D399",
}

DIFF_COLORS = {
    "easy": "#86efac",
    "medium": "#fbbf24",
    "hard": "#f87171",
    "extreme": "#c084fc",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def short_name(model):
    return MODEL_SHORT.get(model, model[:15])


def load_all_runs(directory):
    """Load all run JSONs recursively. Returns flat list."""
    paths = sorted(glob.glob(os.path.join(directory, "**", "run_*.json"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(directory, "run_*.json")))
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def plot_success_rate(runs, threshold=0.01, save_path=None):
    """Grouped bar chart: success rate (error < threshold) by model × difficulty.

    x-axis = models (ordered by overall success rate)
    Bars grouped by difficulty, colored by difficulty
    y-axis = fraction of seeds below threshold
    Annotations show n/N on each bar
    """
    # Group by (model, difficulty)
    data = defaultdict(list)
    for r in runs:
        cfg = r.get("config", {})
        model = cfg.get("model", "unknown")
        diff = cfg.get("difficulty", "unknown")
        total = r.get("results", {}).get("coefficient_errors", {}).get("total")
        if total is not None:
            data[(model, diff)].append(total)

    # Find all models and difficulties present
    all_models = sorted(set(m for m, d in data.keys()))
    all_diffs = [d for d in ["easy", "medium", "hard", "extreme"] if any(d == dd for _, dd in data.keys())]

    # Order models by overall success rate (descending)
    model_success = {}
    for model in all_models:
        all_errors = []
        for diff in all_diffs:
            all_errors.extend(data.get((model, diff), []))
        if all_errors:
            model_success[model] = sum(1 for e in all_errors if e < threshold) / len(all_errors)
        else:
            model_success[model] = 0

    # Use MODEL_ORDER for known models, then sort extras by success rate
    known = [m for m in MODEL_ORDER if m in all_models]
    extras = sorted([m for m in all_models if m not in MODEL_ORDER],
                    key=lambda m: -model_success.get(m, 0))
    models = known + extras

    # Layout
    n_models = len(models)
    n_diffs = len(all_diffs)
    bar_width = 0.8 / n_diffs
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.4), 5))

    for di, diff in enumerate(all_diffs):
        rates = []
        annotations = []
        for model in models:
            errors = data.get((model, diff), [])
            if errors:
                n_success = sum(1 for e in errors if e < threshold)
                rate = n_success / len(errors)
                rates.append(rate)
                annotations.append(f"{n_success}/{len(errors)}")
            else:
                rates.append(0)
                annotations.append("—")

        offset = (di - (n_diffs - 1) / 2) * bar_width
        color = DIFF_COLORS.get(diff, "#94a3b8")
        bars = ax.bar(x + offset, rates, bar_width * 0.85,
                      color=color, edgecolor="white", linewidth=0.5,
                      label=diff, alpha=0.85)

        # Annotate each bar with n/N
        for i, (bar, ann) in enumerate(zip(bars, annotations)):
            if ann != "—" and bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        ann, ha="center", va="bottom", fontsize=7, color="#374151")
            elif ann != "—":
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                        ann, ha="center", va="bottom", fontsize=7, color="#9ca3af")

    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in models], fontsize=9)
    ax.set_ylabel(f"Success rate (error < {threshold})")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Benchmark success rate by model and difficulty (threshold = {threshold})",
                 fontweight="bold", fontsize=12)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.legend(title="Difficulty", loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.15)

    # Summary annotation
    n_runs = len(runs)
    n_scored = sum(1 for r in runs if r.get("results", {}).get("coefficient_errors", {}).get("total") is not None)
    ax.text(0.01, 0.01, f"{n_scored} scored runs across {n_models} models",
            transform=ax.transAxes, fontsize=7, color="gray")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    directory = sys.argv[1]
    threshold = 0.01
    if "--threshold" in sys.argv:
        threshold = float(sys.argv[sys.argv.index("--threshold") + 1])

    runs = load_all_runs(directory)
    if not runs:
        print(f"No run_*.json files found in {directory}")
        sys.exit(1)

    # Summary
    models = sorted(set(r["config"]["model"] for r in runs))
    diffs = sorted(set(r["config"]["difficulty"] for r in runs))
    print(f"Loaded {len(runs)} runs: {len(models)} models × difficulties {diffs}")

    save_path = os.path.join(directory, f"hackathon_success_rate_{threshold}.png")
    plot_success_rate(runs, threshold=threshold, save_path=save_path)
    print("Done.")
