"""
Behavioral cognitive profile figure for the PDE benchmark.

Usage:
    python plot_behavioral_profile.py runs/
    python plot_behavioral_profile.py runs/ --difficulties easy,medium,hard

Produces four figures, one per cognitive faculty, with easy/medium/hard as subplots:
  behavioral_executive_planning.png
  behavioral_in_context_learning.png
  behavioral_cognitive_flexibility.png
  behavioral_metacognitive_monitoring.png
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

# ── Config ────────────────────────────────────────────────────────────────

MODEL_ORDER = [
    "gemini-3.1-pro-preview",
    "claude-opus-4-6",
    "gpt-5.4",
    "gemini-3-flash-preview",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-6",
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

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def short_name(model):
    return MODEL_SHORT.get(model, model[:15])


def model_color(model):
    return MODEL_COLORS.get(model, "#6366f1")


def load_all_runs(directory):
    paths = sorted(glob.glob(os.path.join(directory, "**", "run_*.json"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(directory, "run_*.json")))
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def ordered_models(models):
    known = [m for m in MODEL_ORDER if m in models]
    extras = sorted(set(models) - set(MODEL_ORDER))
    return known + extras


def detect_difficulties(runs):
    """Return difficulties present in runs, in canonical order."""
    present = set(r.get("config", {}).get("difficulty") for r in runs)
    order = ["easy", "medium", "hard", "extreme"]
    return [d for d in order if d in present]


def confidence_hierarchy_accuracy(run):
    """Fraction of confidence reports where conf(a) > conf(b) > conf(c).

    Returns 0.0 if the model produced no confidence reports (monitoring failure).
    """
    mc = run.get("metacognitive", {})
    reports = mc.get("confidence_reports", [])
    if not reports:
        return 0.0
    correct = 0
    total = 0
    for cr in reports:
        a_conf = cr.get("a")
        b_conf = cr.get("b")
        c_conf = cr.get("c")
        if a_conf is not None and b_conf is not None and c_conf is not None:
            total += 1
            if a_conf > b_conf > c_conf:
                correct += 1
    return correct / total if total > 0 else 0.0


def compute_model_profiles(runs, difficulty=None):
    """Compute per-model median behavioral metrics.

    Returns dict: model -> {metric_name: (median, values_list)}
    """
    by_model = defaultdict(list)
    for r in runs:
        cfg = r.get("config", {})
        if difficulty and cfg.get("difficulty") != difficulty:
            continue
        by_model[cfg.get("model", "unknown")].append(r)

    profiles = {}
    for model, model_runs in by_model.items():
        solve_counts = [r.get("behavioral_metrics", {}).get("solve_count", 0)
                        for r in model_runs]
        ti_counts = [r.get("behavioral_metrics", {}).get("term_integral_count", 0)
                     for r in model_runs]
        ti_counts = [t if t is not None else 0 for t in ti_counts]
        check_counts = [r.get("behavioral_metrics", {}).get("check_count", 0)
                        for r in model_runs]
        check_counts = [c if c is not None else 0 for c in check_counts]
        imp_ratios = []
        for r in model_runs:
            ir = r.get("behavioral_metrics", {}).get("improvement_ratio", 1.0)
            if ir is None:
                ir = 1.0
            imp_ratios.append(ir)
        entropies = [r.get("behavioral_metrics", {}).get("family_entropy_normalized", 0)
                     for r in model_runs]
        entropies = [e if e is not None else 0 for e in entropies]
        conf_accs = [confidence_hierarchy_accuracy(r) for r in model_runs]

        profiles[model] = {
            "solve_count": (np.median(solve_counts), solve_counts),
            "ti_count": (np.median(ti_counts), ti_counts),
            "check_count": (np.median(check_counts), check_counts),
            "improvement_ratio": (np.median(imp_ratios), imp_ratios),
            "family_entropy": (np.median(entropies), entropies),
            "confidence_hierarchy": (np.median(conf_accs), conf_accs),
            "n_runs": len(model_runs),
        }

    return profiles


# ── Panel drawing helpers ─────────────────────────────────────────────────

def _draw_executive_planning(ax, profiles, models, title=""):
    """Stacked bar chart for Executive Planning into ax."""
    n_models = len(models)
    if n_models == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", color="gray")
        ax.set_title(title, fontsize=10, fontweight="bold", color="#374151", pad=8)
        return

    x = np.arange(n_models)
    bar_width = 0.7
    stack_colors = {"solves": "#16a34a", "checks": "#f59e0b", "term_integrals": "#06b6d4"}
    stack_metrics = [
        ("solve_count", "solves"),
        ("check_count", "checks"),
        ("ti_count", "term integrals"),
    ]

    bottoms = np.zeros(n_models)
    for metric_key, label in stack_metrics:
        meds = np.array([
            np.median(profiles.get(m, {}).get(metric_key, (0, [0]))[1] or [0])
            for m in models
        ])
        color_key = label.replace(" ", "_")
        color = stack_colors.get(color_key, "#94a3b8")
        ax.bar(x, meds, bottom=bottoms, color=color, edgecolor="white",
               linewidth=0.5, alpha=0.85, width=bar_width, label=label)
        for i, val in enumerate(meds):
            if val > 0.5:
                ax.text(x[i], bottoms[i] + val / 2, f"{val:.0f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold")
        bottoms += meds

    for i, total in enumerate(bottoms):
        ax.text(x[i], total + 0.3, f"{total:.0f}",
                ha="center", va="bottom", fontsize=7,
                fontweight="bold", color="#374151")

    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in models], fontsize=7)
    ax.set_ylabel("Actions per run (median)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#374151", pad=8)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(axis="y", alpha=0.15)


def _draw_metric_panel(ax, profiles, models, metric, ylabel,
                       title="", use_log=False, ylim=None):
    """Bar + jitter + median line for a single metric into ax."""
    rng = np.random.default_rng(42)
    n_models = len(models)

    if n_models == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", color="gray")
        ax.set_title(title, fontsize=10, fontweight="bold", color="#374151", pad=8)
        return

    x = np.arange(n_models)
    bar_width = 0.7
    means, medians, stds, colors = [], [], [], []

    for model in models:
        _, vals = profiles.get(model, {}).get(metric, (0, []))
        if vals:
            means.append(np.mean(vals))
            medians.append(np.median(vals))
            stds.append(np.std(vals))
        else:
            means.append(0); medians.append(0); stds.append(0)
        colors.append(model_color(model))

    bars = ax.bar(x, means, color=colors, edgecolor="white",
                  linewidth=0.5, alpha=0.85, width=bar_width,
                  yerr=stds, capsize=3,
                  error_kw=dict(linewidth=1, color="#374151", capthick=1))

    for i, med_val in enumerate(medians):
        ax.plot([x[i] - bar_width / 2.3, x[i] + bar_width / 2.3],
                [med_val, med_val], color="black", linewidth=1.8, zorder=6)

    for i, model in enumerate(models):
        _, vals = profiles.get(model, {}).get(metric, (0, []))
        if vals:
            jitter = rng.uniform(-0.2, 0.2, len(vals))
            ax.scatter(i + jitter, vals, color=colors[i], s=12,
                       alpha=0.4, edgecolors="none", zorder=5)

    for i, (bar, med_val, std_val) in enumerate(zip(bars, medians, stds)):
        if use_log and med_val > 0:
            ann = f"{med_val:.0f}×"
        elif metric == "confidence_hierarchy":
            ann = f"{med_val:.0%}"
        elif metric == "family_entropy":
            ann = f"{med_val:.2f}"
        else:
            ann = f"{med_val:.1f}"
        y_pos = means[i] + std_val
        if use_log:
            y_pos = max(y_pos, 0.5)
        ax.text(bar.get_x() + bar.get_width() / 2,
                y_pos + 0.02 * (ax.get_ylim()[1] if not use_log else 1),
                ann, ha="center", va="bottom", fontsize=7,
                fontweight="bold", color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in models], fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#374151", pad=8)
    if ylim:
        ax.set_ylim(ylim)
    if use_log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    ax.grid(axis="y", alpha=0.15)


# ── Main plotting entry point ─────────────────────────────────────────────

def plot_behavioral_profiles_by_difficulty(runs, difficulties=None, save_dir="."):
    """Produce 4 images, one per cognitive faculty, with one subplot per difficulty."""
    if difficulties is None:
        difficulties = detect_difficulties(runs)
    if not difficulties:
        difficulties = ["easy", "medium", "hard"]

    diff_profiles = {d: compute_model_profiles(runs, difficulty=d) for d in difficulties}

    # Union of all models across difficulties, in canonical order
    all_model_set = set()
    for p in diff_profiles.values():
        all_model_set.update(p.keys())
    all_models = ordered_models(list(all_model_set))

    n_diffs = len(difficulties)

    def models_for(diff):
        return [m for m in all_models if m in diff_profiles[diff]]

    def save_fig(fig, name):
        path = os.path.join(save_dir, name)
        fig.savefig(path)
        print(f"  Saved: {path}")
        plt.close(fig)

    # ── Executive Planning ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_diffs, figsize=(6 * n_diffs, 4.5))
    if n_diffs == 1:
        axes = [axes]
    fig.suptitle("Executive Planning", fontweight="bold", fontsize=13, y=1.02)
    for ax, diff in zip(axes, difficulties):
        _draw_executive_planning(
            ax, diff_profiles[diff], models_for(diff), title=diff.capitalize()
        )
    plt.tight_layout()
    save_fig(fig, "behavioral_executive_planning.png")

    # ── Simple metric panels ──────────────────────────────────────────────
    metric_configs = [
        ("improvement_ratio", "Improvement ratio", "In-Context Learning",
         True, None, "behavioral_in_context_learning.png"),
        ("family_entropy", "Family entropy H", "Cognitive Flexibility",
         False, (0, 1.0), "behavioral_cognitive_flexibility.png"),
        ("confidence_hierarchy", "Conf(a) > Conf(b) > Conf(c)", "Metacognitive Monitoring",
         False, (0, 1.15), "behavioral_metacognitive_monitoring.png"),
    ]
    for metric, ylabel, faculty, use_log, ylim, fname in metric_configs:
        fig, axes = plt.subplots(1, n_diffs, figsize=(6 * n_diffs, 4.5))
        if n_diffs == 1:
            axes = [axes]
        fig.suptitle(faculty, fontweight="bold", fontsize=13, y=1.02)
        for ax, diff in zip(axes, difficulties):
            _draw_metric_panel(
                ax, diff_profiles[diff], models_for(diff), metric, ylabel,
                title=diff.capitalize(), use_log=use_log, ylim=ylim,
            )
        plt.tight_layout()
        save_fig(fig, fname)


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    directory = sys.argv[1]
    difficulties = None
    if "--difficulties" in sys.argv:
        idx = sys.argv.index("--difficulties")
        difficulties = sys.argv[idx + 1].split(",")

    runs = load_all_runs(directory)
    if not runs:
        print(f"No run_*.json files found in {directory}")
        sys.exit(1)

    models = sorted(set(r["config"]["model"] for r in runs))
    diffs = sorted(set(r["config"].get("difficulty", "unknown") for r in runs))
    print(f"Loaded {len(runs)} runs: {len(models)} models × difficulties {diffs}")

    plot_behavioral_profiles_by_difficulty(runs, difficulties=difficulties, save_dir=directory)
    print("Done.")
