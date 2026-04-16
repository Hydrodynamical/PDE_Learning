"""
Cross-model hackathon figures for "Can Language Models Learn a PDE?"

Usage:
    python plot_hackathon.py <directory_of_jsons/>

Each figure is produced with easy/medium/hard as side-by-side subplots:

  hackathon_fig1_discrimination.png   — Box plot: total error by model
  hackathon_fig2_percoeff.png         — Per-coefficient error breakdown
  hackathon_fig3_convergence.png      — Error vs query count (learning dynamics)
  hackathon_fig6_conditioning.png     — Reaction κ vs c(x) error (causal mechanism)
  hackathon_fig_metacognition.png     — Confidence calibration + control efficiency
  hackathon_fig_calibration.png       — Standalone confidence calibration
"""
import os
import sys
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict

# ── Style ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODEL_ORDER = [
    "gemini-3.1-pro-preview",
    "claude-opus-4-6",
    "gpt-5.4",
    "gemini-3-flash-preview",
    "claude-sonnet-4-6",
    "gpt-5.4-mini",
    "gpt-4o",
]

MODEL_SHORT = {
    "gemini-3.1-pro-preview": "Gemini 3.1\nPro",
    "gemini-3-flash-preview": "Gemini 3\nFlash",
    "claude-opus-4-6": "Claude\nOpus 4.6",
    "claude-sonnet-4-6": "Claude\nSonnet 4.6",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4\nmini",
    "gpt-4o": "GPT-4o",
}

MODEL_COLORS = {
    "gemini-3.1-pro-preview": "#4285F4",
    "gemini-3-flash-preview": "#7BAAF7",
    "claude-opus-4-6": "#D97706",
    "claude-sonnet-4-6": "#F59E0B",
    "gpt-5.4": "#10B981",
    "gpt-5.4-mini": "#6EE7B7",
    "gpt-4o": "#34D399",
}

COEFF_COLORS = {
    "a": "#e63946", "b": "#2a9d8f", "c": "#7b2d8e", "f": "#e9a820",
}


def short_name(model):
    return MODEL_SHORT.get(model, model[:15])


def model_color(model):
    return MODEL_COLORS.get(model, "#6366f1")


# ── Data loading ──────────────────────────────────────────────────────────

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


def group_by_model(runs, difficulty=None):
    """Group runs by model, optionally filtering by difficulty. Returns {model: [runs]}."""
    grouped = defaultdict(list)
    for r in runs:
        cfg = r.get("config", {})
        if difficulty and cfg.get("difficulty") != difficulty:
            continue
        grouped[cfg.get("model", "unknown")].append(r)
    return dict(grouped)


def detect_difficulties(runs):
    """Return difficulties present in runs, in canonical order."""
    present = set(r.get("config", {}).get("difficulty") for r in runs)
    order = ["easy", "medium", "hard", "extreme"]
    return [d for d in order if d in present]


def ordered_models(grouped):
    """Return models in canonical order, then alphabetical for extras."""
    known = [m for m in MODEL_ORDER if m in grouped]
    extras = sorted(set(grouped.keys()) - set(MODEL_ORDER))
    return known + extras


# ── Panel drawing helpers ─────────────────────────────────────────────────

def _draw_discrimination(ax, grouped, title=""):
    """Box/strip plot of total coefficient error into ax."""
    models = ordered_models(grouped)
    if not models:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", color="gray")
        ax.set_title(title, fontweight="bold")
        return

    positions = np.arange(len(models))
    rng = np.random.default_rng(42)

    for i, model in enumerate(models):
        runs = grouped[model]
        errors = [r["results"]["coefficient_errors"]["total"] for r in runs]
        errors = [e if e is not None else 999 for e in errors]
        color = model_color(model)

        ax.boxplot(
            [errors], positions=[i], widths=0.45,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.3, edgecolor=color, linewidth=1.2),
            medianprops=dict(color=color, linewidth=2),
            whiskerprops=dict(color=color, linewidth=1),
            capprops=dict(color=color, linewidth=1),
        )
        jitter = rng.uniform(-0.12, 0.12, len(errors))
        ax.scatter(i + jitter, errors, c=color, s=35, zorder=5, alpha=0.8,
                   edgecolors="white", linewidths=0.5)
        med = np.median(errors)
        ax.text(i, max(errors) * 1.15 if max(errors) > 0.001 else med * 3,
                f"{med:.4f}", ha="center", va="bottom", fontsize=7.5,
                color=color, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([short_name(m) for m in models], fontsize=9)
    ax.set_ylabel("Total coefficient error (log)")
    ax.set_title(title, fontweight="bold")
    ax.axhline(0.1, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.axhline(0.01, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.grid(axis="y", alpha=0.15)


def _draw_percoeff(ax, grouped, title=""):
    """Grouped bar chart of per-coefficient error into ax."""
    models = ordered_models(grouped)
    if not models:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", color="gray")
        ax.set_title(title, fontweight="bold")
        return

    n_models = len(models)
    bar_width = 0.18
    x = np.arange(n_models)

    for ci, coeff in enumerate(["a", "b", "c", "f"]):
        means, stds = [], []
        for model in models:
            errs = [r["results"]["coefficient_errors"].get(coeff, 0) for r in grouped[model]]
            errs = [e if e is not None else 0 for e in errs]
            means.append(np.mean(errs))
            stds.append(np.std(errs))
        offset = (ci - 1.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds,
               color=COEFF_COLORS[coeff], alpha=0.8,
               edgecolor="white", linewidth=0.5,
               label=f"{coeff}(x)", capsize=2,
               error_kw=dict(linewidth=0.8))

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(m) for m in models], fontsize=9)
    ax.set_ylabel("Coefficient error (log)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(axis="y", alpha=0.15)


def _draw_convergence_panel(ax, grouped, coeff, title=""):
    """Error convergence for one coefficient (mean ± 1σ across seeds) into ax."""
    models = ordered_models(grouped)
    for model in models:
        runs = grouped[model]
        color = model_color(model)

        all_qcounts, all_errors = [], []
        for r in runs:
            solves = r.get("solves", [])
            if not solves:
                continue
            qc = [s.get("queries_at_solve") or s["n_equations"] for s in solves]
            if coeff == "total":
                errs = [s["coeff_errors"]["total"] for s in solves]
            else:
                errs = [s["coeff_errors"].get(coeff, 0) for s in solves]
            all_qcounts.append(qc)
            all_errors.append(errs)

        if not all_qcounts:
            continue

        q_min = min(qc[0] for qc in all_qcounts)
        q_max = max(qc[-1] for qc in all_qcounts)
        common_q = np.arange(q_min, q_max + 1)

        interped = []
        for qc, errs in zip(all_qcounts, all_errors):
            ext_q = qc + [q_max + 1]
            ext_e = errs + [errs[-1]]
            interped.append(np.interp(common_q, ext_q, ext_e))

        interped = np.array(interped)
        mean_err = np.mean(interped, axis=0)
        std_err = np.std(interped, axis=0)

        ax.semilogy(common_q, mean_err, "-", color=color, linewidth=2,
                    label=short_name(model).replace("\n", " "), alpha=0.9)
        if len(interped) > 1:
            ax.fill_between(common_q,
                            np.maximum(mean_err - std_err, 1e-14),
                            mean_err + std_err, color=color, alpha=0.1)

    ax.axhline(0.1, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Queries at solve")
    ax.set_ylabel("Coefficient error")
    ax.grid(True, alpha=0.15)


def _draw_conditioning_panel(ax, grouped, block, coeff, title=""):
    """Scatter: block condition number vs coefficient error into ax."""
    models = ordered_models(grouped)
    all_kappas, all_errors, all_colors = [], [], []

    for model in models:
        color = model_color(model)
        for r in grouped[model]:
            blocks = r.get("behavioral_metrics", {}).get("query_space_blocks", {})
            kappa = blocks.get(block, {}).get("condition", None)
            err = r["results"]["coefficient_errors"].get(coeff, None)
            if kappa is not None and err is not None and kappa > 0 and err > 0:
                all_kappas.append(kappa)
                all_errors.append(err)
                all_colors.append(color)

    if not all_kappas:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", color="gray")
        ax.set_title(title, fontsize=9)
        return

    ax.scatter(all_kappas, all_errors, c=all_colors, s=50, alpha=0.7,
               edgecolors="white", linewidths=0.5, zorder=5)

    log_k = np.log10(all_kappas)
    log_e = np.log10(all_errors)
    mask = np.isfinite(log_k) & np.isfinite(log_e)
    if mask.sum() > 2:
        coeffs = np.polyfit(log_k[mask], log_e[mask], 1)
        x_fit = np.linspace(min(log_k[mask]), max(log_k[mask]), 50)
        ax.plot(10**x_fit, 10**np.polyval(coeffs, x_fit), "--",
                color="gray", alpha=0.5, linewidth=1,
                label=f"slope={coeffs[0]:.2f}")
        ax.legend(fontsize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{block.capitalize()} block κ")
    ax.set_ylabel(f"{coeff}(x) error")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.15)


def _draw_calibration(ax, grouped, title=""):
    """Confidence calibration curve into ax."""
    models = ordered_models(grouped)
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for model in models:
        color = model_color(model)
        conf_vals, actual_correct = [], []

        for r in grouped[model]:
            mc = r.get("metacognitive", {})
            conf_reports = mc.get("confidence_reports", [])
            solves = r.get("solves", [])
            if not conf_reports or not solves:
                continue
            for cr in conf_reports:
                qk = cr["query_count"]
                matching_solve = None
                for s in solves:
                    sq = s.get("queries_at_solve") or s["n_equations"]
                    if sq >= qk:
                        matching_solve = s
                        break
                if not matching_solve:
                    matching_solve = solves[-1]
                for coeff in ["a", "b", "c"]:
                    conf = cr.get(coeff)
                    err = matching_solve["coeff_errors"].get(coeff, 999)
                    if conf is not None:
                        conf_vals.append(conf)
                        actual_correct.append(1 if err < 0.1 else 0)

        if len(conf_vals) < 5:
            continue

        conf_arr = np.array(conf_vals)
        correct_arr = np.array(actual_correct)
        bin_acc = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf_arr >= lo) & (conf_arr < hi + 1e-9)
            bin_acc.append(correct_arr[mask].mean() if mask.sum() > 0 else np.nan)

        valid = ~np.isnan(bin_acc)
        ax.plot(bin_centers[valid], np.array(bin_acc)[valid], "o-",
                color=color, linewidth=2, markersize=4, alpha=0.85,
                label=short_name(model).replace("\n", " "))

    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, linewidth=1,
            label="Perfect")
    ax.set_xlabel("Stated confidence")
    ax.set_ylabel("Fraction < 0.1 error")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(True, alpha=0.15)
    ax.set_aspect("equal")


def _draw_metacog_conf_sigma(ax, grouped, title=""):
    """Stated confidence vs true uncertainty σ scatter into ax."""
    models = ordered_models(grouped)
    for model in models:
        color = model_color(model)
        for r in grouped[model]:
            mc = r.get("metacognitive", {})
            conf_reports = mc.get("confidence_reports", [])
            sigma_curves = mc.get("sigma_curves", [])
            if not conf_reports or not sigma_curves:
                continue
            sigma_by_k = {s["k"]: s for s in sigma_curves}
            for cr in conf_reports:
                qk = cr["query_count"]
                closest_k = min(sigma_by_k.keys(), key=lambda k: abs(k - qk))
                sig = sigma_by_k[closest_k]
                for coeff in ["a", "b", "c"]:
                    stated_conf = cr.get(coeff)
                    true_sigma = sig.get(f"sigma_{coeff}")
                    if stated_conf is not None and true_sigma is not None and true_sigma > 0:
                        ax.scatter(stated_conf, true_sigma,
                                   c=COEFF_COLORS[coeff], s=20, alpha=0.5,
                                   edgecolors=color, linewidths=0.8)

    ax.set_yscale("log")
    ax.set_xlabel("Stated confidence")
    ax.set_ylabel("True σ (log)")
    ax.set_title(title, fontsize=9)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax.grid(True, alpha=0.15)
    coeff_handles = [Patch(facecolor=COEFF_COLORS[c], label=f"{c}(x)")
                     for c in ["a", "b", "c"]]
    ax.legend(handles=coeff_handles, fontsize=6, loc="upper right",
              title="coeff", title_fontsize=6)


def _draw_metacog_control(ax, grouped, title=""):
    """Control efficiency C_k box plot into ax."""
    models = ordered_models(grouped)
    model_Ck = {}
    for model in models:
        all_ck = []
        for r in grouped[model]:
            control = r.get("metacognitive", {}).get("control", [])
            all_ck.extend([c["C_k"] for c in control])
        if all_ck:
            model_Ck[model] = all_ck

    ck_models = [m for m in models if m in model_Ck]
    if not ck_models:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", color="gray")
        ax.set_title(title, fontsize=9)
        return

    positions = np.arange(len(ck_models))
    for i, model in enumerate(ck_models):
        color = model_color(model)
        vals = model_Ck[model]
        ax.boxplot(
            [vals], positions=[i], widths=0.45,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.3, edgecolor=color),
            medianprops=dict(color=color, linewidth=2),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
        )
        ax.text(i, 1.02, f"n={len(vals)}", ha="center", fontsize=6, color="gray")

    ax.set_xticks(positions)
    ax.set_xticklabels([short_name(m) for m in ck_models], fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("Control efficiency C_k")
    ax.set_title(title, fontsize=9)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)
    ax.grid(axis="y", alpha=0.15)


# ── Difficulty-stratified figure functions ────────────────────────────────

def plot_fig1_discrimination(all_runs, difficulties, save_path=None):
    """Box/strip plot of total error — one subplot per difficulty."""
    n_diffs = len(difficulties)
    fig, axes = plt.subplots(1, n_diffs, figsize=(5 * n_diffs, 4.5))
    if n_diffs == 1:
        axes = [axes]
    fig.suptitle("Cross-model discrimination: total coefficient error",
                 fontweight="bold", fontsize=12)
    for ax, diff in zip(axes, difficulties):
        _draw_discrimination(ax, group_by_model(all_runs, difficulty=diff),
                             title=diff.capitalize())
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


def plot_fig2_percoeff(all_runs, difficulties, save_path=None):
    """Per-coefficient error breakdown — one subplot per difficulty."""
    n_diffs = len(difficulties)
    fig, axes = plt.subplots(1, n_diffs, figsize=(6 * n_diffs, 4.5))
    if n_diffs == 1:
        axes = [axes]
    fig.suptitle("Per-coefficient error breakdown by model (mean ± 1σ)",
                 fontweight="bold", fontsize=12)
    for ax, diff in zip(axes, difficulties):
        _draw_percoeff(ax, group_by_model(all_runs, difficulty=diff),
                       title=diff.capitalize())
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


def plot_fig3_convergence(all_runs, difficulties, save_path=None):
    """Error convergence — 4 rows (a/b/c/total) × n_diffs cols (difficulties)."""
    coeff_panels = [
        ("a", "Diffusion a(x)"),
        ("b", "Advection b(x)"),
        ("c", "Reaction c(x)"),
        ("total", "Total error"),
    ]
    n_diffs = len(difficulties)
    fig, axes = plt.subplots(4, n_diffs, figsize=(5 * n_diffs, 14))
    if n_diffs == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("Error convergence across models (mean ± 1σ across seeds)",
                 fontweight="bold", fontsize=12)
    for row, (coeff, coeff_title) in enumerate(coeff_panels):
        for col, diff in enumerate(difficulties):
            ax = axes[row, col]
            _draw_convergence_panel(
                ax, group_by_model(all_runs, difficulty=diff), coeff,
                title=f"{coeff_title} — {diff.capitalize()}"
            )
    # Single legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(4, len(handles)), fontsize=8,
                   bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


def plot_fig6_conditioning(all_runs, difficulties, save_path=None):
    """Conditioning scatter — 2 rows (reaction/diffusion) × n_diffs cols."""
    block_panels = [
        ("reaction", "c", "Reaction block κ vs c(x) error"),
        ("diffusion", "a", "Diffusion block κ vs a(x) error"),
    ]
    n_diffs = len(difficulties)
    fig, axes = plt.subplots(2, n_diffs, figsize=(5 * n_diffs, 8))
    if n_diffs == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("Conditioning mechanism: block κ predicts coefficient error",
                 fontweight="bold", fontsize=12)
    for row, (block, coeff, panel_title) in enumerate(block_panels):
        for col, diff in enumerate(difficulties):
            _draw_conditioning_panel(
                axes[row, col], group_by_model(all_runs, difficulty=diff),
                block, coeff, title=f"{panel_title} — {diff.capitalize()}"
            )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


def plot_metacognition(all_runs, difficulties, save_path=None):
    """Metacognition — 3 rows (conf/σ, control, calibration) × n_diffs cols."""
    n_diffs = len(difficulties)
    fig, axes = plt.subplots(3, n_diffs, figsize=(5 * n_diffs, 12))
    if n_diffs == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("Metacognitive profile: monitoring accuracy and control efficiency",
                 fontweight="bold", fontsize=12)
    for col, diff in enumerate(difficulties):
        grouped = group_by_model(all_runs, difficulty=diff)
        _draw_metacog_conf_sigma(axes[0, col], grouped,
                                 title=f"Conf vs σ — {diff.capitalize()}")
        _draw_metacog_control(axes[1, col], grouped,
                              title=f"Control — {diff.capitalize()}")
        _draw_calibration(axes[2, col], grouped,
                          title=f"Calibration — {diff.capitalize()}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


def plot_calibration(all_runs, difficulties, save_path=None):
    """Standalone confidence calibration — one subplot per difficulty."""
    n_diffs = len(difficulties)
    fig, axes = plt.subplots(1, n_diffs, figsize=(5 * n_diffs, 5))
    if n_diffs == 1:
        axes = [axes]
    fig.suptitle("Confidence calibration by model",
                 fontweight="bold", fontsize=13)
    for ax, diff in zip(axes, difficulties):
        _draw_calibration(ax, group_by_model(all_runs, difficulty=diff),
                          title=diff.capitalize())
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

    all_runs = load_all_runs(directory)
    if not all_runs:
        print(f"No run_*.json files found in {directory}")
        sys.exit(1)

    difficulties = detect_difficulties(all_runs)
    if not difficulties:
        difficulties = ["easy", "medium", "hard"]

    models = sorted(set(r["config"]["model"] for r in all_runs))
    print(f"Loaded {len(all_runs)} runs across {len(models)} models, "
          f"difficulties: {difficulties}")
    for m in models:
        seeds = [r["config"]["seed"] for r in all_runs if r["config"]["model"] == m]
        diffs_for_m = sorted(set(
            r["config"].get("difficulty") for r in all_runs
            if r["config"]["model"] == m
        ))
        print(f"  {m}: {len(seeds)} runs, difficulties {diffs_for_m}")

    prefix = os.path.join(directory, "hackathon")

    plot_fig1_discrimination(all_runs, difficulties,
                             save_path=f"{prefix}_fig1_discrimination.png")
    plot_fig2_percoeff(all_runs, difficulties,
                       save_path=f"{prefix}_fig2_percoeff.png")
    plot_fig3_convergence(all_runs, difficulties,
                          save_path=f"{prefix}_fig3_convergence.png")
    plot_fig6_conditioning(all_runs, difficulties,
                           save_path=f"{prefix}_fig6_conditioning.png")
    plot_metacognition(all_runs, difficulties,
                       save_path=f"{prefix}_fig_metacognition.png")
    plot_calibration(all_runs, difficulties,
                     save_path=f"{prefix}_fig_calibration.png")

    print("\nDone. All figures saved.")
