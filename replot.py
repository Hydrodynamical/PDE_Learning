"""
Regenerate all diagnostic plots from a run JSON file.

Usage:
    python replot.py run_extreme_s42_gemini-3-flash-preview_20260409_195716.json

Produces:
    *_dashboard.png   — coefficient recovery (true vs recovered, L¹ error)
    *_metrics.png     — behavioral + metacognitive profile (4-panel)
    *_auc.png         — error convergence curves
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

COLORS = {
    "a": "#e63946", "b": "#2a9d8f", "c": "#7b2d8e", "f": "#e9a820",
    "u": "#1d3557", "flux": "#457b9d",
}
FAMILY_COLORS = {
    "polynomial": "#6366f1", "trigonometric": "#f97316",
    "exponential": "#10b981", "localized": "#ec4899", "other": "#94a3b8",
}


def overlay_actions(ax, d, x_is_turns=False):
    """Add very light action-colored background strips to any panel with queries/turns on x-axis."""
    action_colors = {
        "query": "#3b82f6",
        "solve": "#16a34a",
        "check": "#f59e0b",
        "term_integrals": "#06b6d4",
        "eval_solution": "#a78bfa",
        "predict": "#ef4444",
    }
    turns = d.get("turns", [])
    for t in turns:
        content = (t.get("content", "") or "").lower()
        actions = t.get("parsed_actions", [])
        turn_num = t["turn"]

        if "query" in actions:
            continue  # queries are the default — don't overlay
        elif "predict" in actions:
            act = "predict"
        elif "compute" in actions:
            if "solve" in content and "eval" not in content:
                act = "solve"
            elif "check" in content or "verify" in content:
                act = "check"
            elif "term_integral" in content:
                act = "term_integrals"
            elif "eval_solution" in content:
                act = "eval_solution"
            else:
                continue
        else:
            continue

        color = action_colors.get(act)
        if not color:
            continue

        if x_is_turns:
            x_pos = turn_num
        else:
            qcount = 0
            for q in d.get("queries", []):
                if q["turn"] <= turn_num:
                    qcount = q["query_number"]
            x_pos = qcount

        ax.axvspan(x_pos - 0.4, x_pos + 0.4, alpha=0.12, color=color, zorder=0)


def load_run(path):
    with open(path) as f:
        return json.load(f)


def load_all_runs(directory):
    """Load all run JSON files from a directory, sorted by seed."""
    import glob
    paths = sorted(glob.glob(os.path.join(directory, "run_*.json")))
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    return runs


# ── Figure 1: Coefficient recovery dashboard ─────────────────────────────

def plot_dashboard(d, save_path=None):
    """True vs recovered coefficients with L¹ error and shaded gap."""
    from main_loop import ProbeSession
    from basis import CoefficientFunction

    cfg = d["config"]
    session = ProbeSession.from_difficulty(cfg["difficulty"], cfg["seed"], cfg["max_queries"])
    prediction = d["solves"][-1]["coefficients"]
    session.submit_prediction(prediction)

    pde = session.pde
    basis = session.basis
    sol = session.solution
    score = session.final_score
    x = np.linspace(*pde.domain, 300)

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Solution panel
    ax_u = fig.add_subplot(gs[0, 0])
    ax_u.plot(sol.x, sol.u, color=COLORS["u"], linewidth=2, label="$u(x)$")
    ax_u.plot(sol.x, sol.u_x(), color=COLORS["flux"], linewidth=1.2,
              linestyle="--", alpha=0.6, label="$u'(x)$")
    ax_u.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax_u.set_xlabel("$x$")
    ax_u.set_title("Solution", fontsize=11)
    ax_u.legend(fontsize=8)
    ax_u.set_xlim(pde.domain)

    # Coefficient panels
    panels = [
        (gs[0, 1], "a", pde.a, "Diffusion $a(x)$"),
        (gs[0, 2], "b", pde.b, "Advection $b(x)$"),
        (gs[1, 1], "c", pde.c, "Reaction $c(x)$"),
        (gs[1, 0], "f", pde.f, "Source $f(x)$"),
    ]
    for gs_pos, name, true_fn, title in panels:
        ax = fig.add_subplot(gs_pos)
        true_vals = true_fn(x)
        rec_fn = CoefficientFunction(basis, np.array(prediction[name]))
        rec_vals = rec_fn(x)
        l1_err = np.trapezoid(np.abs(true_vals - rec_vals), x)

        ax.plot(x, true_vals, color=COLORS[name], linewidth=2.2, label="true", zorder=3)
        ax.plot(x, rec_vals, color=COLORS[name], linewidth=1.5,
                linestyle="--", alpha=0.75, label="recovered", zorder=3)
        ax.fill_between(x, true_vals, rec_vals, alpha=0.18,
                        color=COLORS[name], zorder=2, label="error")
        ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
        ax.set_xlabel("$x$")
        ax.set_title(f"{title}   ($L^1$ = {l1_err:.4f})", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.set_xlim(pde.domain)

    # Score summary panel
    ax_s = fig.add_subplot(gs[1, 2])
    ax_s.axis("off")
    if score:
        lines = [
            f"Difficulty: {cfg['difficulty']}",
            f"Queries used: {d['behavioral_metrics']['queries_used']} / {cfg['max_queries']}",
            f"Basis: {basis.n_basis} Legendre polynomials",
            f"Unknowns: {cfg['unknowns']}",
            "",
            "Coefficient vector errors:",
            f"  a(x):  {score.coeff_error_a:.4f}",
            f"  b(x):  {score.coeff_error_b:.4f}",
            f"  c(x):  {score.coeff_error_c:.4f}",
            f"  f(x):   {score.coeff_error_f:.4f}",
            f"  total: {score.total_coeff_error:.4f}",
        ]
        ax_s.text(0.05, 0.95, "\n".join(lines), transform=ax_s.transAxes,
                  fontsize=9, verticalalignment="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa",
                            edgecolor="#dee2e6", alpha=0.9))

    model = cfg.get("model", "?")
    fig.suptitle(f"PDE Identification Results — {model}", fontsize=14, fontweight="bold", y=0.98)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Figure 2: Behavioral + metacognitive metrics ─────────────────────────

def plot_metrics(d, save_path=None):
    """Four-panel behavioral and metacognitive profile."""
    bm = d["behavioral_metrics"]
    mc = d["metacognitive"]
    cfg = d["config"]
    solves = d["solves"]
    model = cfg.get("model", "?")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Behavioral Profile — {model}  ({cfg['difficulty']}, seed {cfg['seed']})",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Error convergence across solves ──
    ax = axes[0, 0]
    if solves:
        solve_queries = [s["queries_at_solve"] or s["n_equations"] for s in solves]
        for coeff in ["a", "b", "c"]:
            errs = [s["coeff_errors"][coeff] for s in solves]
            ax.semilogy(solve_queries, errs, "o-", color=COLORS[coeff],
                        label=f"{coeff}(x)", markersize=4, linewidth=1.5)
        totals = [s["coeff_errors"]["total"] for s in solves]
        ax.semilogy(solve_queries, totals, "s--", color="black",
                    label="total", markersize=5, linewidth=1.5, alpha=0.7)
        ax.axhline(0.1, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Queries at solve")
        ax.set_ylabel("Coefficient error")
        ax.set_title("Error convergence", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)
        overlay_actions(ax, d)
        f_final = solves[-1]["coeff_errors"]["f"]
        ax.text(0.02, 0.02, f"f(x) error: {f_final:.2e} (not plotted)",
                transform=ax.transAxes, fontsize=7, color="gray",
                fontstyle="italic")

    # ── Panel 2: Confidence calibration + control efficiency ──
    ax = axes[0, 1]
    cr = mc.get("confidence_reports", [])
    ctrl = mc.get("control", [])

    if cr:
        qcounts = [c["query_count"] for c in cr]
        for coeff in ["a", "b", "c", "f"]:
            vals = [c[coeff] * 100 for c in cr]
            ax.plot(qcounts, vals, "o-", color=COLORS[coeff],
                    label=f"{coeff} confidence", markersize=3, linewidth=1, alpha=0.8)
        ax.set_ylabel("Stated confidence (%)", fontsize=9)
        ax.set_xlabel("Queries")
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title("Confidence reports", fontsize=10)
        ax.grid(True, alpha=0.2)
        overlay_actions(ax, d)

        if ctrl:
            # Show target coefficient as colored dots along bottom
            ctrl_k = [c["k"] for c in ctrl]
            targets = [c["j_star"] for c in ctrl]
            target_colors = [COLORS.get(t, "gray") for t in targets]
            ax.scatter(ctrl_k, [5] * len(ctrl_k), c=target_colors, s=20,
                       marker="s", edgecolors="none", alpha=0.8, zorder=5)
            ax.text(min(ctrl_k), 8, "target →", fontsize=6, color="gray")
    else:
        ax.text(0.5, 0.5, "No confidence data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="gray")
        ax.set_title("Confidence reports", fontsize=10)

    # ── Panel 3: Test function families ──
    ax = axes[1, 0]
    fc = bm.get("family_counts", {})
    families = [k for k, v in sorted(fc.items(), key=lambda x: -x[1]) if v > 0]
    counts = [fc[f] for f in families]
    fam_colors = [FAMILY_COLORS.get(f, "#94a3b8") for f in families]
    bars = ax.bar(families, counts, color=fam_colors, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=9)
    ent = bm.get("family_entropy_normalized", 0)
    ax.set_title(f"Test function families (entropy = {ent:.2f})", fontsize=10)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)

    # ── Panel 4: Conditioning by block ──
    ax = axes[1, 1]
    blocks = bm.get("query_space_blocks", {})
    block_names = ["diffusion", "advection", "reaction"]
    block_colors = [COLORS["a"], COLORS["b"], COLORS["c"]]
    kappas = [blocks.get(b, {}).get("condition", 0) for b in block_names]
    bars = ax.bar(block_names, kappas, color=block_colors, edgecolor="white", linewidth=0.5)
    for bar, k in zip(bars, kappas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"κ = {k:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Conditioning by block (higher = harder)", fontsize=10)
    ax.set_ylabel("Condition number κ")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Figure 3: AUC error curves ──────────────────────────────────────────

def plot_auc(d, save_path=None):
    """Per-coefficient error curves over queries."""
    curves = d.get("error_curves", [])
    cfg = d["config"]
    model = cfg.get("model", "?")

    if not curves:
        print("No error curve data available.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Coefficient recovery curves — {model}\n"
                 f"Mean error across solves (lower = better)",
                 fontsize=12, fontweight="bold")

    panels = [
        (axes[0, 0], "a", "Diffusion $a(x)$"),
        (axes[0, 1], "b", "Advection $b(x)$"),
        (axes[1, 0], "c", "Reaction $c(x)$"),
        (axes[1, 1], "f", "Source $f(x)$"),
    ]

    for ax, coeff, title in panels:
        qcounts = [c["query_count"] for c in curves]
        errs = [c[f"{coeff}_error"] for c in curves]
        errs_pos = [max(e, 1e-14) for e in errs]  # avoid log(0)

        mean_err = np.mean(errs_pos)

        ax.semilogy(qcounts, errs_pos, "o-", color=COLORS[coeff],
                    linewidth=2, markersize=4,
                    label=f"{model}  (mean={mean_err:.3f})")
        # Final point
        ax.plot(qcounts[-1], errs_pos[-1], "o", color=COLORS[coeff],
                markersize=8, zorder=5)
        ax.axhline(0.1, color="gray", linestyle=":", alpha=0.5, label="0.1 threshold")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Queries submitted")
        ax.set_ylabel("Mean absolute error")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Figure 4: Behavioral summary text ────────────────────────────────────

def print_summary(d):
    """Print a formatted text summary of all metrics."""
    cfg = d["config"]
    bm = d["behavioral_metrics"]
    mc = d["metacognitive"]
    res = d["results"]

    print(f"\n{'='*60}")
    print(f"  {cfg['model']}  |  {cfg['difficulty']}  |  seed {cfg['seed']}")
    print(f"{'='*60}")

    print(f"\n  Total error: {res['coefficient_errors']['total']:.6f}")
    for k in ["a", "b", "c", "f"]:
        e = res["coefficient_errors"][k]
        bar = "█" * max(1, int(min(e * 20, 30)))
        print(f"    {k}(x): {e:.6f}  {bar}")

    print(f"\n  Queries: {bm['queries_used']}/{cfg['max_queries']}  "
          f"Turns: {bm['total_turns']}  Solves: {bm['solve_count']}")
    print(f"  Checks: {bm['check_count']}  Term integrals: {bm['term_integral_count']}")
    print(f"  Wasted turns: {bm['wasted_turns']} ({bm['wasted_turn_fraction']*100:.0f}%)")
    print(f"  Duplicates: {bm['duplicate_queries']}  "
          f"Improvement: {bm['improvement_ratio']:.2f}×")

    fc = bm.get("family_counts", {})
    total_q = sum(fc.values())
    print(f"\n  Families (entropy={bm['family_entropy_normalized']:.2f}):")
    for fam, count in sorted(fc.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / total_q * 100
            bar = "█" * int(pct / 3)
            print(f"    {fam:>14}: {count:>3} ({pct:4.0f}%) {bar}")

    blocks = bm.get("query_space_blocks", {})
    print(f"\n  Conditioning:")
    for name in ["diffusion", "advection", "reaction"]:
        b = blocks.get(name, {})
        print(f"    {name:>10}: κ={b.get('condition',0):>6.2f}  "
              f"rank={b.get('effective_rank',0)}/{b.get('max_rank',0)}")

    print()


# ── Figure 4: Deep diagnostics (6-panel) ─────────────────────────────────

def plot_deep(d, save_path=None):
    """Six-panel deep diagnostic figure:
    Top-left:     Sigma curves (uncertainty evolution)
    Top-right:    Coefficient trajectories (small multiples)
    Middle-left:  SVD spectra per block
    Middle-right: RHS stem plot colored by family
    Bottom:       Action sequence timeline strip
    """
    from main_loop import count_atoms
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cfg = d["config"]
    mc = d["metacognitive"]
    bm = d["behavioral_metrics"]
    solves = d["solves"]
    queries = d["queries"]
    model = cfg.get("model", "?")
    n_basis = cfg["n_basis"]

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3,
                  height_ratios=[1, 0.25])
    fig.suptitle(f"Deep Diagnostics — {model}  ({cfg['difficulty']}, seed {cfg['seed']})",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Sigma curves (uncertainty evolution) ──
    ax = fig.add_subplot(gs[0, 0])
    sigma = mc.get("sigma_curves", [])
    if sigma:
        n_unknowns = cfg.get("unknowns", 0)
        sigma_filtered = [s for s in sigma if s["k"] >= n_unknowns]
        ks = [s["k"] for s in sigma_filtered]
        for coeff, key in [("a", "sigma_a"), ("b", "sigma_b"), ("c", "sigma_c")]:
            vals = [s[key] for s in sigma_filtered]
            ax.semilogy(ks, vals, "-", color=COLORS[coeff], linewidth=2,
                        label=f"σ_{coeff}", alpha=0.9)
        # Mark solves
        for s in solves:
            qk = s.get("queries_at_solve") or s["n_equations"]
            ax.axvline(qk, color="gray", linestyle=":", alpha=0.2, linewidth=0.8)
        ax.set_xlabel("Queries")
        ax.set_ylabel("Coefficient std. dev. σ")
        ax.set_title("Solution sensitivity (higher = less stable)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)
        overlay_actions(ax, d)
    else:
        ax.text(0.5, 0.5, "No sigma data", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        ax.set_title("Solution sensitivity (higher = less stable)", fontsize=10)

    # ── Panel 2: RHS stem plot ──
    ax_rhs = fig.add_subplot(gs[0, 1])
    if queries:
        q_nums = [q["query_number"] for q in queries]
        rhs_vals = [abs(q["rhs_value"]) for q in queries]
        families = [count_atoms(q["test_function"]) for q in queries]
        fam_colors = [FAMILY_COLORS.get(f, "#94a3b8") for f in families]

        ax_rhs.axhline(0, color="black", linewidth=0.3)
        for qn, rv, fc in zip(q_nums, rhs_vals, fam_colors):
            ax_rhs.vlines(qn, 0, rv, color=fc, alpha=0.7, linewidth=1.5)
        ax_rhs.scatter(q_nums, rhs_vals, c=fam_colors, s=14, zorder=3)
        for dot_x, dot_y, color in zip(q_nums, rhs_vals, fam_colors):
            ax_rhs.plot(dot_x, dot_y, "o", color=color, markersize=4, zorder=5)

        # Legend for families
        seen = {}
        for f in families:
            if f not in seen:
                seen[f] = True
        handles = [Line2D([0], [0], color=FAMILY_COLORS.get(f, "#94a3b8"),
                          marker="o", linestyle="-", markersize=4, label=f)
                   for f in seen]
        ax_rhs.legend(handles=handles, fontsize=7, loc="upper right")
        ax_rhs.set_xlabel("Query number")
        ax_rhs.set_ylabel("|∫fφ dx|")
        ax_rhs.set_title("Query information content", fontsize=10)
        ax_rhs.grid(True, alpha=0.15)
        overlay_actions(ax_rhs, d)
    else:
        ax_rhs.text(0.5, 0.5, "No query data", transform=ax_rhs.transAxes,
                    ha="center", va="center", color="gray")

    # ── Panel 3: Action sequence timeline ──
    ax_seq = fig.add_subplot(gs[1, :])
    turns = d.get("turns", [])
    if turns:
        action_colors = {
            "query": "#3b82f6",       # blue
            "solve": "#16a34a",       # green
            "check": "#f59e0b",       # gold
            "term_integrals": "#06b6d4",  # teal
            "eval_solution": "#a78bfa",   # light purple
            "predict": "#ef4444",     # red
            "reasoning": "#e5e7eb",   # light gray
        }
        action_labels = {
            "query": "Q", "solve": "S", "check": "C",
            "term_integrals": "T", "eval_solution": "E",
            "predict": "P", "reasoning": "·",
        }

        n_turns = len(turns)
        bar_width = 1.0

        for i, t in enumerate(turns):
            content = (t.get("content", "") or "").lower()
            actions = t.get("parsed_actions", [])

            # Determine primary action for this turn
            if "query" in actions:
                act = "query"
            elif "predict" in actions:
                act = "predict"
            elif "compute" in actions:
                if "solve" in content and "eval" not in content:
                    act = "solve"
                elif "check" in content or "verify" in content:
                    act = "check"
                elif "term_integral" in content:
                    act = "term_integrals"
                elif "eval_solution" in content:
                    act = "eval_solution"
                else:
                    act = "reasoning"
            else:
                act = "reasoning"

            color = action_colors.get(act, "#e5e7eb")
            ax_seq.barh(0, bar_width, left=i, height=0.7, color=color,
                        edgecolor="white", linewidth=0.3)

            # Add letter label for non-reasoning turns
            if act != "reasoning":
                label = action_labels.get(act, "?")
                ax_seq.text(i + 0.5, 0, label, ha="center", va="center",
                            fontsize=6 if n_turns > 50 else 8,
                            fontweight="bold", color="white" if act != "eval_solution" else "#333")

        # Legend
        legend_items = []
        seen_acts = set()
        for i, t in enumerate(turns):
            content = (t.get("content", "") or "").lower()
            actions = t.get("parsed_actions", [])
            if "query" in actions: seen_acts.add("query")
            elif "predict" in actions: seen_acts.add("predict")
            elif "compute" in actions:
                if "solve" in content and "eval" not in content: seen_acts.add("solve")
                elif "check" in content: seen_acts.add("check")
                elif "term_integral" in content: seen_acts.add("term_integrals")
                elif "eval_solution" in content: seen_acts.add("eval_solution")

        for act in ["query", "solve", "check", "term_integrals", "eval_solution", "predict", "reasoning"]:
            if act in seen_acts or act == "reasoning":
                legend_items.append(Patch(facecolor=action_colors[act], label=act))

        ax_seq.legend(handles=legend_items, fontsize=7, ncol=len(legend_items),
                      loc="upper center", bbox_to_anchor=(0.5, -0.3))
        ax_seq.set_xlim(-0.5, n_turns + 0.5)
        ax_seq.set_ylim(-0.5, 0.5)
        ax_seq.set_yticks([])
        ax_seq.set_xlabel("Turn")
        ax_seq.set_title("Action sequence", fontsize=10)
    else:
        ax_seq.text(0.5, 0.5, "No turn data", transform=ax_seq.transAxes,
                    ha="center", va="center", color="gray")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Multi-seed summary ───────────────────────────────────────────────────

def plot_multiseed(runs, save_path=None):
    """Five-panel multi-seed summary figure."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    if not runs:
        print("No runs to plot.")
        return None

    cfg = runs[0]["config"]
    model = cfg.get("model", "?")
    difficulty = cfg.get("difficulty", "?")
    n_seeds = len(runs)

    fig = plt.figure(figsize=(16, 13))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3,
                  height_ratios=[1, 1, 0.25])
    fig.suptitle(f"Multi-seed Summary — {model}  ({difficulty}, {n_seeds} seeds)",
                 fontsize=13, fontweight="bold")

    rng = np.random.default_rng(42)

    # ── Panel 1: Error convergence overlay (top-left) ──
    ax = fig.add_subplot(gs[0, 0])
    all_curves = {coeff: [] for coeff in ["a", "b", "c"]}
    all_curves["total"] = []

    for d in runs:
        solves = d.get("solves", [])
        if not solves:
            continue
        qcounts = [s.get("queries_at_solve") or s["n_equations"] for s in solves]
        for coeff in ["a", "b", "c"]:
            errs = [s["coeff_errors"][coeff] for s in solves]
            all_curves[coeff].append((qcounts, errs))
            ax.semilogy(qcounts, errs, "-", color=COLORS[coeff], alpha=0.2, linewidth=0.8)
        totals = [s["coeff_errors"]["total"] for s in solves]
        all_curves["total"].append((qcounts, totals))
        ax.semilogy(qcounts, totals, "--", color="black", alpha=0.15, linewidth=0.8)

    budget = cfg.get("max_queries", 48)
    n_unknowns = cfg.get("unknowns", 32)
    common_q = np.arange(n_unknowns, budget + 1)
    for coeff in ["a", "b", "c"]:
        interp_errs = []
        for qcounts, errs in all_curves[coeff]:
            interp_errs.append(np.interp(common_q, qcounts, errs,
                                         left=errs[0], right=errs[-1]))
        if interp_errs:
            ax.semilogy(common_q, np.median(interp_errs, axis=0), "-",
                        color=COLORS[coeff], linewidth=2.5, label=f"{coeff}(x)")

    interp_totals = []
    for qcounts, errs in all_curves["total"]:
        interp_totals.append(np.interp(common_q, qcounts, errs,
                                       left=errs[0], right=errs[-1]))
    if interp_totals:
        ax.semilogy(common_q, np.median(interp_totals, axis=0), "--",
                    color="black", linewidth=2.5, label="total")

    ax.axhline(0.1, color="gray", linestyle=":", alpha=0.5)
    for d in runs:
        total = d["results"]["coefficient_errors"]["total"]
        seed = d["config"]["seed"]
        ax.annotate(f"s{seed}", xy=(budget + 0.5, total),
                    fontsize=6, color="gray", alpha=0.6)
    ax.set_xlabel("Queries at solve")
    ax.set_ylabel("Coefficient error")
    ax.set_title("Error convergence (thin = per seed, thick = median)", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.15)

    # ── Panel 2: Behavioral fingerprint (top-right) ──
    ax = fig.add_subplot(gs[0, 1])

    metrics = [
        ("budget_utilization", "Budget utilization", 1.0),
        ("improvement_ratio", "Improvement ratio", None),
        ("family_entropy_normalized", "Family entropy", 1.0),
        ("solve_count", "Solves / run", None),
        ("check_count", "Checks / run", None),
        ("term_integral_count", "Term integrals / run", None),
        ("wasted_turn_fraction", "Wasted turn %", 1.0),
    ]

    y_pos = np.arange(len(metrics))
    bar_colors = ["#3b82f6", "#10b981", "#f97316", "#16a34a", "#f59e0b", "#06b6d4", "#ef4444"]

    for i, (key, label, max_val) in enumerate(metrics):
        if key == "budget_utilization":
            vals = [d.get("behavioral_metrics", {}).get(key,
                    d.get("behavioral_metrics", {}).get("queries_used", 0) /
                    d.get("config", {}).get("max_queries", 1))
                    for d in runs]
        else:
            vals = [d.get("behavioral_metrics", {}).get(key, 0) for d in runs]
        vals = [v if v is not None else 0 for v in vals]
        med = np.median(vals)

        if max_val is not None:
            bar_val = min(med / max_val, 1.0)
        else:
            scale = max(max(vals), 1)
            bar_val = med / scale

        ax.barh(i, bar_val, color=bar_colors[i % len(bar_colors)],
                alpha=0.7, height=0.6, edgecolor="white", linewidth=0.5)

        if key in ("wasted_turn_fraction", "budget_utilization"):
            val_text = f"{med:.0%}"
        elif key == "family_entropy_normalized":
            val_text = f"{med:.2f}"
        elif key == "improvement_ratio":
            val_text = f"{med:.1f}×"
        else:
            val_text = f"{med:.1f}"

        ax.text(bar_val + 0.02, i, val_text, va="center", fontsize=9, fontweight="bold")

        for v in vals:
            if max_val is not None:
                sv = min(v / max_val, 1.0)
            else:
                sv = v / scale if scale > 0 else 0
            ax.plot(sv, i, "o", color=bar_colors[i % len(bar_colors)],
                    alpha=0.3, markersize=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[1] for m in metrics], fontsize=9)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title("Behavioral fingerprint (median + seeds)", fontsize=10)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.3)

    # ── Panel 3: Family bars (middle-left) ──
    ax = fig.add_subplot(gs[1, 0])
    all_families = ["polynomial", "trigonometric", "exponential", "localized", "other"]
    per_seed_counts = {f: [] for f in all_families}
    for d in runs:
        fc = d.get("behavioral_metrics", {}).get("family_counts", {})
        for f in all_families:
            per_seed_counts[f].append(fc.get(f, 0))
    active = [f for f in all_families if sum(per_seed_counts[f]) > 0]
    x_pos = np.arange(len(active))
    medians = [np.median(per_seed_counts[f]) for f in active]
    fam_colors = [FAMILY_COLORS.get(f, "#94a3b8") for f in active]
    ax.bar(x_pos, medians, color=fam_colors, edgecolor="white", linewidth=0.5, alpha=0.8)
    for f_idx, f in enumerate(active):
        vals = per_seed_counts[f]
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(f_idx + jitter, vals, color=fam_colors[f_idx], s=15,
                   alpha=0.5, edgecolors="none", zorder=5)
        ax.text(f_idx, max(vals) + 0.5, f"{np.median(vals):.0f}",
                ha="center", va="bottom", fontsize=9)
    entropies = [d.get("behavioral_metrics", {}).get("family_entropy_normalized", 0)
                 for d in runs]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(active, rotation=30, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(f"Test function families (median entropy = {np.median(entropies):.2f})",
                 fontsize=10)

    # ── Panel 4: Conditioning bars (middle-right) ──
    ax = fig.add_subplot(gs[1, 1])
    block_names = ["diffusion", "advection", "reaction"]
    block_colors = [COLORS["a"], COLORS["b"], COLORS["c"]]
    per_seed_kappa = {b: [] for b in block_names}
    for d in runs:
        blocks = d.get("behavioral_metrics", {}).get("query_space_blocks", {})
        for b in block_names:
            per_seed_kappa[b].append(blocks.get(b, {}).get("condition", 0))
    x_pos = np.arange(len(block_names))
    medians = [np.median(per_seed_kappa[b]) for b in block_names]
    ax.bar(x_pos, medians, color=block_colors, edgecolor="white", linewidth=0.5, alpha=0.8)
    for b_idx, b in enumerate(block_names):
        vals = per_seed_kappa[b]
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(b_idx + jitter, vals, color=block_colors[b_idx], s=15,
                   alpha=0.5, edgecolors="none", zorder=5)
        ax.text(b_idx, max(vals) + 0.5, f"κ={np.median(vals):.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(block_names, fontsize=9)
    ax.set_ylabel("Condition number κ")
    ax.set_title("Conditioning by block (higher = harder)", fontsize=10)

    # ── Panel 5: Action timeline heatmap (bottom strip) ──
    ax_seq = fig.add_subplot(gs[2, :])
    action_colors_map = {
        "query":          [0.23, 0.51, 0.98],
        "solve":          [0.09, 0.64, 0.26],
        "check":          [0.96, 0.62, 0.04],
        "term_integrals": [0.02, 0.71, 0.83],
        "eval_solution":  [0.65, 0.55, 0.96],
        "predict":        [0.94, 0.27, 0.27],
        "reasoning":      [0.90, 0.91, 0.92],
    }
    max_turns = max(len(d.get("turns", [])) for d in runs)
    heatmap = np.ones((n_seeds, max_turns, 3)) * 0.95
    for row, d in enumerate(runs):
        for col, t in enumerate(d.get("turns", [])):
            content = (t.get("content", "") or "").lower()
            actions = t.get("parsed_actions", [])
            if "query" in actions:
                act = "query"
            elif "predict" in actions:
                act = "predict"
            elif "compute" in actions:
                if "solve" in content and "eval" not in content:
                    act = "solve"
                elif "check" in content or "verify" in content:
                    act = "check"
                elif "term_integral" in content:
                    act = "term_integrals"
                elif "eval_solution" in content:
                    act = "eval_solution"
                else:
                    act = "reasoning"
            else:
                act = "reasoning"
            heatmap[row, col] = action_colors_map.get(act, [0.9, 0.91, 0.92])
    ax_seq.imshow(heatmap, aspect="auto", interpolation="nearest")
    ax_seq.set_yticks(range(n_seeds))
    ax_seq.set_yticklabels([f"s{d['config']['seed']}" for d in runs], fontsize=8)
    ax_seq.set_xlabel("Turn")
    ax_seq.set_title("Action sequences", fontsize=10)
    legend_items = [
        Patch(color=action_colors_map["query"],          label="query"),
        Patch(color=action_colors_map["solve"],          label="solve"),
        Patch(color=action_colors_map["check"],          label="check"),
        Patch(color=action_colors_map["term_integrals"], label="term_integrals"),
        Patch(color=action_colors_map["predict"],        label="predict"),
    ]
    ax_seq.legend(handles=legend_items, fontsize=6, ncol=5, loc="upper right",
                  bbox_to_anchor=(1.0, -0.15))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python replot.py <run_*.json> [--no-show]     # single run")
        print("  python replot.py <directory/> [--no-show]      # multi-seed summary")
        sys.exit(1)

    path = sys.argv[1]
    show = "--no-show" not in sys.argv

    if os.path.isdir(path):
        runs = load_all_runs(path)
        if not runs:
            print(f"No run_*.json files found in {path}")
            sys.exit(1)
        print(f"Loaded {len(runs)} runs from {path}")
        save_path = os.path.join(path, "multiseed_summary.png")
        plot_multiseed(runs, save_path=save_path)
        print(f"  Saved: {save_path}")
    else:
        d = load_run(path)
        stem = path.replace(".json", "")
        print_summary(d)
        plot_dashboard(d, save_path=f"{stem}_dashboard.png")
        print(f"  Saved: {stem}_dashboard.png")
        plot_metrics(d, save_path=f"{stem}_metrics.png")
        print(f"  Saved: {stem}_metrics.png")
        plot_deep(d, save_path=f"{stem}_deep.png")
        print(f"  Saved: {stem}_deep.png")

    if show:
        plt.show()