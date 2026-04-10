"""
Diagnostics and visualization for the weak-form benchmark.

Generates plots for:
    - PDE coefficients a(x), b(x), c(x), f(x)
    - Solution u(x) and its derivatives
    - Test functions and their derivatives
    - Diagnostic summary panels

Also exports tabular data for LLM consumption.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional

from pde import EllipticPDE, EllipticSolution, solve_elliptic
from test_functions import TestFunction, standard_library


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    "u": "#2563eb",       # blue
    "a": "#dc2626",       # red
    "b": "#059669",       # green
    "c": "#7c3aed",       # purple
    "f": "#d97706",       # amber
    "flux": "#0891b2",    # cyan
    "phi": "#e11d48",     # rose
    "dphi": "#4f46e5",    # indigo
    "residual": "#6b7280", # gray
}


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_coefficients(pde: EllipticPDE, n_pts: int = 300,
                      ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Plot all PDE coefficient functions."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.figure

    x = np.linspace(*pde.domain, n_pts)

    for key, coeff in pde.coefficients().items():
        ax.plot(x, coeff(x), color=COLORS[key], label=f"${key}(x)$", linewidth=1.8)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel("$x$")
    ax.set_title("PDE coefficients")
    ax.legend(loc="best")
    ax.set_xlim(pde.domain)
    return fig


def plot_solution(sol: EllipticSolution, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Plot solution u(x) and its first derivative."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.figure

    ax.plot(sol.x, sol.u, color=COLORS["u"], linewidth=2, label="$u(x)$")
    ax.plot(sol.x, sol.u_x(), color=COLORS["flux"], linewidth=1.2,
            linestyle="--", label="$u'(x)$", alpha=0.7)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel("$x$")
    ax.set_title(f"Solution  (residual = {sol.max_residual():.2e})")
    ax.legend(loc="best")
    ax.set_xlim(sol.pde.domain)
    return fig


def plot_residual(sol: EllipticSolution, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Plot pointwise residual."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    else:
        fig = ax.figure

    x_interior = sol.x[1:-1]
    ax.semilogy(x_interior, np.abs(sol.residual), color=COLORS["residual"], linewidth=1.2)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|\\mathrm{residual}|$")
    ax.set_title(f"Pointwise residual (max = {sol.max_residual():.2e})")
    ax.set_xlim(sol.pde.domain)
    return fig


def plot_test_function(tf: TestFunction, n_pts: int = 300,
                       ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Plot a test function and its first derivative."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    else:
        fig = ax.figure

    x = np.linspace(*tf.domain, n_pts)
    ax.plot(x, tf(x), color=COLORS["phi"], linewidth=2, label=f"$\\varphi(x)$")
    ax.plot(x, tf.derivative(x, 1), color=COLORS["dphi"], linewidth=1.2,
            linestyle="--", label="$\\varphi'(x)$", alpha=0.7)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel("$x$")
    ax.set_title(f"Test function: {tf.spec}")
    ax.legend(loc="best")
    ax.set_xlim(tf.domain)
    return fig


def plot_test_function_library(library: list[TestFunction],
                               n_pts: int = 300) -> plt.Figure:
    """Plot an entire library of test functions in a grid."""
    n = len(library)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.5 * nrows),
                              sharex=True)
    axes = axes.flatten()

    x = np.linspace(*library[0].domain, n_pts)
    for i, tf in enumerate(library):
        ax = axes[i]
        ax.plot(x, tf(x), color=COLORS["phi"], linewidth=1.5)
        ax.set_title(tf.spec, fontsize=9)
        ax.axhline(0, color='gray', linewidth=0.3, linestyle='--')
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("$x$", fontsize=8)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Test Function Library", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(sol: EllipticSolution, test_fns: Optional[list[TestFunction]] = None,
                   n_pts: int = 300) -> plt.Figure:
    """
    Full diagnostic dashboard for a PDE + solution.

    Panels:
        Top-left:     Coefficients a, b, c, f
        Top-right:    Solution u(x) and u'(x)
        Bottom-left:  Residual
        Bottom-right: Test functions (first few)
    """
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_coefficients(sol.pde, n_pts=n_pts, ax=ax1)
    plot_solution(sol, ax=ax2)
    plot_residual(sol, ax=ax3)

    # Test functions panel
    if test_fns:
        x = np.linspace(*sol.pde.domain, n_pts)
        cmap = plt.cm.Set2
        n_show = min(6, len(test_fns))
        for i in range(n_show):
            color = cmap(i / max(n_show - 1, 1))
            ax4.plot(x, test_fns[i](x), linewidth=1.5, color=color,
                     label=test_fns[i].spec)
        ax4.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax4.set_xlabel("$x$")
        ax4.set_title("Test functions (sample)")
        ax4.legend(fontsize=7, loc="best")
        ax4.set_xlim(sol.pde.domain)
    else:
        ax4.text(0.5, 0.5, "No test functions provided",
                 ha='center', va='center', transform=ax4.transAxes)

    # Add PDE description as text
    fig.suptitle(sol.pde.describe().split('\n')[0], fontsize=12)

    return fig


# ---------------------------------------------------------------------------
# Data export for LLM consumption
# ---------------------------------------------------------------------------

def solution_to_table(sol: EllipticSolution, stride: int = 1) -> str:
    """
    Export solution diagnostics as a tab-separated table string.

    This is what an LLM would receive as "the data".
    stride controls downsampling (stride=10 → every 10th point).
    """
    diag = sol.diagnostics_table()
    keys = ["x", "u", "u_x", "a", "f"]  # what the LLM sees
    header = "\t".join(keys)
    rows = []
    for i in range(0, len(diag["x"]), stride):
        vals = [f"{diag[k][i]:.6f}" for k in keys]
        rows.append("\t".join(vals))
    return header + "\n" + "\n".join(rows)


def test_function_response_table(
    sol: EllipticSolution,
    test_fns: list[TestFunction],
    n_pts: int = 300,
    stride: int = 1,
) -> str:
    """
    For each test function, show φ(x) values alongside u(x).
    This gives the LLM the raw data to reason about.
    """
    x = np.linspace(*sol.pde.domain, n_pts)
    u_interp = np.interp(x, sol.x, sol.u)

    header = "x\tu(x)" + "".join(f"\tphi_{i}({tf.spec})" for i, tf in enumerate(test_fns))
    rows = []
    for j in range(0, len(x), stride):
        vals = [f"{x[j]:.6f}", f"{u_interp[j]:.6f}"]
        for tf in test_fns:
            vals.append(f"{tf(np.array([x[j]]))[0]:.6f}")
        rows.append("\t".join(vals))
    return header + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# Weak-form diagnostics
# ---------------------------------------------------------------------------

def plot_weak_form_battery(results, ax=None):
    """
    Bar chart showing the per-term decomposition for each test function.
    Each group: diffusion + advection + reaction = lhs ≈ rhs.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    else:
        fig = ax.figure

    specs = [r.spec for r in results]
    diffs = [r.diffusion for r in results]
    advs = [r.advection for r in results]
    reacts = [r.reaction for r in results]
    rhs_vals = [r.rhs for r in results]

    x_pos = np.arange(len(results))
    w = 0.35

    ax.bar(x_pos - w / 2, diffs, w * 0.9, label="diffusion", color=COLORS["a"], alpha=0.8)
    ax.bar(x_pos - w / 2, advs, w * 0.9, bottom=diffs, label="advection",
           color=COLORS["b"], alpha=0.8)
    ax.bar(x_pos - w / 2, reacts, w * 0.9,
           bottom=[d + a for d, a in zip(diffs, advs)], label="reaction",
           color=COLORS["c"], alpha=0.8)
    ax.bar(x_pos + w / 2, rhs_vals, w * 0.9, label="rhs (integral f phi)",
           color=COLORS["f"], alpha=0.6, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(specs, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8)
    ax.set_title("Weak-form decomposition per test function")
    ax.set_ylabel("Integral value")
    fig.tight_layout()
    return fig


def plot_weak_form_residuals(results, ax=None):
    """Bar chart of |lhs - rhs| for each test function (log scale)."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    else:
        fig = ax.figure

    specs = [r.spec for r in results]
    residuals = [r.residual for r in results]

    x_pos = np.arange(len(results))
    ax.bar(x_pos, residuals, color=COLORS["residual"], alpha=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(specs, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("|LHS - RHS|")
    ax.set_title("Weak-form residual per test function")
    fig.tight_layout()
    return fig


def plot_recovery_comparison(pde, recovered_coeffs, basis, n_pts=300):
    """
    Plot true vs recovered coefficient functions a(x), b(x), c(x).

    recovered_coeffs: dict with keys 'a', 'b', 'c' mapping to coefficient arrays.
    """
    from basis import CoefficientFunction

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    x = np.linspace(*pde.domain, n_pts)

    for ax, (name, true_fn) in zip(axes, [("a", pde.a), ("b", pde.b), ("c", pde.c)]):
        true_vals = true_fn(x)
        rec_fn = CoefficientFunction(basis, recovered_coeffs[name])
        rec_vals = rec_fn(x)

        ax.plot(x, true_vals, color=COLORS[name], linewidth=2, label=f"true {name}(x)")
        ax.plot(x, rec_vals, color=COLORS[name], linewidth=1.5, linestyle="--",
                alpha=0.7, label=f"recovered {name}(x)")
        ax.fill_between(x, true_vals, rec_vals, alpha=0.15, color=COLORS[name])
        ax.legend(fontsize=8)
        ax.set_xlabel("$x$")
        ax.set_title(f"{name}(x): max err = {np.max(np.abs(true_vals - rec_vals)):.4f}")

    fig.suptitle("Coefficient Recovery", fontsize=13)
    fig.tight_layout()
    return fig


def plot_convergence_study(pde, basis, grid_sizes=None, test_fns=None):
    """
    How does coefficient recovery improve with grid refinement?
    Plots max recovery error vs grid size on log-log scale.
    """
    from pde import solve_elliptic
    from weak_form import assemble_linear_system
    from test_functions import standard_library

    if grid_sizes is None:
        grid_sizes = [101, 201, 501, 1001, 2001, 4001]
    if test_fns is None:
        test_fns = standard_library(pde.domain)

    errors_a, errors_b, errors_c, hs = [], [], [], []

    for n in grid_sizes:
        sol = solve_elliptic(pde, n_grid=n)
        system = assemble_linear_system(sol, basis, test_fns)
        A_lhs, A_rhs = system.split_matrix()
        rhs_vec = A_rhs @ pde.f.coeffs
        theta_rec, _, _, _ = np.linalg.lstsq(A_lhs, rhs_vec, rcond=None)

        n_b = basis.n_basis
        errors_a.append(np.max(np.abs(theta_rec[:n_b] - pde.a.coeffs)))
        errors_b.append(np.max(np.abs(theta_rec[n_b:2*n_b] - pde.b.coeffs)))
        errors_c.append(np.max(np.abs(theta_rec[2*n_b:3*n_b] - pde.c.coeffs)))
        hs.append(1.0 / (n - 1))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    hs = np.array(hs)

    ax.loglog(hs, errors_a, "o-", color=COLORS["a"], label="a(x)", linewidth=1.5)
    ax.loglog(hs, errors_b, "s-", color=COLORS["b"], label="b(x)", linewidth=1.5)
    ax.loglog(hs, errors_c, "^-", color=COLORS["c"], label="c(x)", linewidth=1.5)

    # Reference slopes
    h_ref = np.array([hs[0], hs[-1]])
    ax.loglog(h_ref, 0.5 * (h_ref / h_ref[0]) ** 2, "k--", alpha=0.3, label="O(h^2)")
    ax.loglog(h_ref, 0.1 * (h_ref / h_ref[0]) ** 4, "k:", alpha=0.3, label="O(h^4)")

    ax.set_xlabel("Grid spacing h")
    ax.set_ylabel("Max coefficient error")
    ax.set_title("Convergence of coefficient recovery")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_singular_values(system, ax=None):
    """Plot singular values of the weak-form linear system."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.figure

    sv = system.singular_values()
    ax.semilogy(range(1, len(sv) + 1), sv, "ko-", markersize=5)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("sigma")
    ax.set_title(f"Singular values ({system.n_tests} x {system.n_unknowns}, "
                 f"cond = {system.condition_number():.1e})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_session_results(session, prediction: dict, save_path: str = None) -> plt.Figure:
    """
    Generate a comprehensive results figure after an interactive session.

    Six panels:
        Top row:    Solution u(x) | Diffusion a(x) | Advection b(x)
        Bottom row: Source f(x)   | Reaction c(x)  | Score summary

    Shows true vs recovered for each coefficient, with shaded error regions.
    """
    from basis import CoefficientFunction

    pde = session.pde
    sol = session.solution
    basis = session.basis
    n_pts = 300
    x = np.linspace(*pde.domain, n_pts)
    score = session.final_score

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # --- Panel 1: Solution u(x) ---
    ax_u = fig.add_subplot(gs[0, 0])
    ax_u.plot(sol.x, sol.u, color=COLORS["u"], linewidth=2, label="$u(x)$")
    ax_u.plot(sol.x, sol.u_x(), color=COLORS["flux"], linewidth=1.2,
              linestyle="--", alpha=0.6, label="$u'(x)$")
    ax_u.axhline(0, color='gray', linewidth=0.3, linestyle='--')
    ax_u.set_xlabel("$x$")
    ax_u.set_title("Solution", fontsize=11)
    ax_u.legend(fontsize=8)
    ax_u.set_xlim(pde.domain)

    # --- Panels 2-5: Coefficient recovery ---
    coeff_panels = [
        (gs[0, 1], "a", pde.a, "Diffusion $a(x)$"),
        (gs[0, 2], "b", pde.b, "Advection $b(x)$"),
        (gs[1, 1], "c", pde.c, "Reaction $c(x)$"),
        (gs[1, 0], "f", pde.f, "Source $f(x)$"),
    ]

    for gs_pos, name, true_fn, title in coeff_panels:
        ax = fig.add_subplot(gs_pos)
        true_vals = true_fn(x)
        rec_fn = CoefficientFunction(basis, np.array(prediction[name]))
        rec_vals = rec_fn(x)
        l1_err = np.trapezoid(np.abs(true_vals - rec_vals), x)

        ax.plot(x, true_vals, color=COLORS.get(name, "#333"), linewidth=2.2,
                label="true", zorder=3)
        ax.plot(x, rec_vals, color=COLORS.get(name, "#333"), linewidth=1.5,
                linestyle="--", alpha=0.75, label="recovered", zorder=3)
        ax.fill_between(x, true_vals, rec_vals, alpha=0.18,
                        color=COLORS.get(name, "#333"), zorder=2, label="error")
        ax.axhline(0, color='gray', linewidth=0.3, linestyle='--')
        ax.set_xlabel("$x$")
        ax.set_title(f"{title}   ($L^1$ = {l1_err:.4f})", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.set_xlim(pde.domain)

    # --- Panel 6: Score summary ---
    ax_score = fig.add_subplot(gs[1, 2])
    ax_score.axis("off")

    if score:
        summary_lines = [
            f"Difficulty: {session.difficulty}",
            f"Queries used: {session.queries_used} / {session.max_queries}",
            f"Basis: {basis.n_basis} Legendre polynomials",
            f"Unknowns: {4 * basis.n_basis}",
            "",
            "Coefficient errors ($\\|\\cdot\\|_\\infty$):",
            f"  a(x):  {score.coeff_error_a:.4f}",
            f"  b(x):  {score.coeff_error_b:.4f}",
            f"  c(x):  {score.coeff_error_c:.4f}",
            f"  f(x):   {score.coeff_error_f:.4f}",
            f"  total: {score.total_coeff_error:.4f}",
            "",
            f"Sparsity match:",
            f"  a: {'yes' if score.sparsity_match_a else 'no'}  "
            f"b: {'yes' if score.sparsity_match_b else 'no'}  "
            f"c: {'yes' if score.sparsity_match_c else 'no'}",
        ]
        ax_score.text(0.05, 0.95, "\n".join(summary_lines),
                      transform=ax_score.transAxes, fontsize=9,
                      verticalalignment="top", fontfamily="monospace",
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa",
                                edgecolor="#dee2e6", alpha=0.9))

    fig.suptitle("PDE Identification Results", fontsize=14, fontweight="bold", y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metacognitive_metrics(metrics: dict,
                               session,
                               model_name: str,
                               save_path: str = None):
    """
    Three-panel metacognitive profile figure.

    Panel 1: Relative uncertainty σ_k(j)/σ_k(a) — conditioning gap over time.
    Panel 2: Stated confidence vs. true accuracy at each solve call.
    Panel 3: Control efficiency C_k, colored by most-uncertain coefficient.
    """
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    sigma_curves = metrics["sigma_curves"]
    control      = metrics["control"]
    conf_reports = metrics["confidence_reports"]

    colors = {"a": "steelblue", "b": "darkorange", "c": "firebrick", "f": "forestgreen"}

    # Ground truth final errors per coefficient
    if session.final_score:
        true_errors = {
            "a": session.final_score.coeff_error_a,
            "b": session.final_score.coeff_error_b,
            "c": session.final_score.coeff_error_c,
            "f": session.final_score.coeff_error_f,
        }
    else:
        true_errors = {"a": 1.0, "b": 1.0, "c": 1.0, "f": 1.0}

    total_error = sum(true_errors.values())
    fig, axes = plt.subplots(3, 1, figsize=(10, 13))
    fig.suptitle(f"Metacognitive Profile — {model_name}\n"
                 f"Final error: {total_error:.4f}",
                 fontsize=13, fontweight="bold")

    # ------------------------------------------------------------------
    # Panel 1: Relative uncertainty σ_k(j) / σ_k(a)
    # ------------------------------------------------------------------
    ax = axes[0]
    ks = [s["k"] for s in sigma_curves]
    sig_a = np.array([s["sigma_a"] for s in sigma_curves])
    sig_b = np.array([s["sigma_b"] for s in sigma_curves])
    sig_c = np.array([s["sigma_c"] for s in sigma_curves])
    safe_a = np.where(sig_a > 0, sig_a, np.nan)
    ax.plot(ks, sig_b / safe_a, color=colors["b"], linewidth=2,
            label="σ(b)/σ(a) advection")
    ax.plot(ks, sig_c / safe_a, color=colors["c"], linewidth=2,
            label="σ(c)/σ(a) reaction")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
               label="Equal conditioning (ratio=1)")
    for sk in [r["query_count"] for r in conf_reports]:
        ax.axvline(sk, color="purple", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("Relative uncertainty σ_k(j) / σ_k(a)", fontsize=10)
    ax.set_title("Panel 1: Conditioning gap over time\n"
                 "(ratio > 1 = harder to recover than diffusion; "
                 "purple dashed = solve call)", fontsize=9)
    ax.legend(fontsize=9)
    ax.set_xlabel("Queries submitted")

    # ------------------------------------------------------------------
    # Panel 2: Stated confidence vs. true accuracy per coefficient
    # ------------------------------------------------------------------
    ax = axes[1]
    if conf_reports:
        coeffs = ["a", "b", "c", "f"]
        bar_width = 0.18
        x_pos = np.arange(len(conf_reports))

        def error_to_accuracy(err):
            return 1.0 / (1.0 + err) if err > 0 else 1.0

        true_acc = {k: error_to_accuracy(true_errors[k]) for k in coeffs}

        for i, coeff in enumerate(coeffs):
            stated = [r[coeff] for r in conf_reports]
            offset = (i - 1.5) * bar_width
            bars = ax.bar(x_pos + offset, stated, bar_width, alpha=0.8,
                          color=colors[coeff])
            t = true_acc[coeff]
            for j, (s, bar) in enumerate(zip(stated, bars)):
                ax.plot([x_pos[j] + offset - bar_width/2,
                         x_pos[j] + offset + bar_width/2],
                        [t, t], color="black", linewidth=2)
                bar.set_edgecolor("red" if s > t + 0.1 else "black")
                bar.set_linewidth(2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Solve @q{r['query_count']}"
                            for r in conf_reports], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Confidence / Accuracy (0–1)", fontsize=10)
        ax.set_title("Panel 2: Stated confidence (bars) vs. true accuracy (black lines)\n"
                     "(Red border = overconfident)", fontsize=9)
        patch_list = [mpatches.Patch(color=colors[c], label=c) for c in coeffs]
        ax.legend(handles=patch_list, fontsize=9, ncol=4,
                  title="Coefficient", loc="upper right")
    else:
        ax.text(0.5, 0.5,
                "No confidence reports found\n"
                "(model did not emit 'Confidence: a=XX%, b=XX%, c=XX%, f=XX%')",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Panel 2: Calibration (no data)")

    # ------------------------------------------------------------------
    # Panel 3: Control efficiency C_k
    # ------------------------------------------------------------------
    ax = axes[2]
    if control:
        ks_c    = [c["k"]      for c in control]
        C_ks    = [c["C_k"]    for c in control]
        j_stars = [c["j_star"] for c in control]
        dot_colors = [colors[j] for j in j_stars]
        ax.scatter(ks_c, C_ks, c=dot_colors, s=60, zorder=3)
        ax.axhline(1.0, color="green", linestyle="--", linewidth=0.8,
                   label="Perfect control")
        ax.axhline(0.5, color="gold", linestyle="--", linewidth=0.8,
                   label="<50% of possible gain on j*")
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        for k, C, j in zip(ks_c, C_ks, j_stars):
            if C < 0.5:
                ax.annotate(f"targeting {j}\ngot {C:.2f}",
                            xy=(k, C), xytext=(k + 0.3, C + 0.08),
                            fontsize=7, color=colors[j],
                            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel("Control efficiency C_k", fontsize=10)
        ax.set_title("Panel 3: Control efficiency — did queries target the most uncertain coefficient?\n"
                     "(color = most uncertain coeff at that step)", fontsize=9)
        legend_elements = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=colors["a"], markersize=8, label="j*=a (diffusion)"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=colors["b"], markersize=8, label="j*=b (advection)"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=colors["c"], markersize=8, label="j*=c (reaction)"),
        ]
        ax.legend(handles=legend_elements, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No control data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Panel 3: Control efficiency (no data)")
    ax.set_xlabel("Queries submitted")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig