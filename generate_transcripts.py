"""
Generate benchmark runs across models, difficulties, and seeds.

Runs each combination and saves JSON logs, dashboard plots, and AUC plots.
Prints a summary comparison table at the end.

Usage:
    python generate_transcripts.py
    python generate_transcripts.py --seeds 42-43 --difficulties easy medium
    python generate_transcripts.py --models gpt-4o gpt-5.4 --difficulties hard
"""

import argparse
import time
import json
import os
import numpy as np

from main_loop import (
    ProbeSession, run_probe_session, compute_efficiency_curve,
    OpenAIBackend, AnthropicBackend, MockBackend,
    DIFFICULTY_CONFIG, _compute_stability, plot_auc_curves,
)


# ── Model registry ──────────────────────────────────────────────────────────

MODELS = {
    # OpenAI — three generations for longitudinal comparison
    "gpt-4o":       {"provider": "openai"},
    "gpt-5.4-mini": {"provider": "openai"},
    "gpt-5.4":      {"provider": "openai"},
    # Anthropic (uncomment to include)
    # "claude-sonnet-4-20250514": {"provider": "anthropic"},
    # "claude-opus-4-6":          {"provider": "anthropic"},
}

DEFAULT_MODELS = ["gpt-4o", "gpt-5.4-mini", "gpt-5.4"]
DEFAULT_DIFFICULTIES = ["easy", "medium", "hard"]
DEFAULT_SEEDS = [42, 137]


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_seeds(seed_str: str) -> list[int]:
    """Parse '42-46' or '42 43 44' into a list of ints."""
    if "-" in seed_str and seed_str.count("-") == 1:
        lo, hi = seed_str.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in seed_str.split()]


def make_backend(model: str, provider: str):
    if provider == "openai":
        return OpenAIBackend(model=model)
    elif provider == "anthropic":
        return AnthropicBackend(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Single run ───────────────────────────────────────────────────────────────

def run_single(model: str, provider: str, difficulty: str, seed: int,
               output_dir: str, baseline: bool = False) -> dict:
    """Execute one benchmark run and save all outputs. Returns summary dict."""

    config = DIFFICULTY_CONFIG[difficulty]
    max_queries = config["budget"]
    max_turns = 2 * max_queries

    session = ProbeSession.from_difficulty(difficulty, seed, max_queries)
    backend = make_backend(model, provider)

    run_log = {
        "config": {
            "model": model,
            "provider": provider,
            "difficulty": difficulty,
            "n_basis": config["n_basis"],
            "unknowns": config["unknowns"],
            "seed": seed,
            "max_queries": max_queries,
            "max_turns": max_turns,
            "budget_source": "auto",
            "prompt_condition": "baseline" if baseline else "standard",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "turns": [],
        "solves": [],
        "queries": [],
        "verifications": [],
        "term_integrals": [],
        "error_curves": [],
        "results": None,
        "ground_truth": None,
        "behavioral_metrics": None,
    }

    result = run_probe_session(
        session, backend, verbose=False,
        max_turns=max_turns, baseline=baseline,
        run_log=run_log,
    )

    # ── Finalize run_log ──
    pde = session.pde
    run_log["ground_truth"] = {
        "a_coeffs": pde.a.coeffs.tolist(),
        "b_coeffs": pde.b.coeffs.tolist(),
        "c_coeffs": pde.c.coeffs.tolist(),
        "f_coeffs": pde.f.coeffs.tolist(),
        "pde_string": pde.describe(),
    }

    if "score" in result:
        sc = result["score"]
        run_log["results"] = {
            "coefficient_errors": {
                "a": sc.get("a_coeff_error"), "b": sc.get("b_coeff_error"),
                "c": sc.get("c_coeff_error"), "f": sc.get("f_coeff_error"),
                "total": sc.get("total_coeff_error"),
            },
            "pointwise_errors": {
                "a": sc.get("a_pointwise_error"), "b": sc.get("b_pointwise_error"),
                "c": sc.get("c_pointwise_error"), "f": sc.get("f_pointwise_error"),
                "total": sc.get("total_pointwise_error"),
            },
        }

    if "efficiency" in result:
        eff = result["efficiency"]
        if run_log["results"] is None:
            run_log["results"] = {}
        run_log["results"]["efficiency_auc"] = eff.get("auc")
        run_log["results"]["queries_to_error_1"] = eff.get("queries_to_below_1")
        run_log["results"]["queries_to_error_01"] = eff.get("queries_to_below_0.1")

    run_log["behavioral_metrics"] = {
        "total_turns": result["turns"],
        "queries_used": result["queries_used"],
        "budget_utilization": result["queries_used"] / max_queries,
        "solve_count": len(run_log["solves"]),
        "check_count": len(run_log["verifications"]),
        "term_integral_count": len(run_log["term_integrals"]),
        "stopped_early": result["queries_used"] < max_queries,
        "unused_queries": max_queries - result["queries_used"],
        "coefficient_stability": _compute_stability(run_log["solves"]),
        "auto_submitted": result.get("auto_submitted", False),
    }

    if run_log["solves"]:
        best = min(run_log["solves"], key=lambda s: s["coeff_errors"]["total"])
        run_log["behavioral_metrics"]["best_solve_error"] = best["coeff_errors"]["total"]
        run_log["behavioral_metrics"]["best_solve_number"] = best["solve_number"]
        run_log["behavioral_metrics"]["submitted_error"] = (
            run_log["results"]["coefficient_errors"]["total"]
            if run_log.get("results") and run_log["results"].get("coefficient_errors")
            else None
        )

    # ── Save files ──
    model_slug = model.replace("/", "_").replace(".", "_")
    stem = f"run_{difficulty}_s{seed}_{model_slug}_{time.strftime('%Y%m%d_%H%M%S')}"

    log_path = os.path.join(output_dir, stem + ".json")
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2, default=str)

    # Dashboard plot
    if session.prediction_submitted and "prediction" in result:
        try:
            import matplotlib
            matplotlib.use("Agg")
            from diagnostics import plot_session_results
            import matplotlib.pyplot as plt
            fig = plot_session_results(session, result["prediction"],
                                       save_path=os.path.join(output_dir, stem + "_dashboard.png"))
            plt.close(fig)
        except Exception:
            pass

    # AUC plot
    if "efficiency" in result:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig = plot_auc_curves({model: result["efficiency"]},
                                  save_path=os.path.join(output_dir, stem + "_auc.png"))
            plt.close(fig)
        except Exception:
            pass

    # ── Build summary ──
    summary = {
        "model": model,
        "provider": provider,
        "difficulty": difficulty,
        "seed": seed,
        "queries_used": result["queries_used"],
        "max_queries": max_queries,
        "turns": result["turns"],
        "log_path": log_path,
    }

    if "score" in result:
        sc = result["score"]
        summary["total_coeff_error"] = sc.get("total_coeff_error")
        summary["a_error"] = sc.get("a_coeff_error") or sc.get("coeff_error_a")
        summary["b_error"] = sc.get("b_coeff_error") or sc.get("coeff_error_b")
        summary["c_error"] = sc.get("c_coeff_error") or sc.get("coeff_error_c")
        summary["f_error"] = sc.get("f_coeff_error") or sc.get("coeff_error_f")

    if "efficiency" in result:
        summary["auc"] = result["efficiency"].get("auc")

    bm = run_log["behavioral_metrics"]
    summary["solve_count"] = bm["solve_count"]
    summary["check_count"] = bm["check_count"]
    summary["best_solve_error"] = bm.get("best_solve_error")
    summary["budget_util"] = bm["budget_utilization"]

    return summary


# ── Summary table ────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]):
    """Print a formatted comparison table grouped by difficulty."""

    print(f"\n{'=' * 120}")
    print(f"{'Model':>16} {'Diff':>8} {'Seed':>5} {'Coeff Err':>10} {'Best Solve':>11} "
          f"{'a':>8} {'b':>8} {'c':>8} {'f':>8} "
          f"{'Q used':>7} {'Solves':>7} {'AUC':>8}")
    print(f"{'-' * 120}")

    for diff in ["easy", "medium", "hard", "extreme"]:
        group = [r for r in results if r["difficulty"] == diff]
        if not group:
            continue
        for r in sorted(group, key=lambda x: (x["model"], x["seed"])):
            ce = r.get("total_coeff_error") or float("inf")
            bse = r.get("best_solve_error") or float("inf")
            ae = r.get("a_error") or float("inf")
            be = r.get("b_error") or float("inf")
            cce = r.get("c_error") or float("inf")
            fe = r.get("f_error") or float("inf")
            q = r.get("queries_used") or 0
            sc = r.get("solve_count") or 0
            auc = r.get("auc") or float("inf")
            model_short = r["model"]
            if len(model_short) > 16:
                model_short = model_short[:16]
            if ce == float("inf") and q == 0:
                print(f"{model_short:>16} {diff:>8} {r['seed']:>5}   {'NO SCORE':>10}")
            else:
                print(f"{model_short:>16} {diff:>8} {r['seed']:>5} {ce:>10.4f} {bse:>11.4f} "
                      f"{ae:>8.4f} {be:>8.4f} {cce:>8.4f} {fe:>8.4f} "
                      f"{q:>7} {sc:>7} {auc:>8.4f}")
        print()

    # Per-model aggregates
    models = sorted(set(r["model"] for r in results))
    print(f"{'=' * 120}")
    print("AGGREGATES (median total coefficient error)")
    print(f"{'=' * 120}")
    for diff in ["easy", "medium", "hard", "extreme"]:
        group = [r for r in results if r["difficulty"] == diff
                 and r.get("total_coeff_error") is not None]
        if not group:
            continue
        print(f"\n  {diff}:")
        for model in models:
            errs = [r["total_coeff_error"] for r in group
                    if r["model"] == model and r["total_coeff_error"] is not None]
            if errs:
                print(f"    {model:>16}: median={np.median(errs):.4f}  "
                      f"mean={np.mean(errs):.4f}  n={len(errs)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run PDE benchmark across models, difficulties, and seeds"
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Models to benchmark (default: {DEFAULT_MODELS})")
    parser.add_argument("--difficulties", nargs="+", default=DEFAULT_DIFFICULTIES,
                        choices=["easy", "medium", "hard", "extreme"],
                        help=f"Difficulty levels (default: {DEFAULT_DIFFICULTIES})")
    parser.add_argument("--seeds", type=str, default="42 137",
                        help="Seeds as space-separated or range (e.g. '42 137' or '42-46')")
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Output directory (default: runs/)")
    parser.add_argument("--baseline", action="store_true",
                        help="Use baseline prompt (no mathematical context)")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve providers
    model_configs = {}
    for m in args.models:
        if m in MODELS:
            model_configs[m] = MODELS[m]["provider"]
        elif "claude" in m or "anthropic" in m.lower():
            model_configs[m] = "anthropic"
        else:
            model_configs[m] = "openai"

    total_runs = len(args.models) * len(args.difficulties) * len(seeds)

    print("=" * 70)
    print("PDE BENCHMARK SWEEP")
    print("=" * 70)
    print(f"Models:       {args.models}")
    print(f"Difficulties: {args.difficulties}")
    print(f"Seeds:        {seeds}")
    print(f"Total runs:   {total_runs}")
    print(f"Output:       {args.output_dir}/")
    print(f"Prompt:       {'baseline' if args.baseline else 'standard'}")
    print()

    results = []
    run_num = 0

    for difficulty in args.difficulties:
        for model in args.models:
            provider = model_configs[model]
            for seed in seeds:
                run_num += 1
                tag = f"[{run_num}/{total_runs}] {model} / {difficulty} / s{seed}"
                print(f"{tag} ...", end=" ", flush=True)
                t0 = time.time()
                try:
                    summary = run_single(
                        model=model,
                        provider=provider,
                        difficulty=difficulty,
                        seed=seed,
                        output_dir=args.output_dir,
                        baseline=args.baseline,
                    )
                    results.append(summary)
                    elapsed = time.time() - t0
                    ce = summary.get("total_coeff_error", float("inf"))
                    q = summary.get("queries_used", 0)
                    print(f"err={ce:.4f}  q={q}  {elapsed:.0f}s")
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"FAILED ({elapsed:.0f}s): {e}")
                    results.append({
                        "model": model, "provider": provider,
                        "difficulty": difficulty, "seed": seed,
                        "error": str(e),
                    })

    # Save aggregate results
    results_path = os.path.join(args.output_dir, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")

    print_summary_table(results)


if __name__ == "__main__":
    main()