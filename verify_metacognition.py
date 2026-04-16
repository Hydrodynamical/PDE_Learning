"""
Verify metacognitive metrics from run JSONs.

Computes M_k (monitoring accuracy) and summarizes C_k (control efficiency)
from the raw data in each JSON to confirm the writeup claims.

Usage:
    python verify_metacognition.py runs/
    python verify_metacognition.py runs/ --difficulty hard
"""
import os
import sys
import glob
import json
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict


def load_runs(directory, difficulty=None, model=None):
    paths = sorted(glob.glob(os.path.join(directory, "**", "run_*.json"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(directory, "run_*.json")))
    runs = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        cfg = d.get("config", {})
        if difficulty and cfg.get("difficulty") != difficulty:
            continue
        if model and model not in cfg.get("model", ""):
            continue
        d["_path"] = p
        runs.append(d)
    return runs


def compute_mk(run):
    """Compute M_k: Spearman correlation between stated uncertainty and true σ.

    For each confidence report, find the nearest sigma curve entry,
    then correlate (1 - stated_confidence) with true σ across {a, b, c}.

    Returns list of (query_count, M_k, details_dict) tuples.
    """
    mc = run.get("metacognitive", {})
    conf_reports = mc.get("confidence_reports", [])
    sigma_curves = mc.get("sigma_curves", [])

    if not conf_reports or not sigma_curves:
        return []

    # Build lookup: k -> sigma values
    sigma_by_k = {}
    for s in sigma_curves:
        sigma_by_k[s["k"]] = s

    results = []
    for cr in conf_reports:
        qk = cr["query_count"]

        # Find closest sigma entry
        if not sigma_by_k:
            continue
        closest_k = min(sigma_by_k.keys(), key=lambda k: abs(k - qk))
        sig = sigma_by_k[closest_k]

        # Extract values for a, b, c
        stated_uncertainty = []
        true_sigma = []
        labels = []
        for coeff in ["a", "b", "c"]:
            conf = cr.get(coeff)
            sigma = sig.get(f"sigma_{coeff}")
            if conf is not None and sigma is not None and sigma > 0:
                stated_uncertainty.append(1.0 - conf)
                true_sigma.append(sigma)
                labels.append(coeff)

        if len(stated_uncertainty) < 3:
            continue

        # Spearman rank correlation
        rho, pval = spearmanr(stated_uncertainty, true_sigma)

        # Also check orderings
        stated_order = sorted(range(3), key=lambda i: -stated_uncertainty[i])
        true_order = sorted(range(3), key=lambda i: -true_sigma[i])
        stated_ranking = [labels[i] for i in stated_order]
        true_ranking = [labels[i] for i in true_order]

        results.append({
            "query_count": qk,
            "sigma_k": closest_k,
            "M_k": rho,
            "p_value": pval,
            "stated_uncertainty": {labels[i]: stated_uncertainty[i] for i in range(3)},
            "true_sigma": {labels[i]: true_sigma[i] for i in range(3)},
            "stated_ranking": stated_ranking,
            "true_ranking": true_ranking,
            "rankings_match": stated_ranking == true_ranking,
        })

    return results


def summarize_ck(run):
    """Summarize C_k from existing control data."""
    mc = run.get("metacognitive", {})
    control = mc.get("control", [])
    if not control:
        return None

    ck_vals = [c["C_k"] for c in control]
    jstars = [c["j_star"] for c in control]
    n_perfect = sum(1 for c in ck_vals if c >= 0.999)

    from collections import Counter
    jstar_counts = Counter(jstars)

    return {
        "n_steps": len(ck_vals),
        "mean_Ck": np.mean(ck_vals),
        "median_Ck": np.median(ck_vals),
        "fraction_perfect": n_perfect / len(ck_vals),
        "j_star_distribution": dict(jstar_counts),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    directory = sys.argv[1]
    difficulty = None
    model_filter = None
    if "--difficulty" in sys.argv:
        difficulty = sys.argv[sys.argv.index("--difficulty") + 1]
    if "--model" in sys.argv:
        model_filter = sys.argv[sys.argv.index("--model") + 1]

    runs = load_runs(directory, difficulty=difficulty, model=model_filter)
    if not runs:
        print("No runs found")
        sys.exit(1)

    print(f"Loaded {len(runs)} runs")
    if difficulty:
        print(f"  filtered: difficulty={difficulty}")

    # Group by model
    by_model = defaultdict(list)
    for r in runs:
        by_model[r["config"]["model"]].append(r)

    print()
    print("=" * 80)
    print("  M_k: MONITORING ACCURACY (Spearman correlation)")
    print("=" * 80)

    for model in sorted(by_model.keys()):
        model_runs = by_model[model]
        print(f"\n  {model}:")

        all_mk = []
        all_match = []
        runs_with_data = 0

        for r in model_runs:
            seed = r["config"]["seed"]
            mk_results = compute_mk(r)

            if not mk_results:
                print(f"    s{seed}: no confidence/sigma data")
                continue

            runs_with_data += 1
            seed_mks = [m["M_k"] for m in mk_results]
            seed_matches = [m["rankings_match"] for m in mk_results]
            all_mk.extend(seed_mks)
            all_match.extend(seed_matches)

            mean_mk = np.mean(seed_mks)
            match_rate = sum(seed_matches) / len(seed_matches)

            # Show detail for first few
            detail = ""
            for m in mk_results[:2]:
                detail += (f"\n      q={m['query_count']}: M_k={m['M_k']:.3f} "
                          f"stated={m['stated_ranking']} "
                          f"true={m['true_ranking']} "
                          f"{'✓' if m['rankings_match'] else '✗'}")

            print(f"    s{seed}: {len(mk_results)} reports, "
                  f"mean M_k={mean_mk:.3f}, "
                  f"ranking match={match_rate:.0%}"
                  f"{detail}")

        if all_mk:
            print(f"    ── AGGREGATE: {len(all_mk)} measurements from {runs_with_data} runs")
            print(f"       mean M_k = {np.mean(all_mk):.3f} ± {np.std(all_mk):.3f}")
            print(f"       median M_k = {np.median(all_mk):.3f}")
            print(f"       ranking match rate = {sum(all_match)/len(all_match):.0%}")
            # Distribution
            bins = [(-1.1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.1)]
            bin_labels = ["M_k<-0.5", "-0.5≤M_k<0", "0≤M_k<0.5", "M_k≥0.5"]
            for (lo, hi), label in zip(bins, bin_labels):
                count = sum(1 for m in all_mk if lo <= m < hi)
                print(f"       {label}: {count}/{len(all_mk)}")

    print()
    print("=" * 80)
    print("  C_k: CONTROL EFFICIENCY")
    print("=" * 80)

    for model in sorted(by_model.keys()):
        model_runs = by_model[model]
        print(f"\n  {model}:")

        all_ck_means = []
        all_ck_perfect = []

        for r in model_runs:
            seed = r["config"]["seed"]
            ck_summary = summarize_ck(r)

            if not ck_summary:
                print(f"    s{seed}: no control data")
                continue

            all_ck_means.append(ck_summary["mean_Ck"])
            all_ck_perfect.append(ck_summary["fraction_perfect"])

            print(f"    s{seed}: n={ck_summary['n_steps']} "
                  f"mean={ck_summary['mean_Ck']:.3f} "
                  f"perfect={ck_summary['fraction_perfect']:.0%} "
                  f"j*={ck_summary['j_star_distribution']}")

        if all_ck_means:
            print(f"    ── AGGREGATE:")
            print(f"       mean C_k = {np.mean(all_ck_means):.3f} ± {np.std(all_ck_means):.3f}")
            print(f"       mean fraction perfect = {np.mean(all_ck_perfect):.0%}")

    # Cross-model summary table
    print()
    print("=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    print(f"  {'Model':<30s} {'M_k mean':>8s} {'M_k med':>8s} {'Rank ✓':>8s} "
          f"{'C_k mean':>8s} {'C_k=1.0':>8s}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for model in sorted(by_model.keys()):
        model_runs = by_model[model]

        all_mk = []
        all_match = []
        all_ck = []
        all_perf = []

        for r in model_runs:
            mk_results = compute_mk(r)
            all_mk.extend([m["M_k"] for m in mk_results])
            all_match.extend([m["rankings_match"] for m in mk_results])

            ck = summarize_ck(r)
            if ck:
                all_ck.append(ck["mean_Ck"])
                all_perf.append(ck["fraction_perfect"])

        mk_mean = f"{np.mean(all_mk):.3f}" if all_mk else "—"
        mk_med = f"{np.median(all_mk):.3f}" if all_mk else "—"
        match = f"{sum(all_match)/len(all_match):.0%}" if all_match else "—"
        ck_mean = f"{np.mean(all_ck):.3f}" if all_ck else "—"
        ck_perf = f"{np.mean(all_perf):.0%}" if all_perf else "—"

        short = model[:30]
        print(f"  {short:<30s} {mk_mean:>8s} {mk_med:>8s} {match:>8s} "
              f"{ck_mean:>8s} {ck_perf:>8s}")


if __name__ == "__main__":
    main()
