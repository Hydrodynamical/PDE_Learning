"""
Aggregate display for multi-seed PDE benchmark sweeps.
Usage:
    sweep = pde_benchmark.run(llm=llm, difficulty="medium")
    display_sweep(sweep.result)
    
    # Multi-model:
    all_results = {}
    for name, llm in models.items():
        r = pde_benchmark.run(llm=llm, difficulty="medium")
        all_results[name] = r.result
    display_comparison(all_results)
"""
import numpy as np


def display_sweep(data):
    """Display aggregated results from a multi-seed sweep."""
    seeds = data["seeds"]
    scores = data["scores"]
    
    print("=" * 65)
    print(f"  {data.get('difficulty','?').upper()} — {len(seeds)} seeds")
    print(f"  Median error: {np.median(scores):.6f}   "
          f"Mean: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
    print("=" * 65)

    # Per-seed scores
    print(f"\n  {'Seed':>6}  {'Total':>10}  {'a':>10}  {'b':>10}  {'c':>10}  {'f':>10}  {'Queries':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")
    for i, seed in enumerate(seeds):
        r = data["per_seed"][i]
        e = r["errors"]
        print(f"  {seed:>6}  {e['total']:>10.6f}  {e['a']:>10.6f}  {e['b']:>10.6f}  "
              f"{e['c']:>10.6f}  {e['f']:>10.6f}  {r['queries_used']:>8}")

    # Aggregate error by coefficient
    all_a = [d["errors"]["a"] for d in data["per_seed"]]
    all_b = [d["errors"]["b"] for d in data["per_seed"]]
    all_c = [d["errors"]["c"] for d in data["per_seed"]]
    all_f = [d["errors"]["f"] for d in data["per_seed"]]
    print(f"\n  Per-coefficient medians:")
    print(f"    a: {np.median(all_a):.6f}   b: {np.median(all_b):.6f}   "
          f"c: {np.median(all_c):.6f}   f: {np.median(all_f):.6f}")

    # Behavioral metrics (averaged)
    bm_keys = ["improvement_ratio", "family_entropy_normalized", "wasted_turn_fraction",
                "duplicate_queries"]
    has_bm = any("behavioral_metrics" in d for d in data["per_seed"])
    if has_bm:
        print(f"\n  Behavioral profile (median across seeds):")
        for key in bm_keys:
            vals = [d["behavioral_metrics"].get(key, 0) for d in data["per_seed"] 
                    if "behavioral_metrics" in d]
            if vals:
                print(f"    {key:<30s}: {np.median(vals):.4f}")

        # Family counts aggregated
        all_fam = {}
        for d in data["per_seed"]:
            bm = d.get("behavioral_metrics", {})
            fc = bm.get("family_counts", {})
            for fam, count in fc.items():
                all_fam[fam] = all_fam.get(fam, 0) + count
        total_q = sum(all_fam.values())
        if total_q > 0:
            print(f"\n  Test function families (pooled across seeds):")
            for fam, count in sorted(all_fam.items(), key=lambda x: -x[1]):
                if count > 0:
                    pct = count / total_q * 100
                    bar = "█" * int(pct / 3)
                    print(f"    {fam:>14}: {count:>3} ({pct:4.0f}%) {bar}")

        # Conditioning (median across seeds)
        cond_data = {}
        for d in data["per_seed"]:
            blocks = d.get("behavioral_metrics", {}).get("query_space_blocks", {})
            for block_name, block in blocks.items():
                if block_name not in cond_data:
                    cond_data[block_name] = []
                cond_data[block_name].append(block.get("condition", 0))
        if cond_data:
            print(f"\n  Conditioning κ (median across seeds):")
            for name in ["diffusion", "advection", "reaction"]:
                if name in cond_data:
                    med = np.median(cond_data[name])
                    mx = max(cond_data[name])
                    bar = "█" * int(min(med, 30))
                    print(f"    {name:>10}: κ={med:>6.2f} (max {mx:.2f})  {bar}")

    # Reliability
    n_success = sum(1 for s in scores if s < 0.01)
    n_fail = sum(1 for s in scores if s > 0.1)
    print(f"\n  Reliability:")
    print(f"    Success rate (<0.01):  {n_success}/{len(scores)}")
    print(f"    Failure rate (>0.1):   {n_fail}/{len(scores)}")
    print()


def display_comparison(all_results, difficulty=""):
    """Display side-by-side comparison of multiple models."""
    print("=" * 75)
    print(f"  MODEL COMPARISON — {difficulty.upper() or '?'} difficulty")
    print("=" * 75)

    # Leaderboard
    print(f"\n  {'Model':<20s}  {'Median':>10s}  {'IQR':>16s}  {'Max':>8s}  {'Pass':>6s}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*16}  {'─'*8}  {'─'*6}")
    
    ranked = sorted(all_results.items(), key=lambda x: np.median(x[1]["scores"]))
    for name, data in ranked:
        s = data["scores"]
        med = np.median(s)
        q1, q3 = np.percentile(s, [25, 75])
        mx = max(s)
        passes = sum(1 for v in s if v < 0.01)
        total = len(s)
        print(f"  {name:<20s}  {med:>10.6f}  [{q1:.4f}, {q3:.4f}]  {mx:>8.4f}  {passes}/{total}")

    # Behavioral comparison
    print(f"\n  {'Model':<20s}  {'Entropy':>8s}  {'Improv':>8s}  {'Wasted':>8s}  {'Queries':>8s}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for name, data in ranked:
        seeds = data["per_seed"]
        ent = np.median([d.get("behavioral_metrics", {}).get("family_entropy_normalized", 0) for d in seeds])
        imp = np.median([d.get("behavioral_metrics", {}).get("improvement_ratio", 1) for d in seeds])
        wst = np.median([d.get("behavioral_metrics", {}).get("wasted_turn_fraction", 0) for d in seeds])
        qry = np.median([d.get("queries_used", 0) for d in seeds])
        print(f"  {name:<20s}  {ent:>8.2f}  {imp:>7.1f}×  {wst:>7.0%}  {qry:>8.0f}")

    # Conditioning comparison
    print(f"\n  {'Model':<20s}  {'κ_diff':>8s}  {'κ_adv':>8s}  {'κ_react':>8s}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}")
    for name, data in ranked:
        seeds = data["per_seed"]
        kd, ka, kr = [], [], []
        for d in seeds:
            blocks = d.get("behavioral_metrics", {}).get("query_space_blocks", {})
            kd.append(blocks.get("diffusion", {}).get("condition", 0))
            ka.append(blocks.get("advection", {}).get("condition", 0))
            kr.append(blocks.get("reaction", {}).get("condition", 0))
        print(f"  {name:<20s}  {np.median(kd):>8.2f}  {np.median(ka):>8.2f}  {np.median(kr):>8.2f}")

    print()
