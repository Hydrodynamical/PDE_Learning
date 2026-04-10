"""
Pretty-print PDE benchmark results in a notebook cell.
Usage:
    data = result.result
    display_run(data)
"""

def display_run(data):
    """Display a single PDE benchmark run with all metrics."""
    err = data["errors"]
    bm = data.get("behavioral_metrics", {})
    mc = data.get("metacognitive", {})
    rl = data["run_log"]

    # Header
    print("=" * 60)
    print(f"  Total error: {data['total_error']:.6f}")
    print(f"  Queries: {data['queries_used']}  |  Turns: {data['turns']}")
    print("=" * 60)

    # Per-coefficient errors
    print("\nCoefficient errors:")
    for k in ["a", "b", "c", "f"]:
        v = err.get(k, 0)
        bar = "█" * max(1, int(min(v / err["total"], 1.0) * 30)) if err["total"] > 0 else ""
        print(f"  {k}(x): {v:.6f}  {bar}")
    print(f"  total: {err['total']:.6f}")

    # Solve history
    if rl.get("solves"):
        print(f"\nSolve history ({len(rl['solves'])} solves):")
        print(f"  {'#':>3}  {'Eqs':>4}  {'a err':>10}  {'b err':>10}  {'c err':>10}  {'total':>10}  {'taxonomy'}")
        print(f"  {'─'*3}  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*20}")
        for s in rl["solves"]:
            ce = s["coeff_errors"]
            tax = s.get("test_function_taxonomy", {})
            tax_str = ", ".join(f"{k[0]}:{v}" for k, v in tax.items() if v > 0)
            print(f"  {s['solve_number']:>3}  {s['n_equations']:>4}  "
                  f"{ce['a']:>10.6f}  {ce['b']:>10.6f}  {ce['c']:>10.6f}  "
                  f"{ce['total']:>10.6f}  {tax_str}")

    # Error convergence
    if rl.get("error_curves"):
        print(f"\nError convergence:")
        for c in rl["error_curves"]:
            print(f"  Q{c['query_count']:>2}: total={c['total_error']:.6f}  "
                  f"(a={c['a_error']:.2e} b={c['b_error']:.2e} c={c['c_error']:.2e})")

    # Term integrals
    if rl.get("term_integrals"):
        print(f"\nTerm integrals ({len(rl['term_integrals'])}):")
        print(f"  {'φ(x)':>24s}  {'diffusion':>12}  {'advection':>12}  {'reaction':>12}")
        print(f"  {'─'*24}  {'─'*12}  {'─'*12}  {'─'*12}")
        for t in rl["term_integrals"]:
            print(f"  {t['test_function']:>24s}  {t['diffusion']:>12.6f}  "
                  f"{t['advection']:>12.6f}  {t['reaction']:>12.6f}")

    # Verifications
    if rl.get("verifications"):
        print(f"\nVerifications ({len(rl['verifications'])}):")
        for v in rl["verifications"]:
            disc = v["discrepancy"]
            status = "✓" if disc < 1e-6 else "✗"
            print(f"  {status} {v['test_function']:>24s}  disc={disc:.2e}")

    # Behavioral metrics
    if bm:
        print(f"\nBehavioral profile:")
        print(f"  Learning rate:      {bm.get('learning_rate', 0):.4f}")
        print(f"  Improvement ratio:  {bm.get('improvement_ratio', 0):.2f}×")
        print(f"  Wasted turns:       {bm.get('wasted_turns', 0)}/{data['turns']} "
              f"({bm.get('wasted_turn_fraction', 0)*100:.0f}%)")
        print(f"  Duplicates:         {bm.get('duplicate_queries', 0)}")
        print(f"  Max unproductive:   {bm.get('max_unproductive_streak', 0)} consecutive")

        # Family diversity
        fc = bm.get("family_counts", {})
        if fc:
            total_q = sum(fc.values())
            print(f"\nTest function families (entropy={bm.get('family_entropy_normalized', 0):.2f}):")
            for fam, count in sorted(fc.items(), key=lambda x: -x[1]):
                if count > 0:
                    pct = count / total_q * 100 if total_q > 0 else 0
                    bar = "█" * int(pct / 3)
                    print(f"  {fam:>14}: {count:>2} ({pct:4.0f}%) {bar}")

        # Conditioning
        blocks = bm.get("query_space_blocks", {})
        if blocks:
            print(f"\nConditioning (higher = harder to identify):")
            for name in ["diffusion", "advection", "reaction"]:
                b = blocks.get(name, {})
                cond = b.get("condition", 0)
                rank = b.get("effective_rank", "?")
                maxr = b.get("max_rank", "?")
                bar = "█" * int(min(cond, 20))
                print(f"  {name:>10}: κ={cond:>6.2f}  rank={rank}/{maxr}  {bar}")

    # Metacognitive metrics
    if mc:
        cr = mc.get("confidence_reports", [])
        if cr:
            print(f"\nConfidence reports ({len(cr)}):")
            print(f"  {'Queries':>8}  {'a':>6}  {'b':>6}  {'c':>6}  {'f':>6}")
            print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
            for r in cr:
                print(f"  {r.get('query_count', '?'):>8}  "
                      f"{r['a']*100:>5.0f}%  {r['b']*100:>5.0f}%  "
                      f"{r['c']*100:>5.0f}%  {r['f']*100:>5.0f}%")

        mon = mc.get("monitoring", [])
        if mon:
            print(f"\nMonitoring accuracy:")
            for m in mon:
                mk = m.get('M_k')
                mk_str = f"{mk:.2f}" if mk is not None else "N/A"
                stated = m.get('stated', [])
                stated_str = ', '.join(f'{s:.2f}' if s is not None else 'N/A' for s in stated)
                print(f"  Q{m.get('k', '?')}: M_k={mk_str}  stated=[{stated_str}]")

        ctrl = mc.get("control", [])
        if ctrl:
            targets = [c.get("j_star", "?") for c in ctrl]
            ck_vals = [c['C_k'] for c in ctrl if c.get('C_k') is not None]
            mean_ck = sum(ck_vals) / len(ck_vals) if ck_vals else 0
            print(f"\nControl efficiency: "
                  f"mean={mean_ck:.2f}  "
                  f"targets: {' → '.join(targets)}")

    print()