"""
Diagnostic checks for PDE benchmark run JSONs.

Usage:
    python diagnostic_kaggle.py Data1/               # all runs
    python diagnostic_kaggle.py Data1/ --difficulty extreme
    python diagnostic_kaggle.py Data1/ --model gpt-5.4

Checks:
  1. Config sanity      — reasoning_effort, prompt_condition, budget/unknowns ratio
  2. Parsing failures    — wasted turns, expression errors, ^ vs ** issues
  3. Behavioral red flags — zero solves, auto-submit, single-solve-and-stop, duplicate queries
  4. C_k degeneracy      — is control efficiency trivially 1.0? is j_star constant?
  5. Suspiciously good   — near-zero error without using budget (lucky instance or bug?)
  6. Suspiciously bad    — high error despite full budget (systematic failure?)
  7. Confidence sanity   — are confidence reports actually varying, or constant/missing?
  8. Per-run transcript smell test — first/last actions, expression samples
"""
import os
import sys
import glob
import json
import numpy as np
from collections import defaultdict, Counter


# ── Loading ───────────────────────────────────────────────────────────────

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


def run_label(d):
    cfg = d["config"]
    return f"{cfg.get('model','?')} / {cfg.get('difficulty','?')} / s{cfg.get('seed','?')}"


# ── Check 1: Config sanity ────────────────────────────────────────────────

def check_config(runs):
    print("\n" + "=" * 70)
    print("  CHECK 1: Config sanity")
    print("=" * 70)

    issues = []
    for d in runs:
        cfg = d["config"]
        label = run_label(d)

        # reasoning_effort
        re = cfg.get("reasoning_effort")
        if re is None:
            issues.append(f"  ⚠  {label}: reasoning_effort = null (may default to 'none' for OpenAI)")
        elif re == "none":
            issues.append(f"  🔴 {label}: reasoning_effort = 'none' — chain-of-thought disabled!")

        # prompt_condition
        pc = cfg.get("prompt_condition", "")
        if "baseline" in str(pc).lower():
            issues.append(f"  ⚠  {label}: prompt_condition = '{pc}' (baseline, no math context)")

        # budget ratio
        unknowns = cfg.get("unknowns", 0)
        budget = cfg.get("max_queries", 0)
        if unknowns > 0 and budget > 0:
            ratio = budget / unknowns
            if ratio < 1.4 or ratio > 1.6:
                issues.append(f"  ⚠  {label}: budget/unknowns = {ratio:.2f} (expected ~1.5)")

    if issues:
        print("\n".join(issues))
    else:
        print("  ✓ All configs look normal")

    # Summary table
    print("\n  Config summary:")
    print(f"  {'Model':<30s} {'Diff':<10s} {'RE':<10s} {'Prompt':<12s} {'Budget':<6s} {'Unknowns':<8s} {'Seeds'}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*12} {'─'*6} {'─'*8} {'─'*10}")
    grouped = defaultdict(list)
    for d in runs:
        cfg = d["config"]
        key = (cfg.get("model"), cfg.get("difficulty"))
        grouped[key].append(cfg)
    for (model, diff), cfgs in sorted(grouped.items()):
        seeds = sorted(set(c.get("seed") for c in cfgs))
        re_vals = set(str(c.get("reasoning_effort")) for c in cfgs)
        pc_vals = set(str(c.get("prompt_condition")) for c in cfgs)
        print(f"  {str(model):<30s} {str(diff):<10s} {','.join(re_vals):<10s} "
              f"{','.join(pc_vals):<12s} {cfgs[0].get('max_queries','?'):<6} "
              f"{cfgs[0].get('unknowns','?'):<8} {seeds}")


# ── Check 2: Parsing failures / wasted turns ──────────────────────────────

def check_parsing(runs):
    print("\n" + "=" * 70)
    print("  CHECK 2: Parsing failures and wasted turns")
    print("=" * 70)

    for d in runs:
        label = run_label(d)
        bm = d.get("behavioral_metrics", {})
        turns = d.get("turns", [])

        wasted = bm.get("wasted_turns", 0)
        wasted_frac = bm.get("wasted_turn_fraction", 0)
        total_turns = bm.get("total_turns", len(turns))
        max_streak = bm.get("max_unproductive_streak", 0)
        dupes = bm.get("duplicate_queries", 0)

        flags = []
        if wasted_frac > 0.15:
            flags.append(f"🔴 {wasted_frac:.0%} wasted")
        elif wasted_frac > 0.05:
            flags.append(f"⚠  {wasted_frac:.0%} wasted")

        if max_streak > 3:
            flags.append(f"🔴 streak={max_streak}")
        elif max_streak > 1:
            flags.append(f"⚠  streak={max_streak}")

        if dupes > 2:
            flags.append(f"⚠  {dupes} duplicate queries")

        # Scan for expression errors in turn content
        caret_count = 0
        parse_error_count = 0
        for t in turns:
            content = t.get("content", "") or ""
            if "^" in content and "**" not in content and "QUERY" in content.upper():
                caret_count += 1
            if any(kw in content.lower() for kw in ["error:", "invalid", "not recognized", "malformed"]):
                parse_error_count += 1

        if caret_count > 0:
            flags.append(f"⚠  {caret_count} turns with ^ in QUERY (should be **)")
        if parse_error_count > 0:
            flags.append(f"⚠  {parse_error_count} turns with error keywords")

        status = " | ".join(flags) if flags else "✓ clean"
        print(f"  {label:<55s}  {wasted}/{total_turns} wasted  {status}")


# ── Check 3: Behavioral red flags ─────────────────────────────────────────

def check_behavioral(runs):
    print("\n" + "=" * 70)
    print("  CHECK 3: Behavioral red flags")
    print("=" * 70)

    for d in runs:
        label = run_label(d)
        cfg = d["config"]
        bm = d.get("behavioral_metrics", {})
        solves = d.get("solves", [])

        budget = cfg.get("max_queries", 1)
        used = bm.get("queries_used", 0)
        utilization = used / budget if budget > 0 else 0
        solve_count = bm.get("solve_count", len(solves))
        auto = bm.get("auto_submitted", False)
        stopped_early = bm.get("stopped_early", False)
        total_err = d.get("results", {}).get("coefficient_errors", {}).get("total", None)
        tail = bm.get("tail_after_last_solve", 0)

        flags = []

        if solve_count == 0:
            flags.append("🔴 ZERO solves")
        elif solve_count == 1:
            flags.append("⚠  single solve (no iteration)")

        if auto:
            flags.append("⚠  auto-submitted (didn't call PREDICT)")

        if utilization < 0.5:
            flags.append(f"🔴 only used {utilization:.0%} of budget")
        elif utilization < 0.75:
            flags.append(f"⚠  used {utilization:.0%} of budget")

        if stopped_early:
            flags.append("⚠  stopped early")

        if tail is not None and tail > 10:
            flags.append(f"⚠  tail={tail} turns after last solve (goal drift)")

        # Suspiciously good
        if total_err is not None and total_err < 1e-6 and utilization < 0.7:
            flags.append(f"🔴 SUSPICIOUS: error={total_err:.2e} but only {utilization:.0%} budget used")

        # Suspiciously bad
        if total_err is not None and total_err > 5 and utilization > 0.9:
            flags.append(f"🔴 HIGH ERROR despite full budget: {total_err:.2f}")

        status = " | ".join(flags) if flags else "✓"
        err_str = f"{total_err:.4f}" if total_err is not None else "n/a"
        print(f"  {label:<55s}  E={err_str:<10s} Q={used}/{budget}  "
              f"solves={solve_count}  {status}")


# ── Check 4: C_k degeneracy ──────────────────────────────────────────────

def check_control(runs):
    print("\n" + "=" * 70)
    print("  CHECK 4: Control efficiency (C_k) degeneracy")
    print("=" * 70)

    by_model = defaultdict(list)
    for d in runs:
        model = d["config"].get("model", "?")
        by_model[model].append(d)

    for model, model_runs in sorted(by_model.items()):
        print(f"\n  {model}:")

        all_ck = []
        all_jstar = []
        per_run_summaries = []

        for d in model_runs:
            label = f"s{d['config'].get('seed','?')}"
            control = d.get("metacognitive", {}).get("control", [])

            if not control:
                per_run_summaries.append(f"    {label}: no control data")
                continue

            ck_vals = [c["C_k"] for c in control]
            jstar_vals = [c["j_star"] for c in control]
            all_ck.extend(ck_vals)
            all_jstar.extend(jstar_vals)

            # Check for degeneracy
            n_perfect = sum(1 for c in ck_vals if c >= 0.999)
            jstar_counts = Counter(jstar_vals)
            dominant_j = jstar_counts.most_common(1)[0] if jstar_counts else ("?", 0)

            flags = []
            if n_perfect == len(ck_vals):
                flags.append(f"🔴 ALL C_k=1.0 ({len(ck_vals)}/{len(ck_vals)})")
            elif n_perfect / len(ck_vals) > 0.9:
                flags.append(f"⚠  {n_perfect}/{len(ck_vals)} C_k=1.0")

            if dominant_j[1] / len(jstar_vals) > 0.8:
                flags.append(f"j_star={dominant_j[0]} {dominant_j[1]}/{len(jstar_vals)} "
                             f"({dominant_j[1]/len(jstar_vals):.0%})")

            status = " | ".join(flags) if flags else f"C_k range [{min(ck_vals):.3f}, {max(ck_vals):.3f}]"
            jstar_str = ", ".join(f"{k}={v}" for k, v in sorted(jstar_counts.items()))
            per_run_summaries.append(f"    {label}: n={len(ck_vals):>3d}  median C_k={np.median(ck_vals):.3f}  "
                                     f"j_star=[{jstar_str}]  {status}")

        for s in per_run_summaries:
            print(s)

        if all_ck:
            n_total = len(all_ck)
            n_perfect = sum(1 for c in all_ck if c >= 0.999)
            jstar_overall = Counter(all_jstar)
            print(f"    ── AGGREGATE: {n_total} steps, {n_perfect} perfect ({n_perfect/n_total:.0%}), "
                  f"j_star distribution: {dict(jstar_overall)}")


# ── Check 5: Confidence report sanity ─────────────────────────────────────

def check_confidence(runs):
    print("\n" + "=" * 70)
    print("  CHECK 5: Confidence report sanity")
    print("=" * 70)

    by_model = defaultdict(list)
    for d in runs:
        model = d["config"].get("model", "?")
        by_model[model].append(d)

    for model, model_runs in sorted(by_model.items()):
        print(f"\n  {model}:")

        for d in model_runs:
            label = f"s{d['config'].get('seed','?')}"
            mc = d.get("metacognitive", {})
            conf = mc.get("confidence_reports", [])
            sigma = mc.get("sigma_curves", [])

            if not conf:
                print(f"    {label}: ⚠  NO confidence reports")
                continue

            n_reports = len(conf)

            # Check for constant confidence
            a_vals = [c.get("a", 0) for c in conf]
            b_vals = [c.get("b", 0) for c in conf]
            c_vals = [c.get("c", 0) for c in conf]

            flags = []

            # All same value?
            for coeff, vals in [("a", a_vals), ("b", b_vals), ("c", c_vals)]:
                if len(set(f"{v:.4f}" for v in vals)) == 1:
                    flags.append(f"{coeff}=constant({vals[0]:.2f})")

            # Confidence always uniform across coefficients?
            uniform_count = 0
            for cr in conf:
                a, b, c = cr.get("a", 0), cr.get("b", 0), cr.get("c", 0)
                if abs(a - b) < 0.02 and abs(b - c) < 0.02:
                    uniform_count += 1
            if uniform_count == n_reports:
                flags.append("🔴 ALL reports have uniform a≈b≈c")
            elif uniform_count > n_reports * 0.5:
                flags.append(f"⚠  {uniform_count}/{n_reports} reports uniform")

            # Does confidence track the right direction? (a should be highest, c lowest)
            a_gt_c = sum(1 for cr in conf if cr.get("a", 0) > cr.get("c", 0))
            if a_gt_c < n_reports * 0.5:
                flags.append(f"⚠  conf(a)>conf(c) only {a_gt_c}/{n_reports} times (reversed hierarchy?)")

            # Confidence extremes
            all_vals = a_vals + b_vals + c_vals
            if all(v > 0.9 for v in all_vals):
                flags.append("⚠  all confidence > 90% (overconfident?)")
            if all(v < 0.1 for v in all_vals):
                flags.append("⚠  all confidence < 10% (underconfident?)")

            status = " | ".join(flags) if flags else "✓ varying"
            print(f"    {label}: {n_reports} reports, σ-curves={len(sigma)}  "
                  f"ranges: a=[{min(a_vals):.2f}–{max(a_vals):.2f}] "
                  f"b=[{min(b_vals):.2f}–{max(b_vals):.2f}] "
                  f"c=[{min(c_vals):.2f}–{max(c_vals):.2f}]  {status}")


# ── Check 6: Quick transcript smell test ──────────────────────────────────

def check_transcripts(runs, n_sample=3):
    print("\n" + "=" * 70)
    print(f"  CHECK 6: Transcript samples (first {n_sample} runs per model)")
    print("=" * 70)

    by_model = defaultdict(list)
    for d in runs:
        model = d["config"].get("model", "?")
        by_model[model].append(d)

    for model, model_runs in sorted(by_model.items()):
        print(f"\n  {model}:")
        for d in model_runs[:n_sample]:
            label = f"s{d['config'].get('seed','?')}"
            turns = d.get("turns", [])
            queries = d.get("queries", [])

            if not turns:
                print(f"    {label}: no turns")
                continue

            # Action sequence summary
            action_seq = []
            for t in turns:
                actions = t.get("parsed_actions", [])
                content = (t.get("content", "") or "").lower()
                if "query" in actions:
                    action_seq.append("Q")
                elif "predict" in actions:
                    action_seq.append("P")
                elif "compute" in actions:
                    if "solve" in content and "eval" not in content:
                        action_seq.append("S")
                    elif "check" in content or "verify" in content:
                        action_seq.append("C")
                    elif "term_integral" in content:
                        action_seq.append("T")
                    else:
                        action_seq.append("c")  # other compute
                else:
                    action_seq.append("·")

            seq_str = "".join(action_seq)
            # Compress long sequences
            if len(seq_str) > 70:
                seq_str = seq_str[:35] + "..." + seq_str[-35:]

            print(f"    {label}: {len(turns)} turns  [{seq_str}]")

            # Sample test functions
            if queries:
                sample_q = queries[:3] + (queries[-2:] if len(queries) > 3 else [])
                tf_samples = [f"Q{q['query_number']}: {q['test_function'][:50]}" for q in sample_q]
                print(f"          queries: {', '.join(tf_samples)}")

            # First and last turn content snippets
            first_content = (turns[0].get("content", "") or "")[:100]
            last_content = (turns[-1].get("content", "") or "")[:100]
            print(f"          first: {first_content}...")
            print(f"          last:  {last_content}...")


# ── Check 7: Error distribution and outlier detection ─────────────────────

def check_errors(runs):
    print("\n" + "=" * 70)
    print("  CHECK 7: Error distribution and outliers")
    print("=" * 70)

    by_model_diff = defaultdict(list)
    for d in runs:
        cfg = d["config"]
        key = f"{cfg.get('model','?')} / {cfg.get('difficulty','?')}"
        total = d.get("results", {}).get("coefficient_errors", {}).get("total", None)
        if total is not None:
            by_model_diff[key].append((d["config"].get("seed"), total, d))

    for key, entries in sorted(by_model_diff.items()):
        entries.sort(key=lambda x: x[1])
        errors = [e[1] for e in entries]
        median = np.median(errors)
        iqr = np.percentile(errors, 75) - np.percentile(errors, 25)

        print(f"\n  {key}  (n={len(entries)}):")
        print(f"    median={median:.4f}  IQR={iqr:.4f}  "
              f"range=[{min(errors):.4f}, {max(errors):.4f}]")

        # Flag outliers (>3× IQR from median in log space)
        log_errors = np.log10(np.maximum(errors, 1e-15))
        log_median = np.median(log_errors)
        log_iqr = np.percentile(log_errors, 75) - np.percentile(log_errors, 25)

        for seed, err, d in entries:
            bm = d.get("behavioral_metrics", {})
            flags = []
            log_e = np.log10(max(err, 1e-15))
            if log_iqr > 0 and abs(log_e - log_median) > 3 * log_iqr:
                flags.append("OUTLIER")
            if err < 1e-6:
                flags.append("suspiciously_good")
            if err > 10:
                flags.append("catastrophic")

            marker = " ".join(flags)
            marker = f"  ← {marker}" if marker else ""
            print(f"    s{seed:>3}: {err:>10.4f}  "
                  f"Q={bm.get('queries_used','?')}/{d['config'].get('max_queries','?')}  "
                  f"solves={bm.get('solve_count','?')}  "
                  f"H={bm.get('family_entropy_normalized',0):.2f}  "
                  f"waste={bm.get('wasted_turn_fraction',0):.0%}"
                  f"{marker}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    directory = sys.argv[1]
    difficulty = None
    model = None
    if "--difficulty" in sys.argv:
        difficulty = sys.argv[sys.argv.index("--difficulty") + 1]
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]

    runs = load_runs(directory, difficulty=difficulty, model=model)
    if not runs:
        print(f"No runs found in {directory}")
        sys.exit(1)

    print(f"Loaded {len(runs)} runs from {directory}")
    if difficulty:
        print(f"  filtered: difficulty={difficulty}")
    if model:
        print(f"  filtered: model contains '{model}'")

    check_config(runs)
    check_behavioral(runs)
    check_parsing(runs)
    check_control(runs)
    check_confidence(runs)
    check_errors(runs)
    check_transcripts(runs, n_sample=2)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
