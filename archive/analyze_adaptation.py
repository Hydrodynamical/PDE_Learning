"""
Analyze whether the enhanced model adapts its test function strategy
based on accumulated measurements, or applies a fixed rule.

Reads transcripts and extracts:
1. The sequence of term_integral checks (which functions were evaluated but NOT queried)
2. The weight ratios of queried vs rejected candidates
3. Whether the model's stated reasoning references previous equations

Usage:
    python analyze_adaptation.py transcript_baseline.txt transcript_enhanced.txt
"""

import re
import sys
import json
import argparse
import numpy as np


def parse_transcript(filepath):
    """
    Extract the sequence of actions from a saved transcript.

    Processes events in document order using finditer so that
    term_integral checks are correctly associated with the query
    that follows them.
    """
    with open(filepath) as f:
        text = f.read()

    # Build a list of (position, event) pairs in document order
    raw_events = []

    # term_integral results
    for m in re.finditer(
        r"Term integral \((\w+)\) for '([^']+)': ([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)",
        text
    ):
        raw_events.append((m.start(), {
            "type": "term_integral",
            "term": m.group(1),
            "spec": m.group(2),
            "value": float(m.group(3)),
        }))

    # query results (value on same or next line)
    for m in re.finditer(
        r"QUERY result for '([^']+)':\s*∫f·φ dx = ([+-]?\d+\.\d+)",
        text
    ):
        raw_events.append((m.start(), {
            "type": "query",
            "spec": m.group(1),
            "value": float(m.group(2)),
        }))

    # solve results
    for m in re.finditer(r"Solve result", text):
        raw_events.append((m.start(), {"type": "solve"}))

    # assistant reasoning blocks: text between [ASSISTANT] and next [USER]
    for m in re.finditer(r"\[ASSISTANT\](.*?)(?=\[USER\]|\Z)", text, re.DOTALL):
        reasoning = m.group(1).strip()
        if any(phrase in reasoning.lower() for phrase in [
            "still dominates",
            "better balance",
            "well balanced",
            "more balanced",
            "good balance",
            "good information about",
            "primarily constrain",
            "emphasize",
            "only about",
            "diffusion weight",
            "reaction weight",
            "advection weight",
            "diffusion:reaction",
            "diffusion term dominates",
            "ratio",
            "5x smaller",
            "6x",
            "10x",
            "100x",
            "negligible",
            "much smaller",
            "not yet constrained",
            "underrepresented",
            "need more",
            "dominated",
            "dominant",
            "constrains",
        ]):
            raw_events.append((m.start(), {
                "type": "adaptive_reasoning",
                "text": reasoning[:200],
            }))

    # Sort all events by position in the document
    raw_events.sort(key=lambda x: x[0])

    # Walk in order: emit term_integral events directly AND accumulate
    # them into checks_before for the following query.
    events = []
    current_checks = []

    for _, ev in raw_events:
        if ev["type"] == "term_integral":
            events.append(ev)          # emit for global analysis
            current_checks.append(ev)  # also track for per-query association
        elif ev["type"] == "query":
            events.append({**ev, "checks_before": list(current_checks)})
            current_checks = []
        else:
            events.append(ev)

    return events


def compute_weight_ratios(events):
    """
    Classify all pre-checked test functions globally as queried or rejected.

    A spec is "queried" if it was ever submitted as a QUERY.
    A spec is "rejected" if it was checked with term_integral but never queried.
    Only specs with all three terms measured are included.
    """
    # Collect all term_integral measurements by spec
    all_weights: dict[str, dict] = {}
    for ev in events:
        if ev["type"] == "term_integral":
            all_weights.setdefault(ev["spec"], {})[ev["term"]] = ev["value"]

    # All queried specs
    queried_specs = {ev["spec"] for ev in events if ev["type"] == "query"}

    queried_ratios = []
    rejected_ratios = []

    for spec, weights in all_weights.items():
        if "diffusion" not in weights or "reaction" not in weights:
            continue
        diff = abs(weights["diffusion"])
        react = abs(weights["reaction"])
        if react == 0:
            continue
        entry = {
            "spec": spec,
            "diff_react_ratio": diff / react,
            "diffusion": weights["diffusion"],
            "reaction": weights["reaction"],
            "advection": weights.get("advection", 0),
        }
        if spec in queried_specs:
            queried_ratios.append(entry)
        else:
            rejected_ratios.append(entry)

    return queried_ratios, rejected_ratios


def compute_rejection_rate(events):
    """
    Count how many test functions were checked with term_integral
    but never queried (truly rejected).
    """
    checked_specs = {ev["spec"] for ev in events if ev["type"] == "term_integral"}
    queried_specs = {ev["spec"] for ev in events if ev["type"] == "query"}

    rejected = checked_specs - queried_specs
    return {
        "checked": len(checked_specs),
        "queried": len(queried_specs),
        "rejected": len(rejected),
        "rejection_rate": len(rejected) / max(len(checked_specs), 1),
    }


def analyze(filepath, label):
    """Full analysis of one transcript."""
    events = parse_transcript(filepath)

    queries = [e for e in events if e["type"] == "query"]
    adaptive = [e for e in events if e["type"] == "adaptive_reasoning"]
    queried_ratios, rejected_ratios = compute_weight_ratios(events)
    rejection = compute_rejection_rate(events)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Queries submitted:          {len(queries)}")
    print(f"  Adaptive reasoning events:  {len(adaptive)}")
    print(f"  Candidates checked:         {rejection['checked']}")
    print(f"  Candidates rejected:        {rejection['rejected']}")
    print(f"  Rejection rate:             {rejection['rejection_rate']:.1%}")

    if queried_ratios:
        dr_ratios = [r["diff_react_ratio"] for r in queried_ratios]
        print(f"\n  Diffusion:Reaction ratios of QUERIED functions:")
        print(f"    Mean:   {np.mean(dr_ratios):.1f}x")
        print(f"    Median: {np.median(dr_ratios):.1f}x")
        print(f"    Min:    {np.min(dr_ratios):.1f}x")
        print(f"    Max:    {np.max(dr_ratios):.1f}x")

        # Check if ratios decrease over time (adaptive behavior)
        if len(dr_ratios) >= 4:
            first_half = np.mean(dr_ratios[:len(dr_ratios)//2])
            second_half = np.mean(dr_ratios[len(dr_ratios)//2:])
            print(f"\n  Adaptation over time:")
            print(f"    First half mean ratio:  {first_half:.1f}x")
            print(f"    Second half mean ratio: {second_half:.1f}x")
            if second_half < first_half:
                print(f"    → Model shifted toward more balanced functions")
            else:
                print(f"    → No clear adaptation trend")

    if rejected_ratios:
        rej_dr = [r["diff_react_ratio"] for r in rejected_ratios]
        print(f"\n  Diffusion:Reaction ratios of REJECTED functions:")
        print(f"    Mean:   {np.mean(rej_dr):.1f}x")
        print(f"    Median: {np.median(rej_dr):.1f}x")
        print(f"    Min:    {np.min(rej_dr):.1f}x")
        print(f"    Max:    {np.max(rej_dr):.1f}x")

        if queried_ratios:
            q_mean = np.mean([r["diff_react_ratio"] for r in queried_ratios])
            r_mean = np.mean(rej_dr)
            print(f"\n  Filtering effect:")
            print(f"    Mean ratio of queried:  {q_mean:.1f}x")
            print(f"    Mean ratio of rejected: {r_mean:.1f}x")
            if r_mean > q_mean:
                print(f"    → Model rejected functions {r_mean/q_mean:.1f}x more diffusion-dominated")
            else:
                print(f"    → No clear filtering pattern")

    if adaptive:
        print(f"\n  Sample adaptive reasoning:")
        for a in adaptive[:3]:
            print(f"    \"{a['text'][:120]}...\"")

    return {
        "label": label,
        "n_queries": len(queries),
        "n_adaptive": len(adaptive),
        "rejection": rejection,
        "queried_ratios": queried_ratios,
        "rejected_ratios": rejected_ratios,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze adaptation patterns in probe transcripts"
    )
    parser.add_argument("transcripts", nargs="+",
                        help="Transcript files to analyze")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each transcript")
    args = parser.parse_args()

    labels = args.labels or [f"Run {i+1}" for i in range(len(args.transcripts))]

    results = []
    for filepath, label in zip(args.transcripts, labels):
        results.append(analyze(filepath, label))

    if len(results) >= 2:
        print(f"\n{'=' * 60}")
        print(f"  COMPARISON")
        print(f"{'=' * 60}")
        for r in results:
            rej = r["rejection"]
            print(f"  {r['label']:20s}: {r['n_adaptive']:2d} adaptive events, "
                  f"{rej['rejected']:2d} rejections, "
                  f"{rej['rejection_rate']:.0%} rejection rate")


if __name__ == "__main__":
    main()
