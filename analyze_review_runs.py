#!/usr/bin/env python3
"""Analyze the pinned archive for the reviewer revision; no network/model calls.

Usage: MPLCONFIGDIR=/private/tmp/pde-review-mpl python3 analyze_review_runs.py --runs-dir data/recorded_runs

The exemplar choice is fixed independently of its outcomes: medium difficulty,
seed 1 (the smallest recorded seed), Gemini Pro and Claude Opus. Inputs are
read in place, never extracted or executed. All empirical output is regenerated
only after the input passes its structural checks. ZIP input also requires the
pinned ZIP checksum; directory input verifies local per-file hashes and records
explicitly that the historical ZIP bytes were not available for comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, median, pstdev
import zipfile

ROOT = Path(__file__).resolve().parent
DIFFICULTIES = ('easy', 'medium', 'hard')
COEFFICIENTS = ('a', 'b', 'c', 'f')
CONFIDENCE = re.compile(r'Confidence:\s*a=(\d+)%,\s*b=(\d+)%,\s*c=(\d+)%,\s*f=(\d+)%', re.I)
MODEL_LABELS = {
    'gemini-3.1-pro-preview': 'Gemini 3.1 Pro',
    'claude-opus-4-6': 'Claude Opus 4.6',
    'gpt-5.4': 'GPT-5.4',
    'gemini-3-flash-preview': 'Gemini 3 Flash',
    'claude-sonnet-4-20250514': 'Claude Sonnet 4',
    'gpt-5.4-mini': 'GPT-5.4 mini',
}
MODEL_COLORS = dict(zip(MODEL_LABELS,
    ('#226c9c', '#bd641d', '#24856b', '#7953a1', '#a74553', '#6d6d6d')))
EXEMPLAR_MODELS = ('gemini-3.1-pro-preview', 'claude-opus-4-6')
EXEMPLAR_DIFFICULTY = 'medium'
EXEMPLAR_SEED = 1
ACTIONS = ('Query', 'Solve', 'Check', 'Inspect terms', 'Evaluate', 'Submit')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def load_archive(path, manifest):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest['archive_sha256'],
            'Archive SHA-256 does not match the pinned manifest.')
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist()) == manifest['archive_members'], 'Unexpected archive members.')
        names = sorted(n for n in archive.namelist() if n.startswith('run_') and n.endswith('.json'))
        runs = [json.loads(archive.read(n)) for n in names]
        sweep = json.loads(archive.read('sweep_results.json'))
    return validate_run_records(runs, sweep, manifest)


def load_runs_directory(path, manifest):
    """Validate local records, without claiming their enclosing ZIP was verified."""
    names = sorted(p.name for p in path.glob('run_*.json'))
    require(len(names) == manifest['expected_runs'], 'Unexpected local run count.')
    local_manifest_path = path.parent/'local_run_manifest.json'
    require(local_manifest_path.is_file(), 'Local per-file hash manifest is required.')
    local_manifest = json.loads(local_manifest_path.read_text())
    recorded = {f['file']:f for f in local_manifest['files']}
    require(set(recorded) == set(names)|{'sweep_results.json'}, 'Local manifest file set differs.')
    for name in [*names, 'sweep_results.json']:
        raw = (path/name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == recorded[name]['sha256'], f'Local hash mismatch: {name}')
        require(len(raw) == recorded[name]['bytes'], f'Local size mismatch: {name}')
    runs = [json.loads((path/n).read_text()) for n in names]
    sweep = json.loads((path/'sweep_results.json').read_text())
    return validate_run_records(runs, sweep, manifest)


def validate_run_records(runs, sweep, manifest):
    require(len(runs) == manifest['expected_runs'], 'Unexpected run count.')
    require(len(sweep) == len(runs), 'Sweep summary count differs.')
    keys = [(r['config']['model'], r['config']['difficulty'], r['config']['seed']) for r in runs]
    require(len(keys) == len(set(keys)), 'Duplicate run key.')
    require({k[0] for k in keys} == set(manifest['models']), 'Unexpected model set.')
    require({k[1] for k in keys} == set(DIFFICULTIES), 'Unexpected difficulty set.')
    summaries = {(s['model'], s['difficulty'], s['seed']): s for s in sweep}
    for run, key in zip(runs, keys):
        cfg = run['config']
        require(all(cfg[k] == v for k, v in manifest['difficulty_settings'][cfg['difficulty']].items()),
                f'Unexpected settings: {key}')
        require(cfg['prompt_condition'] == manifest['prompt_condition'], f'Prompt mismatch: {key}')
        expected_effort = 'medium' if cfg['provider'] == 'openai' else None
        require(cfg['reasoning_effort'] == expected_effort, f'Reasoning-effort mismatch: {key}')
        errors = run['results']['coefficient_errors']
        require(all(finite_nonnegative(errors[k]) for k in ('a', 'b', 'c', 'f', 'total')),
                f'Invalid final error: {key}')
        require(math.isclose(sum(errors[k] for k in ('a', 'b', 'c', 'f')), errors['total'],
                             rel_tol=1e-12, abs_tol=1e-12), f'Final component sum mismatch: {key}')
        require(summaries[key]['total_coeff_error'] == errors['total'], f'Sweep mismatch: {key}')
    for difficulty in DIFFICULTIES:
        for model in manifest['models']:
            require(sorted(k[2] for k in keys if k[:2] == (model, difficulty)) == manifest['expected_seeds'],
                    f'Seed set mismatch: {model}/{difficulty}')
        for seed in manifest['expected_seeds']:
            truths = {json.dumps(r['ground_truth'], sort_keys=True) for r in runs
                      if r['config']['difficulty'] == difficulty and r['config']['seed'] == seed}
            require(len(truths) == 1, f'Ground truth differs across models: {difficulty}/{seed}')
    return runs


def requested_actions(run):
    """Recover actual command requests, without treating every request as success."""
    events, issues = [], []
    for turn in run['turns']:
        if turn.get('role') != 'assistant':
            continue
        parsed_types = []
        for raw_line in turn.get('content', '').strip().split('\n'):
            line = raw_line.strip()
            match = re.match(r'^QUERY:\s*(.+)$', line, re.I)
            if match:
                parsed_types.append('query')
                events.append({'turn': turn['turn'], 'action': 'Query', 'argument': match.group(1)})
                continue
            match = re.match(r'^COMPUTE:\s*(.+)$', line, re.I)
            if match:
                parsed_types.append('compute')
                command = match.group(1).strip()
                name = command.split()[0] if command.split() else ''
                label = {'solve': 'Solve', 'check': 'Check', 'verify': 'Check',
                         'term_integrals': 'Inspect terms', 'eval_solution': 'Evaluate'}.get(name)
                if label:
                    events.append({'turn': turn['turn'], 'action': label, 'argument': command})
                else:
                    issues.append({'turn': turn['turn'], 'unknown_compute': command})
                continue
            if re.match(r'^PREDICT:', line, re.I):
                parsed_types.append('predict')
                events.append({'turn': turn['turn'], 'action': 'Submit', 'argument': ''})
                continue
            if line:
                parsed_types.append('reasoning')
        recorded_types = turn.get('parsed_actions')
        require(recorded_types == parsed_types,
                f'Parsed action sequence differs from archive at {run["config"]}/{turn["turn"]}')
    return events, issues


def summarize_run(run):
    cfg = run['config']
    solves = sorted(run.get('solves', []), key=lambda s: (s['turn'], s.get('solve_number', 0)))
    solve_points = []
    for solve in solves:
        error = solve.get('coeff_errors', {}).get('total')
        solve_points.append({'turn': solve['turn'], 'solve_number': solve.get('solve_number'),
                             'queries': solve.get('queries_at_solve', solve.get('n_equations')),
                             'error': error if finite_nonnegative(error) else None})
    first = solve_points[0]['error'] if solve_points else None
    last = solve_points[-1]['error'] if solve_points else None
    final = run['results']['coefficient_errors']['total']
    legacy = run.get('behavioral_metrics', {}).get('improvement_ratio')
    if not solves:
        category, ratio = 'no_recorded_solve', None
    elif len(solves) == 1 and first is not None:
        category, ratio = 'single_solve', None
    elif first is None or last is None:
        category, ratio = 'invalid_endpoint_error', None
    elif last == 0:
        category, ratio = 'multi_solve_zero_final_error', None
    else:
        category, ratio = 'multi_solve_observed', first / last
    if legacy is None:
        reason = category if category != 'multi_solve_observed' else 'unexpected_missing_ratio_despite_valid_endpoints'
    else:
        reason = None
    ratio_agreement = (math.isclose(float(legacy), ratio, rel_tol=1e-10, abs_tol=1e-12)
                       if finite_nonnegative(legacy) and ratio is not None else None)
    events, issues = requested_actions(run)
    requested_counts = dict(Counter(e['action'] for e in events))
    recorded_counts = {'Query': len(run.get('queries', [])), 'Solve': len(solves),
                       'Check': len(run.get('verifications', [])),
                       'Inspect terms': len(run.get('term_integrals', []))}
    final_turn = max((t['turn'] for t in run['turns']), default=None)
    submit_requests = [e['turn'] for e in events if e['action'] == 'Submit']
    auto = run.get('behavioral_metrics', {}).get('auto_submitted')
    return {
        'model': cfg['model'], 'difficulty': cfg['difficulty'], 'seed': cfg['seed'],
        'n_recorded_solves': len(solves), 'solve_points': solve_points,
        'first_recorded_solve_error': first, 'last_recorded_solve_error': last,
        'submitted_error': final, 'submitted_success_e_lt_0.01': final < .01,
        'ratio_classification': category, 'observed_multi_solve_ratio': ratio,
        'archived_improvement_ratio': legacy, 'missing_archived_ratio_reason': reason,
        'observed_ratio_matches_archived': ratio_agreement,
        'submitted_minus_last_recorded_solve_error': final-last if last is not None else None,
        'same_strict_success_last_solve_and_submission': (last < .01) == (final < .01) if last is not None else None,
        'source_rounding_context': 'Recorded solve coefficient vectors are rounded to eight decimals; final scoring retains the internal estimate.',
        'requested_action_counts': requested_counts, 'completed_record_action_counts': recorded_counts,
        'requested_actions': events, 'unknown_compute_requests': issues,
        'family_entropy': run.get('behavioral_metrics', {}).get('family_entropy_normalized'),
        'last_recorded_turn': final_turn, 'submit_request_turns': submit_requests,
        'auto_submitted_flag': auto,
    }


def build_analysis(runs, manifest):
    records = [summarize_run(r) for r in runs]
    cells = []
    for difficulty in DIFFICULTIES:
        for model in manifest['models']:
            cell = [r for r in records if (r['model'], r['difficulty']) == (model, difficulty)]
            ratios = [r['observed_multi_solve_ratio'] for r in cell if r['observed_multi_solve_ratio'] is not None]
            cells.append({'model': model, 'difficulty': difficulty, 'n_runs': len(cell),
                          'classification_counts': dict(Counter(r['ratio_classification'] for r in cell)),
                          'missing_archived_ratio_counts_by_reason': dict(Counter(r['missing_archived_ratio_reason'] for r in cell
                                                                                 if r['missing_archived_ratio_reason'] is not None)),
                          'observed_multi_solve_ratios': ratios,
                          'observed_multi_solve_median_ratio': median(ratios) if ratios else None,
                          'successes_e_lt_0.01': sum(r['submitted_success_e_lt_0.01'] for r in cell)})
    sonnet = [r for r in records if r['model'] == 'claude-sonnet-4-20250514' and r['difficulty'] == 'medium']
    failed = [r for r in sonnet if not r['submitted_success_e_lt_0.01']]
    exemplars = [next(r for r in records if (r['model'], r['difficulty'], r['seed']) ==
                     (m, EXEMPLAR_DIFFICULTY, EXEMPLAR_SEED)) for m in EXEMPLAR_MODELS]
    differences = [abs(r['submitted_minus_last_recorded_solve_error']) for r in records
                   if r['submitted_minus_last_recorded_solve_error'] is not None]
    source_solve_errors = [s['coeff_errors']['f'] for run in runs for s in run['solves']]
    source_final_errors = [run['results']['coefficient_errors']['f'] for run in runs]
    return {
        'schema_version': 1, 'expected_historical_archive_sha256': manifest['archive_sha256'],
        'dataset_url': manifest['dataset_url'], 'analysis_kind': 'descriptive reanalysis of existing archived runs',
        'definitions': {
            'first_vs_final': 'x = first recorded solve error; y = archived submitted coefficient error. Both use evaluator ground truth.',
            'observed_improvement_ratio': 'First divided by last recorded solve error, only for >=2 recorded solves with valid endpoints and positive last error; single solves remain separate.',
            'trajectory_error_visibility': 'True coefficient errors are evaluator-only and are not numerical feedback delivered to the model.',
            'action_raster': 'A mark indicates at least one command request of that type in a turn, parsed from recorded assistant text and validated against archived parsed_actions; not necessarily a completed tool call.',
            'exemplar_selection': 'Fixed before reading outcomes: medium seed 1, smallest recorded seed, comparing Pro and Opus because of their equal aggregate medium success counts.',
            'zero_log_values': 'If zero errors occur they are explicitly counted and plotted at half the smallest positive error; no zero is silently discarded.',
        },
        'validation': {'runs': len(records), 'manifest_logical_checks_passed': True,
                       'parsed_action_sequences_match': True,
                       'first_vs_submitted_points': sum(r['first_recorded_solve_error'] is not None for r in records),
                       'first_vs_submitted_excluded_missing_first_error': sum(r['first_recorded_solve_error'] is None for r in records),
                       'single_solve_runs': sum(r['n_recorded_solves'] == 1 for r in records),
                       'multi_solve_runs': sum(r['n_recorded_solves'] >= 2 for r in records),
                       'zero_first_or_submitted_endpoints': sum(r[k] == 0 for r in records for k in
                                                               ('first_recorded_solve_error','submitted_error')),
                       'observed_ratio_mismatches': sum(r['observed_ratio_matches_archived'] is False for r in records),
                       'max_absolute_submission_minus_last_solve_error': max(differences) if differences else None,
                       'median_absolute_submission_minus_last_solve_error': median(differences) if differences else None,
                       'submission_last_solve_differences_above_2e_minus_8': sum(d > 2e-8 for d in differences),
                       'last_solve_submission_strict_success_disagreements': sum(r['same_strict_success_last_solve_and_submission'] is False for r in records)},
        'cells': cells,
        'source_confidence_exclusion_context': {
            'n_recorded_solves': len(source_solve_errors),
            'n_source_errors_below_0.1_at_recorded_solves': sum(e < .1 for e in source_solve_errors),
            'maximum_recorded_solve_source_error': max(source_solve_errors),
            'n_final_submissions': len(source_final_errors),
            'n_final_source_errors_below_0.1': sum(e < .1 for e in source_final_errors),
            'maximum_final_source_error': max(source_final_errors),
            'interpretation': 'At the confidence correctness threshold, including source f would add an always-correct outcome class in this archive. This documents an analysis rationale, not the original historical motivation.',
        },
        'sonnet_medium_joint_outcomes': {
            'n_runs': len(sonnet), 'n_failed': len(failed),
            'failed_and_ratio_gt_1': sum((r['observed_multi_solve_ratio'] or 0) > 1 for r in failed),
            'failed_and_ratio_ge_10': sum((r['observed_multi_solve_ratio'] or 0) >= 10 for r in failed),
            'failed_and_ratio_ge_100': sum((r['observed_multi_solve_ratio'] or 0) >= 100 for r in failed),
            'failed_records': [{k:r[k] for k in ('seed','first_recorded_solve_error','last_recorded_solve_error',
                                                'submitted_error','observed_multi_solve_ratio')} for r in failed],
        },
        'exemplars': exemplars, 'runs': records,
    }



# These four functions preserve the existing saved-summary definitions exactly.
def describe(values):
    """Population SD describes the archived runs, not estimation uncertainty."""
    values = [float(v) for v in values]
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "population_sd": pstdev(values),
        "min": min(values),
        "max": max(values),
    }

def causal_confidence_pairs(run):
    """Pair reports to previously returned solve outputs using transcript turns.

    Archived reports omit turn numbers, but their exact confidence strings occur
    in the assistant transcript in the same order. A solve in the report's own
    response is not yet observable to the model and is therefore excluded.
    """
    reports = run.get("metacognitive", {}).get("confidence_reports", [])
    occurrences = [(turn["turn"], match) for turn in run["turns"]
                   if turn.get("role") == "assistant"
                   for match in CONFIDENCE.finditer(turn.get("content", ""))]
    require(len(occurrences) == len(reports), "Confidence reports do not match transcript occurrence counts.")
    pairs, stats = [], Counter(reports=len(reports))
    for (turn_number, occurrence), report in zip(occurrences, reports):
        require([int(v) / 100 for v in occurrence.groups()] == [report[k] for k in COEFFICIENTS],
                "Archived confidence values disagree with the transcript.")
        eligible = [s for s in run.get("solves", [])
                    if s["turn"] < turn_number and (s.get("queries_at_solve") or s["n_equations"]) <= report["query_count"]]
        if not eligible:
            stats["reports_excluded_no_preceding_solve"] += 1
            continue
        solve = max(eligible, key=lambda s: (s["turn"], s.get("solve_number", 0)))
        stats["reports_paired"] += 1
        # Audit the legacy forward-looking matching and overlapping boundaries.
        legacy = next((s for s in run["solves"]
                       if (s.get("queries_at_solve") or s["n_equations"]) >= report["query_count"]), run["solves"][-1])
        if legacy.get("solve_number") != solve.get("solve_number"):
            stats["reports_with_different_solve_from_legacy"] += 1
        for coefficient in ("a", "b", "c"):
            confidence = float(report[coefficient])
            require(0 <= confidence <= 1, "Confidence is outside [0, 1].")
            error = float(solve["coeff_errors"][coefficient])
            pairs.append((confidence, int(error < 0.1)))
            if 0 < confidence < 1 and math.isclose(confidence * 10, round(confidence * 10), abs_tol=1e-9):
                stats["observations_on_shared_legacy_bin_boundaries"] += 1
    stats["coefficient_observations"] = len(pairs)
    return pairs, stats

def calibration_summary(pairs):
    """Disjoint bins [lo, hi), with 1 included in the final bin."""
    bins = [[] for _ in range(10)]
    for confidence, success in pairs:
        index = min(int(confidence * 10), 9)
        bins[index].append((confidence, success))
    return [{
        "lower": i / 10,
        "upper": (i + 1) / 10,
        "upper_inclusive": i == 9,
        "n": len(values),
        "successes": sum(v[1] for v in values),
        "mean_confidence": mean(v[0] for v in values) if values else None,
        "success_fraction": mean(v[1] for v in values) if values else None,
    } for i, values in enumerate(bins)]

def build_summary(runs, manifest):
    summary = {
        "schema_version": 1,
        "archive_sha256": manifest["archive_sha256"],
        "dataset_url": manifest["dataset_url"],
        "validation": {
            "runs": len(runs), "scored_runs": len(runs), "models": len(manifest["models"]),
            "difficulties": list(DIFFICULTIES), "seeds_per_cell": manifest["expected_seeds"],
            "shared_ground_truth_per_seed_difficulty": True,
            "sweep_summary_matches_all_run_scores": True,
            "attempts_and_retries_before_archive": "not recorded",
        },
        "settings": {
            "difficulty_settings": manifest["difficulty_settings"],
            "prompt_condition": manifest["prompt_condition"],
            "reasoning_effort": {"openai": "medium", "anthropic": None, "google": None},
            "run_timestamp_range": [min(r["config"]["timestamp"] for r in runs), max(r["config"]["timestamp"] for r in runs)],
            "run_timestamp_timezone": "not recorded",
        },
        "definitions": {
            "coefficient_error": "E_j = max_l abs(predicted_theta[j,l] - true_theta[j,l]); absolute coefficient error, not relative error",
            "total_error": "E = E_a + E_b + E_c + E_f; strict threshold comparison",
            "improvement_ratio": "first solve total error / last solve total error; missing values assigned 1 to retain legacy plotting convention",
            "family_entropy": "-sum(p_i log2(p_i)) / log2(5), over the final solve taxonomy: polynomial/trigonometric/exponential/localized/other; later queries are omitted",
            "action_stack": "separate per-action medians, summed; not the median total action count",
            "population_sd": "ddof=0 across the ten archived runs; descriptive variation, not a confidence interval",
            "calibration": "pool a/b/c reports; pair with latest solve on an earlier transcript turn with queries_at_solve <= report query count; omit reports without prior solve; event is E_j < 0.1",
            "calibration_bins": "disjoint [lo,hi), with 1 in the final bin; plot mean stated confidence against empirical success fraction; marker area in points squared is 5 + 1.1 times bin count",
        },
        "cells": [],
        "calibration_totals": {},
        "changes_from_legacy_assets": [
            "GPT-5.4 mini hard success at E < 0.01 is 2/10 in both run logs and sweep_results.json; the checked-in PNG displays 0/10.",
            "Calibration uses causally available solve outputs and disjoint confidence bins, replacing forward-looking solve matching and overlapping bin boundaries.",
            "Improvement figure shows individual runs and medians on a logarithmic axis; it omits arithmetic-mean bars and standard-deviation whiskers."
        ],
    }
    total_calibration = Counter()
    for difficulty in DIFFICULTIES:
        for model in manifest["models"]:
            cell_runs = [r for r in runs if r["config"]["difficulty"] == difficulty and r["config"]["model"] == model]
            errors = [r["results"]["coefficient_errors"]["total"] for r in cell_runs]
            metrics = [r["behavioral_metrics"] for r in cell_runs]
            improvement = [m.get("improvement_ratio") if m.get("improvement_ratio") is not None else 1.0 for m in metrics]
            pairs, counts = [], Counter()
            for run in cell_runs:
                run_pairs, run_counts = causal_confidence_pairs(run)
                pairs.extend(run_pairs)
                counts.update(run_counts)
            total_calibration.update(counts)
            summary["cells"].append({
                "model": model, "difficulty": difficulty, "n": len(cell_runs),
                "success": {str(t): {"count": sum(e < t for e in errors), "n": len(errors), "rate": sum(e < t for e in errors) / len(errors)} for t in (0.01, 0.1)},
                "total_error": describe(errors),
                "coefficient_errors": {k: describe([r["results"]["coefficient_errors"][k] for r in cell_runs]) for k in COEFFICIENTS},
                "solves": describe([m.get("solve_count", 0) or 0 for m in metrics]),
                "checks": describe([m.get("check_count", 0) or 0 for m in metrics]),
                "term_integrals": describe([m.get("term_integral_count", 0) or 0 for m in metrics]),
                "budget_utilization": describe([m["budget_utilization"] for m in metrics]),
                "improvement_ratio": describe(improvement),
                "improvement_ratio_imputed_runs": sum(m.get("improvement_ratio") is None for m in metrics),
                "family_entropy": describe([m.get("family_entropy_normalized", 0) or 0 for m in metrics]),
                "calibration_support": dict(counts),
                "calibration_bins": calibration_summary(pairs),
            })
    summary["calibration_totals"] = dict(total_calibration)
    return summary


def compare_recursive(expected, actual, path='', mismatches=None):
    """Compare every regenerated summary value, allowing only floating roundoff."""
    if mismatches is None:
        mismatches=[]
    if isinstance(expected,dict) and isinstance(actual,dict):
        if expected.keys()!=actual.keys():
            mismatches.append({'path':path,'problem':'keys differ','expected':sorted(expected),'actual':sorted(actual)})
        for key in expected.keys() & actual.keys():
            compare_recursive(expected[key],actual[key],f'{path}/{key}',mismatches)
    elif isinstance(expected,list) and isinstance(actual,list):
        if len(expected)!=len(actual):
            mismatches.append({'path':path,'problem':'list length differs'})
        for i,(a,b) in enumerate(zip(expected,actual)):
            compare_recursive(a,b,f'{path}/{i}',mismatches)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool) and isinstance(actual,(int,float)):
        if not math.isclose(expected,actual,rel_tol=1e-12,abs_tol=1e-12):
            mismatches.append({'path':path,'expected':expected,'actual':actual})
    elif expected != actual:
        mismatches.append({'path':path,'expected':expected,'actual':actual})
    return mismatches


def validate_saved_outputs(runs,manifest,analysis):
    saved=json.loads((ROOT/'paper_artifacts/summary.json').read_text())
    regenerated=build_summary(runs,manifest)
    differences=compare_recursive(saved,regenerated)
    require(not differences,f'Regenerated summary differs from saved summary: {differences[:5]}')
    analysis['validation']['complete_regenerated_summary_matches_saved']=True
    analysis['validation']['summary_comparison_float_tolerance']={'relative':1e-12,'absolute':1e-12}
    saved_stats=json.loads((ROOT/'paper_artifacts/statistical_analysis.json').read_text())
    by_cell={(c['model'],c['difficulty']):c for c in saved_stats['cells']}
    checked=0
    for run in runs:
        cfg=run['config']; pairs,_=causal_confidence_pairs(run)
        reference={r['seed']:r for r in by_cell[cfg['model'],cfg['difficulty']]['confidence']['runs']}
        if not pairs:
            require(cfg['seed'] not in reference,'Unsupported confidence run present in saved stats.')
            continue
        require(cfg['seed'] in reference,'Supported confidence run absent from saved stats.')
        derived={'seed':cfg['seed'],'coefficient_observations':len(pairs),
                 'brier_score':mean((p-y)**2 for p,y in pairs),
                 'mean_confidence':mean(p for p,y in pairs),
                 'success_fraction':mean(y for p,y in pairs),
                 'signed_calibration_in_the_large_gap':mean(p-y for p,y in pairs)}
        differences=compare_recursive(reference[cfg['seed']],derived)
        require(not differences,f'Per-run confidence statistics differ: {cfg}/{differences[:5]}')
        checked+=1
    analysis['validation']['per_run_confidence_records_matched']=checked


def setup_plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.titlesize':10,
                         'axes.labelsize':9, 'xtick.labelsize':8.5, 'ytick.labelsize':8.5,
                         'pdf.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False})
    return plt


def save(fig, basename, plt):
    fig.savefig(ROOT/'images'/f'{basename}.pdf', bbox_inches='tight',
                metadata={'Author':'Joseph K. Miller', 'CreationDate':None, 'ModDate':None})
    fig.savefig(ROOT/'images'/f'{basename}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_trajectory(analysis, plt):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(2, 2, figsize=(6.75,4.8), sharex='col',
                             gridspec_kw={'height_ratios':[1.35,1], 'hspace':.18, 'wspace':.42})
    positive = [s['error'] for r in analysis['exemplars'] for s in r['solve_points'] if s['error'] and s['error'] > 0]
    positive += [r['submitted_error'] for r in analysis['exemplars'] if r['submitted_error'] > 0]
    floor = min(positive)/2
    ymin, ymax = min(floor,.01/2), max(max(positive)*2,.01*2)
    common_last_turn = max(r['last_recorded_turn'] for r in analysis['exemplars'])
    for col, run in enumerate(analysis['exemplars']):
        ax, raster = axes[:,col]
        color = MODEL_COLORS[run['model']]
        points = [s for s in run['solve_points'] if s['error'] is not None]
        ax.plot([s['turn'] for s in points], [max(s['error'],floor) for s in points],
                '-o', color=color, lw=1.8, ms=5, zorder=3)
        endpoint = run['last_recorded_turn']
        ax.scatter([endpoint],[max(run['submitted_error'],floor)],marker='D',s=42,
                    edgecolor=color,facecolor='white',linewidth=1.5,zorder=4)
        ax.axhline(.01, color='#444444', ls='--', lw=1)
        ax.set_yscale('log'); ax.set_ylim(ymin,ymax)
        ax.set_title(f'{"AB"[col]}  {MODEL_LABELS[run["model"]]}', loc='left', pad=10)
        ax.grid(axis='y',alpha=.16)
        if col == 0: ax.set_ylabel('Coefficient error E\n(evaluator only)')
        info = 'Auto-submitted' if run['auto_submitted_flag'] else 'Submitted'
        ax.annotate(f'{info}: {run["submitted_error"]:.2g}',
                    (endpoint,max(run['submitted_error'],floor)), xytext=(-5,12),
                    textcoords='offset points',ha='right',fontsize=8.5,color=color)
        events = Counter((e['turn'],e['action']) for e in run['requested_actions'])
        for (turn, action), number in events.items():
            raster.scatter([turn],[ACTIONS.index(action)],s=25,color=color,zorder=3)
        raster.set_yticks(range(len(ACTIONS)), ACTIONS)
        raster.invert_yaxis(); raster.set_ylim(len(ACTIONS)-.5,-.5)
        raster.set_xlabel('Assistant turn')
        raster.set_xlim(.3,max(common_last_turn+1,3))
        raster.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=7))
        raster.grid(axis='x',alpha=.15)
    fig.legend(handles=[Line2D([0],[0],marker='o',color='#555',lw=1.5,label='Recorded solve'),
                        Line2D([0],[0],marker='D',color='#555',markerfacecolor='white',lw=0,label='Final submitted score'),
                        Line2D([0],[0],color='#444',ls='--',label='Success threshold E < 0.01')],
               loc='upper center',bbox_to_anchor=(.53,.98),ncol=3,frameon=False,fontsize=8)
    fig.text(.53,.025,'Medium task, seed 1. Raster marks command requests, not certified execution.',
             ha='center',fontsize=8)
    fig.subplots_adjust(top=.86,bottom=.14,left=.14,right=.985)
    save(fig,'review_trajectory',plt)


def plot_error_change(analysis, plt):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator
    eligible = [r for r in analysis['runs'] if r['first_recorded_solve_error'] is not None]
    positive = [r[k] for r in eligible for k in ('first_recorded_solve_error','submitted_error') if r[k]>0]
    floor = min(positive)/2
    zero_count = sum(r[k] == 0 for r in eligible for k in ('first_recorded_solve_error','submitted_error'))
    lo = 10**math.floor(math.log10(floor if zero_count else min(positive)))
    hi = 10**math.ceil(math.log10(max(positive)))
    fig, axes = plt.subplots(1,3,figsize=(6.75,3.3),sharex=True,sharey=True)
    for index,(difficulty,ax) in enumerate(zip(DIFFICULTIES,axes)):
        ax.plot([lo,hi],[lo,hi],color='#a0a0a0',lw=1,zorder=0)
        ax.axhline(.01,color='#444',ls='--',lw=1); ax.axvline(.01,color='#444',ls=':',lw=.8)
        group=[r for r in eligible if r['difficulty']==difficulty]
        for model in MODEL_LABELS:
            for single,marker in [(False,'o'),(True,'x')]:
                cell=[r for r in group if r['model']==model and (r['n_recorded_solves']==1)==single]
                if not cell: continue
                kwargs={'s':30,'alpha':.78,'marker':marker,'color':MODEL_COLORS[model],'linewidths':1}
                if not single: kwargs.update(edgecolors='white',linewidths=.45)
                ax.scatter([max(r['first_recorded_solve_error'],floor) for r in cell],
                           [max(r['submitted_error'],floor) for r in cell],**kwargs)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
        ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4))
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
        ax.set_aspect('equal',adjustable='box')
        ax.set_title(f'{"ABC"[index]}  {difficulty.capitalize()} (n={len(group)})',loc='left')
        ax.grid(alpha=.12)
        ax.set_xlabel('First solve error')
    axes[0].set_ylabel('Final submitted error')
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=MODEL_COLORS[m],
                    markeredgecolor='none',label=label,markersize=6) for m,label in MODEL_LABELS.items()]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.17),ncol=3,frameon=False,fontsize=8)
    note='Circles: multiple solves. Crosses: one solve. Diagonal: equal error. Dashed: final success.'
    if zero_count:
        note+=f' {zero_count} zero endpoints plotted at {floor:.1g}.'
    fig.text(.53,.015,note,
             ha='center',fontsize=8)
    fig.subplots_adjust(top=.9,bottom=.29,left=.10,right=.99,wspace=.16)
    save(fig,'review_error_change',plt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--archive',type=Path)
    inputs.add_argument('--runs-dir',type=Path)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'paper_artifacts/data_manifest.json').read_text())
    runs=load_archive(args.archive,manifest) if args.archive else load_runs_directory(args.runs_dir,manifest)
    analysis=build_analysis(runs,manifest)
    analysis['validation']['local_per_file_hashes_verified'] = bool(args.runs_dir)
    analysis['input_provenance'] = {
        'mode':'pinned_zip' if args.archive else 'local_recorded_runs',
        'historical_zip_checksum_verified': bool(args.archive),
        'historical_zip_checksum_note': 'Verified against pinned manifest.' if args.archive else
            'Historical ZIP bytes unavailable; no claim of byte-for-byte identity with that ZIP. Local files pass logical checks and saved-summary comparisons.',
        'local_file_manifest': None if args.archive else json.loads((args.runs_dir.parent/'local_run_manifest.json').read_text()),
    }
    validate_saved_outputs(runs,manifest,analysis)
    require(analysis['validation']['observed_ratio_mismatches']==0,'Observed ratio differs from archived ratio.')
    plt=setup_plotting()
    plot_trajectory(analysis,plt)
    plot_error_change(analysis,plt)
    (ROOT/'paper_artifacts/review_run_analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'validation':analysis['validation'],
                      'sonnet_medium_joint_outcomes':analysis['sonnet_medium_joint_outcomes'],
                      'exemplars':[{k:r[k] for k in ('model','difficulty','seed','n_recorded_solves','submitted_error',
                                                    'requested_action_counts','completed_record_action_counts','auto_submitted_flag')}
                                   for r in analysis['exemplars']]},indent=2))


if __name__=='__main__':
    main()
