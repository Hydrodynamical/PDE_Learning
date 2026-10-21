#!/usr/bin/env python3
"""Plot new offline controls and audit results, keeping data sources separate.

Run the three analysis scripts first; this script only reads their derived data.
It makes no network or provider calls. Figure-source checksums are saved beside
its TeX tables so figures can be traced to an exact analysis output.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIFFICULTIES = ('easy', 'medium', 'hard')
COLORS = ('#0072B2', '#D55E00', '#009E73')
LABELS = {
    'gemini-3.1-pro-preview': 'Gemini 3.1 Pro',
    'claude-opus-4-6': 'Claude Opus 4.6',
    'gpt-5.4': 'GPT-5.4',
    'gemini-3-flash-preview': 'Gemini 3 Flash',
    'claude-sonnet-4-20250514': 'Claude Sonnet 4',
    'gpt-5.4-mini': 'GPT-5.4 mini',
}
STRATEGY_LABELS = {'fixed_sine': 'Fixed sine', 'boundary_legendre': 'Boundary Legendre',
                   'random_mixed': 'Random mixed'}


def read_json(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as source:
        return json.load(source)


def configure():
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
                         'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.bbox': 'tight', 'savefig.pad_inches': .03})


def plot_numerics(audit, output):
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.55), layout='constrained')
    rng = np.random.default_rng(108)
    for index, difficulty in enumerate(DIFFICULTIES):
        values = [row['all_stored_queries']['unit_column_norm']['min_over_max_singular_value']
                  for row in audit['replayed_runs'] if row['difficulty'] == difficulty]
        axes[0].scatter(index + rng.uniform(-.17, .17, len(values)), values,
                        s=12, alpha=.7, color=COLORS[index], edgecolor='none')
        axes[0].plot([index - .2, index + .2], [np.median(values)] * 2, color='black', lw=1.6)
    axes[0].axhline(1e-8, ls='--', color='.4', lw=.8)
    axes[0].set(yscale='log', ylabel=r'$\sigma_{\min}/\sigma_{\max}$ (unit columns)',
                xticks=range(3), xticklabels=[d.title() for d in DIFFICULTIES])
    axes[0].grid(axis='y', alpha=.2)
    for index, row in enumerate(audit['summary']['difficulty_groups']):
        entries = row['amplitude_sensitivity']
        axes[1].plot([v['common_amplitude'] for v in entries],
                     [v['successes_at_001'] / v['n'] for v in entries],
                     marker=('o','s','^')[index], color=COLORS[index], label=row['difficulty'].title())
    axes[1].set(xscale='log', xlabel='Common amplitude before 12dp rounding',
                ylabel=r'Refit success fraction ($E<0.01$)', ylim=(0, 1.04),
                xticks=[1e-3, 1, 1e3, 1e6])
    axes[1].legend(loc='lower right', frameon=False)
    axes[1].grid(alpha=.2)
    fig.savefig(output / 'paper_numerical_audit.pdf')
    plt.close(fig)


def plot_controls(controls, output):
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.45), sharey=True, layout='constrained')
    for ax, difficulty in zip(axes, DIFFICULTIES):
        for index, strategy in enumerate(STRATEGY_LABELS):
            cell = next(row for row in controls['summary'] if row['difficulty'] == difficulty
                        and row['strategy'] == strategy and row['grid_factor'] == 1)
            points = cell['checkpoints']
            xs = [p['queries'] for p in points]
            ys = [p['total_error']['success_e_lt_0.01'] / p['total_error']['n'] for p in points]
            ax.plot(xs, ys, color=COLORS[index], ls=('-', '--', ':')[index],
                    lw=1.6, marker=('o','s','^')[index], ms=3, label=STRATEGY_LABELS[strategy])
            if strategy == 'random_mixed':
                runs = [r for r in controls['runs'] if r['difficulty'] == difficulty
                        and r['strategy'] == strategy and r['grid_factor'] == 1]
                by_rep = np.asarray([[np.mean([r['checkpoints'][j]['coefficient_errors']['total'] < .01
                                               for r in runs if r['replicate'] == rep])
                                      for j in range(len(points))]
                                     for rep in range(controls['protocol']['random_replicates'])])
                ax.fill_between(xs, by_rep.min(axis=0), by_rep.max(axis=0),
                                color=COLORS[index], alpha=.12, linewidth=0)
        ax.set(title=difficulty.title(), xlabel='Stored queries', ylim=(-.04, 1.06))
        ax.set_xticks([xs[0], (xs[0]+xs[-1])//2, xs[-1]])
        ax.grid(alpha=.2)
    axes[0].set_ylabel(r'Success fraction ($E<0.01$)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.savefig(output / 'paper_controls.pdf')
    plt.close(fig)


def plot_confidence(statistics, output):
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 4.25), layout='constrained')
    for col, difficulty in enumerate(DIFFICULTIES):
        cells = [row for row in statistics['cells'] if row['difficulty'] == difficulty]
        for row_index, metric in enumerate(('brier_score','signed_calibration_in_the_large_gap')):
            ax = axes[row_index, col]
            for index, cell in enumerate(cells):
                data = cell['confidence']['run_balanced'][metric]
                value = data['estimate']
                lo, hi = data['whole_run_bootstrap_95_interval']
                ax.errorbar(value, index, xerr=[[max(0, value-lo)], [max(0, hi-value)]],
                            fmt='o', ms=3, capsize=2, color=COLORS[col], lw=1)
            ax.set(yticks=range(len(cells)), ylim=(len(cells)-.5, -.5))
            ax.set_yticklabels([LABELS[c['model']] for c in cells] if col == 0 else [])
            if row_index == 0:
                ax.set_title(difficulty.title())
                ax.set_xlabel('Run-balanced Brier score')
                ax.set_xlim(0, .7)
            else:
                ax.axvline(0, color='.5', ls='--', lw=.7)
                ax.set_xlabel('Confidence − correctness')
                ax.set_xlim(-.6, .8)
            ax.grid(axis='x', alpha=.2)
    fig.savefig(output / 'paper_confidence_uncertainty.pdf')
    plt.close(fig)


def make_tables(controls, statistics, destination):
    lines = [r'% Generated by scripts/plot_research_audit.py; do not edit numbers manually.',
             r'\begin{table}[t]', r'\caption{New offline controls at the original resolution and full stored-query budget. Deterministic policies have ten tasks per difficulty. Random rows summarize ten query-list replicates on the same ten tasks; the denominator 100 is not 100 independent PDEs.}',
             r'\label{tab:controls}', r'\centering\small',
             r'\begin{tabular}{@{}llrr@{}}', r'\toprule',
             r'Policy & Setting & Successes & Median $E$\\', r'\midrule']
    for strategy in STRATEGY_LABELS:
        for difficulty in DIFFICULTIES:
            row = next(r for r in controls['summary'] if r['difficulty'] == difficulty
                       and r['strategy'] == strategy and r['grid_factor'] == 1)
            scores = row['checkpoints'][-1]['total_error']
            lines.append(f"{STRATEGY_LABELS[strategy]} & {difficulty.title()} & "
                         f"{scores['success_e_lt_0.01']}/{scores['n']} & {scores['median']:.2g}\\\\")
        lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    destination.write_text('\n'.join(lines))
    lines = [r'% Generated by scripts/plot_research_audit.py.',
             r'\begin{table}[ht]', r'\caption{Strict-threshold archived successes with Wilson 95\% intervals. Each cell uses ten scored tasks; intervals cannot account for missing attempts or model-sampling variability.}',
             r'\centering\small', r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
             r'Model & Easy & Medium & Hard\\', r'\midrule']
    for model, label in LABELS.items():
        values = []
        for difficulty in DIFFICULTIES:
            cell = next(c for c in statistics['cells'] if c['model'] == model and c['difficulty'] == difficulty)['success']['0.01']
            lo, hi = cell['wilson_95_interval']
            values.append(f"{cell['count']}/10 [{lo:.2f}, {hi:.2f}]")
        lines.append(label+' & '+' & '.join(values)+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '',
              r'\begin{table}[ht]', r'\caption{Exploratory exact paired success tests, Holm-adjusted within the 15 model pairs of each difficulty. All 45 comparisons are shown. Values are conditional on the retained scored tasks; no adjustment is made across difficulties.}',
              r'\centering\small', r'\begin{tabular}{@{}llrrr@{}}', r'\toprule',
              r'First model & Second model & Easy & Medium & Hard\\', r'\midrule']
    pairs = [(r['first_model'],r['second_model']) for r in statistics['pairwise_success_comparisons'] if r['difficulty']=='easy']
    for first, second in pairs:
        values=[]
        for difficulty in DIFFICULTIES:
            row=next(r for r in statistics['pairwise_success_comparisons'] if r['difficulty']==difficulty and r['first_model']==first and r['second_model']==second)
            values.append(f"{row['holm_adjusted_p_within_difficulty']:.4f}")
        lines.append(LABELS[first]+' & '+LABELS[second]+' & '+' & '.join(values)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    destination.with_name('statistical_tables.tex').write_text('\n'.join(lines))


def generalization_table(data, destination):
    lines = [r'% Generated by scripts/plot_research_audit.py.',
             r'\begin{table}[t]',
             r'\caption{Fixed sine on 100 fresh seeds per setting. All planned tasks remain in the denominator. Parentheses give Wilson 95\% intervals for the minimum-budget rate; zero-source counts are descriptive.}',
             r'\label{tab:fresh}',r'\centering\small',
             r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',
             r'Setting & $4N$ successes & $6N$ successes & $f=0$\\',r'\midrule']
    for row in data['summary']:
        minimum, full = row['minimum_queries'], row['full_budget']
        lo, hi = minimum['wilson_95_interval']
        lines.append(f"{row['difficulty'].title()} & {minimum['successes_e_lt_0.01']}/100 & "
                     f"{full['successes_e_lt_0.01']}/100 & {row['zero_source_tasks']}\\\\")
        lines.append(f" & ({lo:.2f}, {hi:.2f}) & & \\\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    destination.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_controls = ROOT / 'paper_artifacts/control_baselines.json.gz'
    if not default_controls.exists():
        default_controls = ROOT / 'paper_artifacts/control_baselines.json'
    parser.add_argument('--controls', type=Path, default=default_controls)
    parser.add_argument('--audit', type=Path, default=ROOT/'paper_artifacts/identifiability_audit.json')
    parser.add_argument('--statistics', type=Path, default=ROOT/'paper_artifacts/statistical_analysis.json')
    parser.add_argument('--generalization', type=Path, default=ROOT/'paper_artifacts/sine_generalization.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'images')
    args=parser.parse_args()
    controls, audit, statistics = [read_json(p) for p in (args.controls,args.audit,args.statistics)]
    if len({v['archive_sha256'] for v in (controls,audit,statistics)}) != 1:
        raise ValueError('Derived inputs do not refer to the same pinned archive.')
    generalization = read_json(args.generalization)
    configure()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plot_numerics(audit,args.output_dir)
    plot_controls(controls,args.output_dir)
    plot_confidence(statistics,args.output_dir)
    make_tables(controls,statistics,ROOT/'paper_artifacts/control_table.tex')
    generalization_table(generalization,ROOT/'paper_artifacts/sine_generalization_table.tex')
    provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.controls,args.audit,args.statistics,args.generalization)}
    provenance['plot_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (ROOT/'paper_artifacts/research_figure_sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Saved three audit figures and numerical tables.')

if __name__=='__main__':
    main()
