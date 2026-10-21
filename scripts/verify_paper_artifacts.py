#!/usr/bin/env python3
"""Fail if checked-in research outputs no longer match their inputs or code.

This integrity check uses only the standard library and requires no archive,
network, numerical runtime, or provider credentials. It checks provenance rather
than asserting bitwise portability of floating-point experiments.
"""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'paper_artifacts'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = json.loads((ARTIFACTS/'data_manifest.json').read_text())
    with gzip.open(ARTIFACTS/'control_baselines.json.gz', 'rt') as source:
        controls = json.load(source)
    compact = json.loads((ARTIFACTS/'control_baselines_summary.json').read_text())
    audit = json.loads((ARTIFACTS/'identifiability_audit.json').read_text())
    stats = json.loads((ARTIFACTS/'statistical_analysis.json').read_text())
    for name, data in [('controls',controls), ('audit',audit), ('statistics',stats)]:
        check(data['archive_sha256'] == manifest['archive_sha256'], f'{name}: archive mismatch')
        hashes = data['method']['source_code_sha256'] if name == 'audit' else data['source_sha256']
        check(bool(hashes), f'{name}: no source provenance')
        for path, expected in hashes.items():
            check(digest(ROOT/path) == expected, f'{name}: stale output after changing {path}')
    for key in ('protocol','source_sha256','summary','regeneration_audit'):
        check(compact[key] == controls[key], f'Compact control output differs: {key}')
    check(len(controls['runs']) == 1080, 'Unexpected control policy/grid run count')
    check(len(audit['replayed_runs']) == 180, 'Unexpected archive replay count')
    check(len(stats['pairwise_success_comparisons']) == 45, 'Missing paired comparisons')
    figure_sources = json.loads((ARTIFACTS/'research_figure_sources.json').read_text())
    for name, expected in figure_sources.items():
        path = ROOT/'scripts/plot_research_audit.py' if name == 'plot_script_sha256' else ARTIFACTS/name
        check(digest(path) == expected, f'Research figures/tables are stale after changing {path.name}')
    print('Verified source, input, summary, and research-figure provenance.')


if __name__ == '__main__':
    main()
