"""Offline integrity and arithmetic checks; Python standard library only.

This checks the saved evidence, not GPU execution or biological correctness.
Run: python3 verify.py [--output recomputed.json]
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads((ROOT / path).read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_samples(value, path):
    count = 0
    if isinstance(value, dict):
        for sample_key, median_key in [('samples', 'median'), ('samples_ms', 'median_ms')]:
            if sample_key in value and median_key in value:
                samples = value[sample_key]
                check(bool(samples) and all(math.isfinite(x) for x in samples), f'{path}: nonfinite/empty samples')
                check(math.isclose(statistics.median(samples), value[median_key], rel_tol=1e-10, abs_tol=1e-9), f'{path}: median mismatch')
                count += 1
        for key, item in value.items():
            count += verify_samples(item, f'{path}/{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, (dict, list)):
                count += verify_samples(item, f'{path}/{index}')
    return count


def junit(path):
    tree = ET.parse(ROOT / path)
    counts = dict(passed=0, failed=0, errors=0, skipped=0, xfailed=0)
    cases = {}
    for case in tree.iter('testcase'):
        identity = case.attrib['classname'] + '::' + case.attrib['name']
        check(identity not in cases, f'{path}: duplicate test identity {identity}')
        skip = case.find('skipped')
        if case.find('failure') is not None:
            outcome = 'failed'
        elif case.find('error') is not None:
            outcome = 'errors'
        elif skip is not None:
            outcome = 'xfailed' if skip.attrib.get('type') == 'pytest.xfail' else 'skipped'
        else:
            outcome = 'passed'
        cases[identity] = outcome
        counts[outcome] += 1
    return counts, cases


def aba(before, candidate, after):
    control = statistics.mean([before, after])
    return dict(before_ms=before, candidate_ms=candidate, after_ms=after,
                latency_reduction_percent=100*(1-candidate/control),
                speedup=control/candidate, control_drift_percent=100*(after/before-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = load('MANIFEST.json')
    timing_sets = 0
    paths = set()
    for entry in manifest['files']:
        path = entry['path']
        check(path not in paths, f'duplicate manifest path {path}')
        paths.add(path)
        resolved = (ROOT/path).resolve()
        check(resolved.is_relative_to(ROOT), f'path outside bundle: {path}')
        raw = resolved.read_bytes()
        check(len(raw) == entry['published_bytes'], f'{path}: byte length differs')
        check(hashlib.sha256(raw).hexdigest() == entry['published_sha256'], f'{path}: SHA256 differs')
        if path.endswith('.json'):
            timing_sets += verify_samples(json.loads(raw), path)
    for category in ['data', 'probes']:
        actual = {str(p.relative_to(ROOT)) for p in (ROOT/category).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        check(actual <= paths, f'unmanifested files: {actual-paths}')

    report = dict(manifest_files=len(paths), checked_timing_sample_sets=timing_sets, junit={}, benchmarks={})
    counts, _ = junit('data/validation/run-1/locked-pytest.xml')
    check(counts == dict(passed=592, failed=0, errors=0, skipped=3, xfailed=7), f'CPU counts: {counts}')
    report['junit']['locked_cpu'] = counts
    for name, path in [('esm3_torch211','data/validation/run-1/locked-esm3-regression.xml'), ('esm3_torch215','data/validation/run-2/performance-esm3-regression.xml')]:
        counts, _ = junit(path)
        check(counts == dict(passed=24, failed=0, errors=0, skipped=0, xfailed=0), f'{name} counts: {counts}')
        report['junit'][name] = counts
    old_counts, old = junit('data/validation/run-1/locked-small-gpu.xml')
    new_counts, new = junit('data/validation/run-3/gpu-tests.xml')
    check(old_counts == dict(passed=40, failed=9, errors=0, skipped=0, xfailed=0), f'initial GPU counts: {old_counts}')
    check(new_counts == dict(passed=59, failed=0, errors=0, skipped=0, xfailed=0), f'final GPU counts: {new_counts}')
    check(set(old) <= set(new), 'final run omitted an original test')
    check(all(new[name] == 'passed' for name, outcome in old.items() if outcome == 'failed'), 'an original failure did not pass')
    report['junit'].update(initial_gpu=old_counts, repaired_gpu=new_counts, all_original_cases_retained=True,
                           original_failures_now_passing=[name for name,outcome in old.items() if outcome=='failed'])

    capture = load('data/capture/patched.json')['timings']
    med = lambda name: statistics.median(capture[name]['wall_ms']['samples'])
    fixed = statistics.mean([med('graph_fixed_before'),med('graph_fixed_after')])
    report['benchmarks']['capture_h2d'] = aba(med('eager_host_before'), med('graph_pinned_host_copy'), med('eager_host_after'))
    report['benchmarks']['capture_h2d']['copy_increment_us_vs_fixed_medians'] = 1000*(med('graph_pinned_host_copy')-fixed)
    for category in ['swiglu-original','swiglu-rerun']:
        for path in sorted((ROOT/'data'/category).glob('*.json')):
            record = json.loads(path.read_text())
            check(record['status'] == 'passed', f'{path.name}: status')
            check(record.get('exact_output_equality',record.get('bitwise_equivalent')) is True, f'{path.name}: exact equality')
            times = [statistics.median(record[key]['samples_ms']) for key in ['graph_before','candidate_graph','graph_after']]
            check(all(len(record[key]['samples_ms']) == record['options']['repeats'] for key in ['graph_before','candidate_graph','graph_after']),f'{path.name}: sample count')
            report['benchmarks'][category+'/'+path.stem] = aba(*times)
    for path in sorted((ROOT/'data/recipe-validation').glob('*.json')):
        record = json.loads(path.read_text())
        report.setdefault('recipe_numerical_status',{})[path.stem] = record['status']
    historic = subprocess.run([sys.executable, str(ROOT/'probes/historical/recompute_historical.py')], check=True, text=True, capture_output=True)
    report['historical'] = json.loads(historic.stdout)
    report['scope'] = 'Integrity and saved-record arithmetic only. No GPU rerun, source-authenticity attestation, statistical independence, or biological-quality validation.'
    text = json.dumps(report, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
