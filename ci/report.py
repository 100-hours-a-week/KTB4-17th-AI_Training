"""Combine preflight, fixture JUnit and validator-only coverage; fail on missing evidence."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from ci.evidence import CONTRACTS, DATASETS, identity, strict_json


def identifier(value):
    return isinstance(value, str) and re.fullmatch(r'[a-z][a-z0-9_]*', value) is not None


def valid_checks(checks):
    # Every field consumed by Markdown must be present and validated before publication.
    return isinstance(checks, list) and bool(checks) and all(
        isinstance(c, dict) and c.get('status') in ('passed', 'failed')
        and identifier(c.get('name')) and isinstance(c.get('metrics'), dict)
        and all(identifier(k) and type(v) in (int, float) and (type(v) is int or math.isfinite(v)) and (v >= 0 or k == 'exit_code') for k, v in c['metrics'].items())
        and type(c.get('duration_seconds')) in (int, float) and c['duration_seconds'] >= 0
        and all(isinstance(c.get(k, []), list) and all(
            isinstance(e, dict) and identifier(e.get('code'))
            and type(e.get('count')) is int and e['count'] > 0
            for e in c.get(k, [])) for k in ('errors', 'warnings'))
        for c in checks)


ERROR_METRICS = {'unresolved_source_ids':'unresolved_source_ids','unresolved_families':'unresolved_families',
                 'duplicate_ids':'duplicate_ids','duplicate_content':'duplicate_content','family_overlap':'family_leakage',
                 'prompt_overlap':'prompt_leakage','source_label_mismatches':'source_label_mismatch'}
PASS_METRICS = {'source_rows','teacher_rows','train_rows','val_rows','train_families','val_families','joined_rows',
                'declared_skipped_rows','source_labels_compared'} | set(ERROR_METRICS)


def consistent(check, dataset=False):
    errors, metrics = check.get('errors',[]), check['metrics']
    if (check['status']=='failed') != bool(errors):
        return False
    if dataset:
        for metric,code in ERROR_METRICS.items():
            value = metrics.get(metric,0)
            if value and (check['status']=='passed' or sum(e['count'] for e in errors if e['code']==code)!=value):
                return False
        if check['status']=='passed':
            if not PASS_METRICS <= set(metrics) or not all(type(metrics[k]) is int for k in PASS_METRICS):
                return False
            total = metrics['train_rows']+metrics['val_rows']
            if (min(metrics['train_rows'],metrics['val_rows'],metrics['teacher_rows'])<=0
                    or metrics['joined_rows']!=total or metrics['source_labels_compared']!=total
                    or metrics['source_rows']!=total+metrics['declared_skipped_rows']
                    or not 0<metrics['train_families']<=metrics['train_rows']
                    or not 0<metrics['val_families']<=metrics['val_rows']):
                return False
    elif check['status']=='passed':
        if check['name']=='python_syntax' and not (metrics.get('python_files',0)>0 and metrics.get('invalid_files')==0):
            return False
        if check['name']=='train_lora_argparse_help' and metrics.get('exit_code')!=0:
            return False
    return True


def junit_totals(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag=='testsuite' else root.findall('testsuite') if root.tag=='testsuites' else []
    if not suites:
        raise ValueError('missing testsuite')
    totals = dict(tests=0,failures=0,errors=0,skipped=0,duration_seconds=0.0)
    cases = []
    for suite in suites:
        counts = {k:int(suite.attrib[k]) for k in ('tests','failures','errors','skipped')}
        duration = float(suite.attrib['time'])
        current = suite.findall('testcase')
        if (any(n<0 for n in counts.values()) or counts['tests']!=len(current)
                or sum(counts[k] for k in ('failures','errors','skipped'))>counts['tests']
                or not math.isfinite(duration) or duration<0):
            raise ValueError('invalid JUnit counters')
        for key,value in counts.items():
            totals[key] += value
        totals['duration_seconds'] += duration
        cases.extend(current)
    for tag,counter in [('failure','failures'),('error','errors'),('skipped','skipped')]:
        if sum(c.find(tag) is not None for c in cases)!=totals[counter]:
            raise ValueError('inconsistent JUnit outcomes')
    if not totals['tests']:
        raise ValueError('empty JUnit')
    return totals,cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('reports'))
    for name in ('unit','preflight','contracts'):
        parser.add_argument(f'--{name}-exit', type=int, required=True)
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    summary = dict(schema_version=1, **identity(Path('.')), status='failed', training_executed=False,
                   external_model_downloads=0, gpu_used=False, training_ready=False,
                   errors=[], checks=[], contracts=[], readiness_reasons=[], unit_tests={}, coverage={},
                   stage_exit_codes=dict(unit=args.unit_exit,preflight=args.preflight_exit,contracts=args.contracts_exit),
                   companion_contracts='not verified: app package, serving prompts/schema semantics, tokenizer length, trainer/GPU')
    try:
        preflight = strict_json((out / 'preflight.json').read_text(encoding='utf-8'))
        if not isinstance(preflight,dict) or preflight.get('status') not in {'passed','failed'} or not valid_checks(preflight.get('checks')) or {c['name'] for c in preflight['checks']}!=DATASETS or len(preflight['checks'])!=2:
            raise ValueError('invalid preflight')
        readiness = preflight.get('readiness_reasons', [])
        if not isinstance(readiness, list) or not all(identifier(reason) for reason in readiness) or preflight.get('training_ready', False) is not False:
            raise ValueError('invalid preflight')
        summary['checks'] = preflight['checks']
        summary['readiness_reasons'] = readiness
        if any(not consistent(c,dataset=True) for c in preflight['checks']) or (preflight['status']=='failed') != any(c['status']=='failed' for c in preflight['checks']):
            summary['errors'].append('preflight_evidence_inconsistent')
        if preflight['status'] != 'passed' or any(c.get('status') != 'passed' or c.get('errors') for c in preflight['checks']):
            summary['errors'].append('preflight_failed')
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RecursionError):
        summary['checks'] = []
        summary['readiness_reasons'] = []
        summary['errors'].append('preflight_evidence_missing_or_invalid')
    try:
        totals,_ = junit_totals(out / 'unit-junit.xml')
        summary['unit_tests'] = totals
        if totals['failures'] or totals['errors'] or totals['skipped']:
            summary['errors'].append('unit_tests_failed_or_incomplete')
    except (OSError, ValueError, KeyError, ET.ParseError):
        summary['errors'].append('junit_evidence_missing_or_invalid')
    try:
        totals,cases = junit_totals(out / 'preflight-junit.xml')
        if (totals['tests']!=2 or totals['errors'] or totals['skipped']
                or len(cases)!=2 or {c.get('name') for c in cases}!=DATASETS or not summary['checks']):
            raise ValueError('incomplete preflight JUnit')
        failures = {c.get('name') for c in cases if c.find('failure') is not None}
        expected = {c['name'] for c in summary['checks'] if c['status']=='failed'}
        if failures!=expected:
            raise ValueError('preflight JUnit contradicts checks')
        summary['preflight_junit'] = totals
    except (OSError, ValueError, KeyError, ET.ParseError):
        summary['errors'].append('preflight_junit_evidence_missing_or_invalid')
    try:
        coverage = strict_json((out / 'coverage.json').read_text(encoding='utf-8'))
        if not isinstance(coverage, dict) or not isinstance(coverage.get('totals'), dict):
            raise ValueError('invalid coverage')
        totals = coverage['totals']
        if (type(totals.get('covered_lines')) is not int or type(totals.get('num_statements')) is not int
                or type(totals.get('percent_covered')) not in (int, float)
                or not (0 <= totals['covered_lines'] <= totals['num_statements'] and totals['num_statements'] > 0
                        and 0 <= totals['percent_covered'] <= 100)):
            raise ValueError('invalid coverage')
        if not isinstance(coverage.get('files'), dict) or set(coverage['files']) != {'ci/validators.py'} or not isinstance(coverage['files']['ci/validators.py'], dict):
            raise ValueError('unexpected coverage scope')
        # Publish only after validating totals and source scope together.
        summary['coverage'] = {key:totals[key] for key in ('covered_lines','num_statements','percent_covered')}
        summary['coverage']['scope'] = 'ci/validators.py only'
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        summary['errors'].append('coverage_evidence_missing_or_invalid')
    try:
        contracts = strict_json((out / 'contracts.json').read_text(encoding='utf-8'))
        if (not isinstance(contracts,dict) or contracts.get('status') not in ('passed','failed')
                or not valid_checks(contracts.get('checks')) or len(contracts['checks'])!=2
                or {c['name'] for c in contracts['checks']}!=CONTRACTS):
            raise ValueError('invalid contracts')
        summary['contracts'] = contracts['checks']
        if (any(not consistent(c) for c in contracts['checks'])
                or (contracts['status']=='failed') != any(c['status']=='failed' for c in contracts['checks'])):
            summary['errors'].append('contracts_evidence_inconsistent')
        if contracts['status']=='failed':
            summary['errors'].append('contracts_failed')
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RecursionError):
        summary['errors'].append('contracts_evidence_missing_or_failed')
    for name, exit_code in summary['stage_exit_codes'].items():
        if exit_code:
            summary['errors'].append(f'{name}_command_failed')
    summary['status'] = 'failed' if summary['errors'] else 'passed'
    (out / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    coverage_text = (f"Coverage: {summary['coverage']['percent_covered']}% (**ci/validators.py only**, fixture execution; no training/model coverage)"
                     if summary['coverage'] else 'Coverage: unavailable (evidence missing or invalid)')
    lines = ['# Model pre-training CI', '', f"Status: **{summary['status']}** · training_ready: **false**", '',
             'training_executed=false · external_model_downloads=0 · gpu_used=false', '',
             f"Commit: `{summary['commit']}` · run: `{summary['run_id']}` · UTC: {summary['generated_at']}", '',
             f"Fixture tests: {summary['unit_tests'].get('tests', 'unavailable')} · failures: {summary['unit_tests'].get('failures', 'unavailable')} · errors: {summary['unit_tests'].get('errors', 'unavailable')}",
             coverage_text, '',
             '| Dataset | Result | Train | Validation | Joined rows | Train families | Validation families | Overlap families | Seconds |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for check in summary['checks']:
        m = check.get('metrics',{})
        lines.append('| ' + ' | '.join(str(v) for v in [check.get('name'),check.get('status'),m.get('train_rows','?'),m.get('val_rows','?'),m.get('joined_rows','?'),m.get('train_families','?'),m.get('val_families','?'),m.get('family_overlap','?'),check.get('duration_seconds','?')]) + ' |')
    for check in summary['checks']:
        for issue in check.get('errors',[]) + check.get('warnings',[]):
            lines.append(f"\n- {check.get('name')}: `{issue['code']}` (count={issue['count']})")
    for check in summary['contracts']:
        lines.append(f"\n- Contract `{check['name']}`: {check['status']} ({check['duration_seconds']} seconds), metrics={json.dumps(check['metrics'], sort_keys=True)}")
    lines += ['', 'Readiness reasons: ' + ', '.join(summary['readiness_reasons']), '',
              summary['companion_contracts'], '', 'Full input SHA256, byte counts and metrics: `preflight.json` in the artifact.', '',
              'Aggregate errors: ' + (', '.join(summary['errors']) or 'none'), '']
    (out / 'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status=summary['status'],errors=summary['errors'],training_ready=False)))
    return int(bool(summary['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
