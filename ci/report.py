"""Combine preflight, fixture JUnit and validator-only coverage; fail on missing evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from ci.preflight import identity
from ci.validators import strict_json


def valid_checks(checks):
    return isinstance(checks, list) and bool(checks) and all(
        isinstance(c, dict) and c.get('status') in ('passed', 'failed')
        and isinstance(c.get('name'), str) and isinstance(c.get('metrics', {}), dict)
        and isinstance(c.get('duration_seconds', 0), (int, float))
        and all(isinstance(c.get(k, []), list) and all(isinstance(e, dict) and isinstance(e.get('code'), str) and type(e.get('count', 1)) is int for e in c.get(k, [])) for k in ('errors', 'warnings'))
        for c in checks)


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
        if not isinstance(preflight,dict) or preflight.get('status') not in {'passed','failed'} or not valid_checks(preflight.get('checks')):
            raise ValueError('invalid preflight')
        summary['checks'] = preflight['checks']
        summary['readiness_reasons'] = preflight.get('readiness_reasons', [])
        if preflight['status'] != 'passed' or any(c.get('status') != 'passed' or c.get('errors') for c in preflight['checks']):
            summary['errors'].append('preflight_failed')
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RecursionError):
        summary['checks'] = []
        summary['errors'].append('preflight_evidence_missing_or_invalid')
    try:
        tree = ET.parse(out / 'unit-junit.xml').getroot()
        suites = [tree] if tree.tag == 'testsuite' else tree.findall('testsuite')
        if not suites:
            raise ValueError('missing testsuite')
        totals = {key:sum(int(s.get(key,'0')) for s in suites) for key in ('tests','failures','errors','skipped')}
        totals['duration_seconds'] = sum(float(s.get('time','0')) for s in suites)
        summary['unit_tests'] = totals
        if totals['tests'] <= 0 or totals['failures'] or totals['errors'] or totals['skipped']:
            summary['errors'].append('unit_tests_failed_or_incomplete')
    except (OSError, ValueError, ET.ParseError):
        summary['errors'].append('junit_evidence_missing_or_invalid')
    try:
        coverage = strict_json((out / 'coverage.json').read_text(encoding='utf-8'))
        totals = coverage['totals']
        summary['coverage'] = {key:totals[key] for key in ('covered_lines','num_statements','percent_covered')}
        summary['coverage']['scope'] = 'ci/validators.py only'
        if not (0 <= totals['covered_lines'] <= totals['num_statements'] and totals['num_statements'] > 0 and 0 <= totals['percent_covered'] <= 100):
            raise ValueError('invalid coverage')
        if set(coverage.get('files',{})) != {'ci/validators.py'}:
            raise ValueError('unexpected coverage scope')
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        summary['errors'].append('coverage_evidence_missing_or_invalid')
    try:
        contracts = strict_json((out / 'contracts.json').read_text(encoding='utf-8'))
        if contracts.get('status') != 'passed' or not valid_checks(contracts.get('checks')) or any(c.get('status') != 'passed' or c.get('errors') for c in contracts['checks']):
            raise ValueError('contract checks failed')
        summary['contracts'] = contracts['checks']
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RecursionError):
        summary['errors'].append('contracts_evidence_missing_or_failed')
    for name, exit_code in summary['stage_exit_codes'].items():
        if exit_code:
            summary['errors'].append(f'{name}_command_failed')
    summary['status'] = 'failed' if summary['errors'] else 'passed'
    (out / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    lines = ['# Model pre-training CI', '', f"Status: **{summary['status']}** · training_ready: **false**", '',
             'training_executed=false · external_model_downloads=0 · gpu_used=false', '',
             f"Commit: `{summary['commit']}` · run: `{summary['run_id']}` · UTC: {summary['generated_at']}", '',
             f"Fixture tests: {summary['unit_tests'].get('tests', 'unavailable')} · failures: {summary['unit_tests'].get('failures', 'unavailable')} · errors: {summary['unit_tests'].get('errors', 'unavailable')}",
             f"Coverage: {summary['coverage'].get('percent_covered','unavailable')}% (**ci/validators.py only**, fixture execution; no training/model coverage)", '',
             '| Dataset | Result | Train | Validation | Joined rows | Train families | Validation families | Overlap families | Seconds |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for check in summary['checks']:
        m = check.get('metrics',{})
        lines.append('| ' + ' | '.join(str(v) for v in [check.get('name'),check.get('status'),m.get('train_rows','?'),m.get('val_rows','?'),m.get('joined_rows','?'),m.get('train_families','?'),m.get('val_families','?'),m.get('family_overlap','?'),check.get('duration_seconds','?')]) + ' |')
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
