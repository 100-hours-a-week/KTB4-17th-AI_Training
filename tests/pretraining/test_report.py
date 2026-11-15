"""Report success must require complete evidence, including failing preflight."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('scenario', ['passed', 'leakage', 'missing', 'malformed', 'unit-failed', 'contracts-failed'])
def test_report_aggregates_evidence_and_keeps_failures(tmp_path, scenario):
    if scenario != 'missing':
        result = dict(schema_version=1, status='failed' if scenario == 'leakage' else 'passed', training_ready=False, readiness_reasons=['unreviewed_data'], training_executed=False, external_model_downloads=0, gpu_used=False, checks=[dict(name='tagging', status='failed' if scenario == 'leakage' else 'passed', metrics={'family_overlap':575}, duration_seconds=0.1, errors=[dict(code='family_leakage',count=575)] if scenario == 'leakage' else [],warnings=[],skips=[],files=[])])
        (tmp_path / 'preflight.json').write_text('{broken' if scenario == 'malformed' else json.dumps(result))
        (tmp_path / 'unit-junit.xml').write_text('<testsuites><testsuite tests="7" failures="1" errors="0" skipped="0" time="0.4"/></testsuites>' if scenario == 'unit-failed' else '<testsuites><testsuite tests="7" failures="0" errors="0" skipped="0" time="0.4"/></testsuites>')
        (tmp_path / 'contracts.json').write_text(json.dumps({'status':'passed','checks':[{'name':'syntax','status':'passed','duration_seconds':0.1,'metrics':{}}]}))
        (tmp_path / 'coverage.json').write_text(json.dumps({'totals': {'covered_lines':90,'num_statements':100,'percent_covered':90}, 'files':{'ci/validators.py':{}}}))
    run = subprocess.run([sys.executable,'-m','ci.report','--output-dir',str(tmp_path),'--unit-exit', '1' if scenario=='unit-failed' else '0','--preflight-exit','1' if scenario=='leakage' else '0','--contracts-exit','1' if scenario=='contracts-failed' else '0'],cwd=REPO,text=True,capture_output=True)
    assert run.returncode == (0 if scenario == 'passed' else 1)
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['status'] == ('passed' if scenario == 'passed' else 'failed')
    assert summary['training_executed'] is False
    assert summary['training_ready'] is False
    markdown = (tmp_path / 'summary.md').read_text()
    if scenario != 'missing': assert 'ci/validators.py only' in markdown
    if scenario not in ('missing', 'malformed'):
        assert summary['unit_tests']['tests'] == 7
        assert '575' in markdown
        assert summary['coverage']['percent_covered'] == 90
    if scenario in ('missing', 'malformed'): assert summary['errors']


@pytest.mark.parametrize('corruption', ['nested-preflight','junit-number','coverage-scope','coverage-missing-files','contradictory-preflight'])
def test_corrupt_evidence_still_produces_failed_summary(tmp_path,corruption):
    (tmp_path / 'preflight.json').write_text(json.dumps({'status':'passed','checks':[None]} if corruption=='nested-preflight' else {'status':'passed','checks':[{'status':'passed','name':'tagging','duration_seconds':0,'metrics':{},'errors':[],'warnings':[]}]}))
    if corruption == 'contradictory-preflight':
        data = json.loads((tmp_path / 'preflight.json').read_text())
        data['checks'][0]['errors'] = [{'code':'family_leakage','count':1}]
        (tmp_path / 'preflight.json').write_text(json.dumps(data))
    (tmp_path / 'contracts.json').write_text(json.dumps({'status':'passed','checks':[{'status':'passed','name':'syntax','duration_seconds':0,'metrics':{}}]}))
    (tmp_path / 'unit-junit.xml').write_text('<testsuites><testsuite tests="bad"/></testsuites>' if corruption=='junit-number' else '<testsuites><testsuite tests="1"/></testsuites>')
    files = {} if corruption=='coverage-missing-files' else {'application.py':{}} if corruption=='coverage-scope' else {'ci/validators.py':{}}
    (tmp_path / 'coverage.json').write_text(json.dumps({'totals':dict(covered_lines=1,num_statements=2,percent_covered=50),'files':files}))
    run = subprocess.run([sys.executable,'-m','ci.report','--output-dir',str(tmp_path),'--unit-exit','0','--preflight-exit','0','--contracts-exit','0'],cwd=REPO,capture_output=True,text=True)
    assert run.returncode == 1
    assert json.loads((tmp_path / 'summary.json').read_text())['status'] == 'failed'


@pytest.fixture
def complete_evidence(tmp_path):
    check = dict(name='tagging', status='failed', duration_seconds=0.1, metrics={}, errors=[dict(code='family_leakage', count=1)], warnings=[])
    (tmp_path / 'preflight.json').write_text(json.dumps(dict(status='failed', checks=[check], readiness_reasons=['unreviewed_data'])))
    (tmp_path / 'contracts.json').write_text(json.dumps(dict(status='passed', checks=[dict(name='syntax', status='passed', duration_seconds=0.1, metrics={})])))
    (tmp_path / 'unit-junit.xml').write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0" time="0.1"/></testsuites>')
    (tmp_path / 'coverage.json').write_text(json.dumps(dict(totals=dict(covered_lines=1, num_statements=2, percent_covered=50), files={'ci/validators.py':{}})))
    return tmp_path


def render_evidence(directory):
    return subprocess.run([sys.executable, '-m', 'ci.report', '--output-dir', str(directory), '--unit-exit', '0', '--preflight-exit', '1', '--contracts-exit', '0'], cwd=REPO, capture_output=True, text=True)


@pytest.mark.parametrize('corruption', ['issue-count', 'contract-duration', 'contract-metrics', 'readiness-string', 'readiness-element', 'readiness-object'])
def test_missing_fields_and_invalid_readiness_always_write_sanitized_summary(complete_evidence, corruption):
    preflight_path = complete_evidence / 'preflight.json'
    preflight = json.loads(preflight_path.read_text())
    if corruption == 'issue-count':
        preflight['checks'][0]['errors'][0].pop('count')
    elif corruption.startswith('readiness-'):
        preflight['readiness_reasons'] = {'readiness-string':'PRIVATE raw text', 'readiness-element':[None], 'readiness-object':{'private':'PRIVATE raw text'}}[corruption]
    else:
        path = complete_evidence / 'contracts.json'
        contracts = json.loads(path.read_text())
        contracts['checks'][0].pop('duration_seconds' if corruption == 'contract-duration' else 'metrics')
        path.write_text(json.dumps(contracts))
    preflight_path.write_text(json.dumps(preflight))
    run = render_evidence(complete_evidence)
    assert run.returncode == 1
    summary = json.loads((complete_evidence / 'summary.json').read_text())
    markdown = (complete_evidence / 'summary.md').read_text()
    expected = 'contracts_evidence_missing_or_failed' if corruption.startswith('contract-') else 'preflight_evidence_missing_or_invalid'
    assert expected in summary['errors']
    assert summary['status'] == 'failed'
    assert 'PRIVATE' not in json.dumps(summary) + markdown
    assert 'Traceback' not in run.stderr


def test_rejected_coverage_does_not_publish_unvalidated_percentage_or_scope(complete_evidence):
    path = complete_evidence / 'coverage.json'
    evidence = json.loads(path.read_text())
    evidence['files'] = {'application.py':{}}
    path.write_text(json.dumps(evidence))
    run = render_evidence(complete_evidence)
    assert run.returncode == 1
    summary = json.loads((complete_evidence / 'summary.json').read_text())
    assert summary['coverage'] == {}
    markdown = (complete_evidence / 'summary.md').read_text()
    assert 'Coverage: unavailable' in markdown
    assert '50%' not in markdown
    assert 'ci/validators.py only' not in markdown
