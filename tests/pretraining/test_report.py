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
    assert 'ci/validators.py only' in markdown
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
