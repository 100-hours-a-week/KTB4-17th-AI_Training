"""Aggregate success requires complete, consistent measured evidence."""
import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

REPO = Path(__file__).resolve().parents[2]


def dataset_check(name):
    return dict(name=name,status='passed',duration_seconds=0.1,errors=[],warnings=[],skips=[],files=[],metrics=dict(
        source_rows=2,teacher_rows=2,train_rows=1,val_rows=1,train_families=1,val_families=1,joined_rows=2,
        unresolved_source_ids=0,unresolved_families=0,duplicate_ids=0,duplicate_content=0,family_overlap=0,
        prompt_overlap=0,declared_skipped_rows=0,source_labels_compared=2,source_label_mismatches=0))


def write_junit(path, checks=None, failures=0):
    cases = checks or [dict(name=f'fixture_{n}',status='failed' if n < failures else 'passed') for n in range(7)]
    failed = sum(c['status']=='failed' for c in cases)
    suite = ET.Element('testsuite',name='saved_sft_preflight' if checks else 'pytest',tests=str(len(cases)),failures=str(failed),errors='0',skipped='0',time='0.2')
    for check in cases:
        case = ET.SubElement(suite,'testcase',name=check['name'],time='0.1')
        if check['status']=='failed': ET.SubElement(case,'failure',message='sanitized test failure')
    ET.ElementTree(suite).write(path,encoding='utf-8',xml_declaration=True)


@pytest.fixture
def complete_evidence(tmp_path):
    checks = [dataset_check('tagging'),dataset_check('extraction')]
    (tmp_path / 'preflight.json').write_text(json.dumps(dict(status='passed',checks=checks,readiness_reasons=['application_prompt_and_semantics_unavailable'],training_ready=False)))
    write_junit(tmp_path / 'preflight-junit.xml',checks)
    write_junit(tmp_path / 'unit-junit.xml')
    contracts = [dict(name='python_syntax',status='passed',duration_seconds=0.1,metrics=dict(python_files=30,invalid_files=0),errors=[],warnings=[]),dict(name='train_lora_argparse_help',status='passed',duration_seconds=0.2,metrics=dict(exit_code=0),errors=[],warnings=[])]
    (tmp_path / 'contracts.json').write_text(json.dumps(dict(status='passed',checks=contracts)))
    (tmp_path / 'coverage.json').write_text(json.dumps(dict(totals=dict(covered_lines=90,num_statements=100,percent_covered=90),files={'ci/validators.py':{}})))
    return tmp_path


def render_evidence(directory, unit=0, preflight=0, contracts=0):
    return subprocess.run([sys.executable,'-m','ci.report','--output-dir',str(directory),'--unit-exit',str(unit),'--preflight-exit',str(preflight),'--contracts-exit',str(contracts)],cwd=REPO,capture_output=True,text=True)


def mutate_json(directory,name,update):
    path = directory / name
    result = json.loads(path.read_text())
    update(result)
    path.write_text(json.dumps(result))


@pytest.mark.parametrize('scenario',['passed','leakage','missing','malformed','unit-failed','contracts-failed'])
def test_report_aggregates_evidence_and_keeps_failures(complete_evidence,scenario):
    if scenario=='missing':
        for path in complete_evidence.iterdir(): path.unlink()
    if scenario=='malformed': (complete_evidence / 'preflight.json').write_text('{broken')
    if scenario=='unit-failed': write_junit(complete_evidence / 'unit-junit.xml',failures=1)
    if scenario=='leakage':
        data = json.loads((complete_evidence / 'preflight.json').read_text())
        data['status']='failed'
        check=data['checks'][0]
        check['status']='failed'
        check['metrics'].update(train_rows=6915,val_rows=828,joined_rows=7743,source_rows=7743,teacher_rows=10279,train_families=997,val_families=575,family_overlap=575,source_labels_compared=7743)
        check['errors']=[dict(code='family_leakage',count=575)]
        (complete_evidence / 'preflight.json').write_text(json.dumps(data))
        write_junit(complete_evidence / 'preflight-junit.xml',data['checks'])
    run = render_evidence(complete_evidence,unit=int(scenario=='unit-failed'),preflight=int(scenario=='leakage'),contracts=int(scenario=='contracts-failed'))
    assert run.returncode == (0 if scenario=='passed' else 1)
    summary=json.loads((complete_evidence / 'summary.json').read_text())
    markdown=(complete_evidence / 'summary.md').read_text()
    assert summary['status']==('passed' if scenario=='passed' else 'failed')
    assert summary['training_executed'] is False and summary['training_ready'] is False
    if scenario!='missing':
        assert summary['unit_tests']['tests']==7
        assert summary['coverage']['percent_covered']==90
        assert 'ci/validators.py only' in markdown
    if scenario=='leakage': assert '575' in markdown
    if scenario in ('missing','malformed'): assert summary['errors']


@pytest.mark.parametrize('corruption',['nested-preflight','junit-number','coverage-scope','coverage-missing-files','contradictory-preflight'])
def test_corrupt_evidence_still_produces_failed_summary(complete_evidence,corruption):
    if corruption=='nested-preflight': mutate_json(complete_evidence,'preflight.json',lambda d:d.update(checks=[None]))
    if corruption=='junit-number': (complete_evidence / 'unit-junit.xml').write_text('<testsuite tests="bad"/>')
    if corruption.startswith('coverage-'): mutate_json(complete_evidence,'coverage.json',lambda d:d.update(files={} if corruption=='coverage-missing-files' else {'application.py':{}}))
    if corruption=='contradictory-preflight': mutate_json(complete_evidence,'preflight.json',lambda d:d['checks'][0]['errors'].append(dict(code='family_leakage',count=1)))
    assert render_evidence(complete_evidence).returncode==1
    assert json.loads((complete_evidence / 'summary.json').read_text())['status']=='failed'
    assert (complete_evidence / 'summary.md').exists()


@pytest.mark.parametrize('corruption',['issue-count','contract-duration','contract-metrics','readiness-string','readiness-element','readiness-object'])
def test_missing_fields_and_invalid_readiness_always_write_sanitized_summary(complete_evidence,corruption):
    if corruption=='issue-count': mutate_json(complete_evidence,'preflight.json',lambda d:d['checks'][0]['errors'].append(dict(code='family_leakage')))
    elif corruption.startswith('readiness-'):
        bad={'readiness-string':'PRIVATE raw text','readiness-element':[None],'readiness-object':{'private':'PRIVATE raw text'}}[corruption]
        mutate_json(complete_evidence,'preflight.json',lambda d:d.update(readiness_reasons=bad))
    else: mutate_json(complete_evidence,'contracts.json',lambda d:d['checks'][0].pop('duration_seconds' if corruption=='contract-duration' else 'metrics'))
    run=render_evidence(complete_evidence)
    assert run.returncode==1
    summary=json.loads((complete_evidence / 'summary.json').read_text())
    markdown=(complete_evidence / 'summary.md').read_text()
    expected='contracts_evidence_missing_or_failed' if corruption.startswith('contract-') else 'preflight_evidence_missing_or_invalid'
    assert expected in summary['errors']
    assert 'PRIVATE' not in json.dumps(summary)+markdown
    assert 'Traceback' not in run.stderr


def test_rejected_coverage_does_not_publish_unvalidated_percentage_or_scope(complete_evidence):
    mutate_json(complete_evidence,'coverage.json',lambda d:d.update(files={'application.py':{}}))
    assert render_evidence(complete_evidence).returncode==1
    summary=json.loads((complete_evidence / 'summary.json').read_text())
    assert summary['coverage']=={}
    markdown=(complete_evidence / 'summary.md').read_text()
    assert 'Coverage: unavailable' in markdown
    assert '90%' not in markdown and 'ci/validators.py only' not in markdown


@pytest.mark.parametrize('omission',['dataset','contract','preflight-junit'])
def test_missing_required_real_checks_cannot_certify_success(complete_evidence,omission):
    if omission=='dataset': mutate_json(complete_evidence,'preflight.json',lambda d:d['checks'].pop())
    if omission=='contract': mutate_json(complete_evidence,'contracts.json',lambda d:d['checks'].pop())
    if omission=='preflight-junit': (complete_evidence / 'preflight-junit.xml').unlink()
    run=render_evidence(complete_evidence)
    assert run.returncode==1
    assert json.loads((complete_evidence / 'summary.json').read_text())['status']=='failed'


def test_nonzero_leakage_metric_cannot_be_passed_without_error(complete_evidence):
    mutate_json(complete_evidence,'preflight.json',lambda d:d['checks'][0]['metrics'].update(family_overlap=1))
    assert render_evidence(complete_evidence).returncode==1
    assert json.loads((complete_evidence / 'summary.json').read_text())['status']=='failed'


def test_preflight_junit_must_match_original_check_statuses(complete_evidence):
    write_junit(complete_evidence / 'preflight-junit.xml',[dict(name='tagging',status='failed'),dict(name='extraction',status='passed')])
    assert render_evidence(complete_evidence).returncode==1


def test_valid_failed_contracts_retain_names_metrics_and_timing(complete_evidence):
    def fail(data):
        data['status']='failed'
        data['checks'][0].update(status='failed',metrics=dict(python_files=30,invalid_files=1),errors=[dict(code='syntax_invalid_or_missing',count=1)])
    mutate_json(complete_evidence,'contracts.json',fail)
    run=render_evidence(complete_evidence,contracts=1)
    assert run.returncode==1
    summary=json.loads((complete_evidence / 'summary.json').read_text())
    assert len(summary['contracts'])==2
    assert summary['contracts'][0]['duration_seconds']==0.1
    assert summary['contracts'][0]['metrics']['invalid_files']==1
    assert 'contracts_failed' in summary['errors']
    assert 'python_syntax' in (complete_evidence / 'summary.md').read_text()


def test_markdown_keeps_both_dataset_rows_before_warning_bullets(complete_evidence):
    mutate_json(complete_evidence,'preflight.json',lambda d:d['checks'][0]['warnings'].append(dict(code='unreviewed_data',count=1)))
    assert render_evidence(complete_evidence).returncode==0
    markdown=(complete_evidence / 'summary.md').read_text()
    assert markdown.index('| tagging |') < markdown.index('| extraction |') < markdown.index('- tagging:')
