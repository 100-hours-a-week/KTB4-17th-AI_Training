"""Run genuine standalone CLI and token-length guard, with no ML imports."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_contract_command_writes_observable_results(tmp_path):
    result = subprocess.run([sys.executable,'-m','ci.contracts','--root',str(REPO),'--output-dir',str(tmp_path)],cwd=REPO,capture_output=True,text=True)
    assert result.returncode == 0
    report = json.loads((tmp_path / 'contracts.json').read_text())
    assert report['status'] == 'passed'
    assert all(c['duration_seconds'] >= 0 for c in report['checks'])
    assert report['checks'][0]['metrics']['python_files'] > 10
    assert report['checks'][1]['metrics']['exit_code'] == 0
    assert report['training_executed'] is False


def test_syntax_corruption_and_missing_help_entrypoint_fail(tmp_path):
    (tmp_path / 'ci').mkdir()
    (tmp_path / 'ci/broken.py').write_text('def broken(')
    out = tmp_path / 'reports'
    run = subprocess.run([sys.executable,'-m','ci.contracts','--root',str(tmp_path),'--output-dir',str(out)],cwd=REPO,capture_output=True,text=True)
    assert run.returncode == 1
    result = json.loads((out / 'contracts.json').read_text())
    assert result['status'] == 'failed'
    assert [c['status'] for c in result['checks']] == ['failed','failed']


@pytest.mark.parametrize('length,limit,expected_failure', [(3,3,False),(4,3,True)])
def test_real_assert_fits_stops_before_truncation(length,limit,expected_failure):
    spec = importlib.util.spec_from_file_location('standalone_train_lora', REPO / 'Lena-Training/training/tagging/train_lora.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class LengthOnlyTokenizer:
        """Deterministic tokenizer boundary double; never downloads model assets."""
        def apply_chat_template(self, messages, tokenize=False):
            assert messages == [{'role':'user','content':'q'},{'role':'assistant','content':'a'}]
            assert tokenize is False
            return 'rendered'
        def __call__(self, text, add_special_tokens=False):
            assert text == 'rendered' and add_special_tokens is False
            return {'input_ids': list(range(length))}
    rows = [dict(prompt=[dict(role='user',content='q')],completion=[dict(role='assistant',content='a')])]
    if expected_failure:
        with pytest.raises(SystemExit): module.assert_fits(rows,LengthOnlyTokenizer(),max_length=limit)
    else:
        module.assert_fits(rows,LengthOnlyTokenizer(),max_length=limit)


@pytest.mark.parametrize('validator_text', ['def broken(', 'raise ImportError("PRIVATE unavailable dependency")'])
def test_independent_evidence_survives_broken_validator_and_unavailable_pytest(tmp_path, validator_text):
    import shutil
    shutil.copytree(REPO / 'ci',tmp_path / 'ci',ignore=shutil.ignore_patterns('__pycache__'))
    (tmp_path / 'ci/validators.py').write_text(validator_text)
    trainer = tmp_path / 'Lena-Training/training/tagging/train_lora.py'
    trainer.parent.mkdir(parents=True)
    shutil.copyfile(REPO / 'Lena-Training/training/tagging/train_lora.py',trainer)
    out = tmp_path / 'reports'
    missing_tools = subprocess.run([sys.executable,'-S','-m','pytest'],cwd=tmp_path,capture_output=True)
    assert missing_tools.returncode != 0
    for command in ('ci.contracts','ci.preflight'):
        result = subprocess.run([sys.executable,'-S','-m',command,'--output-dir',str(out)],cwd=tmp_path,capture_output=True,text=True)
        expected = 1 if command == 'ci.preflight' or validator_text.startswith('def ') else 0
        assert result.returncode == expected
    run = subprocess.run([sys.executable,'-S','-m','ci.report','--output-dir',str(out),'--unit-exit','1','--preflight-exit','1','--contracts-exit','1'],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode == 1
    report = json.loads((out / 'summary.json').read_text())
    assert report['status']=='failed'
    assert len(report['checks']) == 2
    assert len(report['contracts']) == 2
    assert any(e['code']=='validator_unavailable' for e in report['checks'][0]['errors'])
    assert report['unit_tests'] == {}
    assert (out / 'preflight-junit.xml').exists()
    markdown = (out / 'summary.md').read_text()
    assert 'unavailable' in markdown
    assert 'PRIVATE' not in json.dumps(report)+markdown
    assert 'Traceback' not in run.stderr
