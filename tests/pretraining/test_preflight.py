"""Behavior regressions: corruption and family leakage must never turn green."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


@pytest.fixture
def dataset(tmp_path):
    base = tmp_path / 'Lena-Training'
    out = base / 'training/tagging/out'
    source = base / 'data/teacher/qwen-fp8-sub1000/tagging-sft-a0.8.jsonl'
    teacher = base / 'data/teacher/qwen-fp8-sub1000/tagging.jsonl'
    rows = [dict(id=f'item-{i}', prompt=[dict(role='system', content='system'), dict(role='user', content=f'question {i}')], completion=[dict(role='assistant', content='{"primary":[],"secondary":[],"off_topic":false}')]) for i in range(2)]
    write_jsonl(out / 'train.jsonl', rows[:1])
    write_jsonl(out / 'val.jsonl', rows[1:])
    write_jsonl(source, [dict(id=r['id']) for r in rows])  # group is optional
    write_jsonl(teacher, [dict(id=r['id'], familyId=f'family-{i}') for i, r in enumerate(rows)])
    manifest = dict(input=str(source.relative_to(base)), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), prompt_sha256=hashlib.sha256(b'system').hexdigest(), counts=dict(train=1,val=1), sources=['synthetic'], unreviewed_allowed=False)
    (out / 'manifest.json').write_text(json.dumps(manifest))
    return tmp_path, out, source, teacher, rows, manifest


def check(dataset):
    from ci.validators import validate_dataset
    return validate_dataset(dataset[0] / 'Lena-Training', 'tagging')


def codes(result):
    return {e['code'] for e in result['errors']}


def test_valid_counts_hashes_and_optional_group(dataset):
    result = check(dataset)
    assert result['status'] == 'passed'
    assert result['metrics']['train_rows'] == 1
    assert result['metrics']['val_rows'] == 1
    assert result['metrics']['family_overlap'] == 0
    assert result['metrics']['joined_rows'] == 2
    assert len(result['files']) == 5
    assert all(len(f['sha256']) == 64 for f in result['files'])
    assert result['duration_seconds'] >= 0


@pytest.mark.parametrize('relative', ['manifest.json', 'train.jsonl', 'val.jsonl'])
def test_missing_saved_input_fails(dataset, relative):
    (dataset[1] / relative).unlink()
    assert 'missing_file' in codes(check(dataset))


@pytest.mark.parametrize('position', [2, 3])
def test_missing_source_cannot_skip_family_join(dataset, position):
    dataset[position].unlink()
    assert 'missing_file' in codes(check(dataset))


@pytest.mark.parametrize('text', ['{bad', '[]', 'null', '{"counts":false}', '{"input":"../../outside"}'])
def test_invalid_manifest_fails_without_traceback(dataset, text):
    (dataset[1] / 'manifest.json').write_text(text)
    assert check(dataset)['status'] == 'failed'


@pytest.mark.parametrize('text', ['{private prompt', '[]', 'null', '\n', '{"id":"x"}', '{"id":7}'])
def test_invalid_jsonl_and_empty_data_fail_safely(dataset, text):
    (dataset[1] / 'train.jsonl').write_text(text)
    result = check(dataset)
    assert result['status'] == 'failed'
    assert 'private prompt' not in json.dumps(result)


@pytest.mark.parametrize('mutation,code', [
    ('role', 'invalid_messages'), ('empty', 'invalid_messages'), ('assistant', 'invalid_messages'),
    ('json', 'invalid_completion'), ('list', 'invalid_completion'), ('nan', 'invalid_completion'), ('overflow', 'invalid_completion'),
    ('system', 'prompt_hash_mismatch'), ('tags', 'invalid_tagging_labels'),
])
def test_message_and_completion_contracts(dataset, mutation, code):
    row = dataset[4][0]
    if mutation == 'role': row['prompt'][1]['role'] = 'tool'
    if mutation == 'empty': row['prompt'][1]['content'] = ''
    if mutation == 'assistant': row['completion'][0]['role'] = 'user'
    if mutation == 'json': row['completion'][0]['content'] = 'SECRET malformed completion'
    if mutation == 'list': row['completion'][0]['content'] = '[]'
    if mutation == 'overflow': row['completion'][0]['content'] = '{"value":1e999}'
    if mutation == 'nan': row['completion'][0]['content'] = '{"value":NaN}'
    if mutation == 'system': row['prompt'][0]['content'] = 'changed'
    if mutation == 'tags': row['completion'][0]['content'] = '{"primary":[],"secondary":[],"off_topic":"false"}'
    write_jsonl(dataset[1] / 'train.jsonl', [row])
    result = check(dataset)
    assert code in codes(result)
    assert 'SECRET' not in json.dumps(result)


def test_duplicate_ids_and_content_are_counted(dataset):
    row = dataset[4][0]
    write_jsonl(dataset[1] / 'train.jsonl', [row, row])
    result = check(dataset)
    assert {'duplicate_ids', 'duplicate_content', 'count_mismatch'} <= codes(result)
    assert result['metrics']['duplicate_ids'] == 1
    assert result['metrics']['duplicate_content'] == 1


def test_source_id_join_reveals_family_leak_without_groups(dataset):
    write_jsonl(dataset[3], [dict(id=r['id'], familyId='shared') for r in dataset[4]])
    result = check(dataset)
    assert result['metrics']['family_overlap'] == 1
    assert 'family_leakage' in codes(result)


def test_optional_group_cannot_hide_teacher_family(dataset):
    for i, row in enumerate(dataset[4]): row['group'] = f'unrelated-{i}'
    write_jsonl(dataset[1] / 'train.jsonl', dataset[4][:1])
    write_jsonl(dataset[1] / 'val.jsonl', dataset[4][1:])
    write_jsonl(dataset[3], [dict(id=r['id'], familyId='shared') for r in dataset[4]])
    assert 'family_leakage' in codes(check(dataset))


@pytest.mark.parametrize('mutation,code', [('missing','unresolved_source_ids'),('family','unresolved_families'),('duplicate','duplicate_source_ids')])
def test_unreliable_join_fails_closed(dataset, mutation, code):
    rows = [dict(id=r['id'], familyId=f'f-{i}') for i, r in enumerate(dataset[4])]
    if mutation == 'missing': rows.pop()
    if mutation == 'family': rows[0].pop('familyId')
    if mutation == 'duplicate': rows.append(rows[0])
    write_jsonl(dataset[3], rows)
    assert code in codes(check(dataset))


def test_input_hash_and_manifest_counts_are_verified(dataset):
    manifest = dataset[5]
    manifest['input_sha256'] = '0' * 64
    manifest['counts']['val'] = 2
    (dataset[1] / 'manifest.json').write_text(json.dumps(manifest))
    assert {'input_hash_mismatch', 'count_mismatch'} <= codes(check(dataset))


def test_unreviewed_is_visible_and_never_ready(dataset):
    manifest = dataset[5]
    manifest['unreviewed_allowed'] = True
    (dataset[1] / 'manifest.json').write_text(json.dumps(manifest))
    result = check(dataset)
    assert result['status'] == 'passed'
    assert result['training_ready'] is False
    assert 'unreviewed_data' in result['readiness_reasons']
    assert any(w['code'] == 'unreviewed_data' for w in result['warnings'])


def test_preflight_cli_always_writes_failure_json_and_junit(tmp_path):
    out = tmp_path / 'reports'
    command = [sys.executable, '-m', 'ci.preflight', '--root', str(tmp_path), '--output-dir', str(out)]
    run = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    assert run.returncode == 1
    result = json.loads((out / 'preflight.json').read_text())
    assert result['status'] == 'failed'
    assert result['training_executed'] is False
    assert result['gpu_used'] is False
    assert result['external_model_downloads'] == 0
    assert len(result['checks']) == 2
    from xml.etree import ElementTree as ET
    assert len(ET.parse(out / 'preflight-junit.xml').findall('.//failure')) == 2


@pytest.fixture
def extraction(dataset):
    base = dataset[0] / 'Lena-Training'
    out = base / 'training/extraction/out'
    source = base / 'data/teacher/qwen-fp8-sub1000/extraction.jsonl'
    write_jsonl(source, [dict(id=r['id'], familyId=f'family-{i}') for i,r in enumerate(dataset[4])])
    rows = dataset[4]
    for row in rows:
        row['completion'][0]['content'] = json.dumps(dict(assurances=50,compliance=50,ideal_status=50,ideal_warmth=50,positivity=50,withdrawal=50,narrative=dict(headline='h',body='b',traits=['t']),summaries=[]))
    write_jsonl(out / 'train.jsonl',rows[:1])
    write_jsonl(out / 'val.jsonl',rows[1:])
    manifest = dataset[5]
    manifest['input'] = str(source.relative_to(base))
    manifest['input_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    (out / 'manifest.json').write_text(json.dumps(manifest))
    return base,out,rows


def test_extraction_saved_score_dimensions_are_accepted_with_stale_warning(extraction):
    from ci.validators import validate_dataset
    result = validate_dataset(extraction[0],'extraction')
    assert result['status'] == 'passed'
    assert result['metrics']['joined_rows'] == 2
    assert 'saved_extraction_rebuild_required' in result['readiness_reasons']


@pytest.mark.parametrize('field,value', [('assurances',True),('assurances',101),('assurances',-1),('unexpected',50),('narrative',None),('summaries',{})])
def test_extraction_rejects_bad_scores_or_structure(extraction, field, value):
    from ci.validators import validate_dataset
    output = json.loads(extraction[2][0]['completion'][0]['content'])
    output[field] = value
    extraction[2][0]['completion'][0]['content'] = json.dumps(output)
    write_jsonl(extraction[1] / 'train.jsonl',extraction[2][:1])
    assert 'invalid_extraction_labels' in codes(validate_dataset(extraction[0],'extraction'))


def test_duplicate_json_keys_are_corruption(dataset):
    (dataset[1] / 'train.jsonl').write_text('{"id":"first","id":"second"}')
    assert 'invalid_jsonl' in codes(check(dataset))


def test_invalid_encoding_is_reported(dataset):
    (dataset[1] / 'train.jsonl').write_bytes(b'\xff\xfe')
    assert 'invalid_encoding' in codes(check(dataset))


def test_input_symlink_outside_root_is_not_read(dataset, tmp_path):
    outside = tmp_path / 'outside.jsonl'
    outside.write_text('PRIVATE DATA')
    path = dataset[1] / 'train.jsonl'
    path.unlink()
    path.symlink_to(outside)
    result = check(dataset)
    assert 'unsafe_input_path' in codes(result)
    assert 'PRIVATE' not in json.dumps(result)


@pytest.mark.parametrize('role', [[], {}, None, 5, 'assistant'])
def test_invalid_role_type_or_turn_order_fails_without_exception(dataset, role):
    row = dataset[4][0]
    row['prompt'] = [dict(role='system',content='system'),dict(role=role,content='q'),dict(role='user',content='a')]
    write_jsonl(dataset[1] / 'train.jsonl',[row])
    assert 'invalid_messages' in codes(check(dataset))


@pytest.mark.parametrize('skipped', [{'reason':True},{'reason':-1},[]])
def test_bad_skip_accounting_is_invalid_manifest(extraction, skipped):
    from ci.validators import validate_dataset
    path = extraction[1] / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['skipped'] = skipped
    path.write_text(json.dumps(manifest))
    assert 'invalid_manifest' in codes(validate_dataset(extraction[0],'extraction'))


def test_source_row_conservation_detects_unaccounted_drop(extraction):
    from ci.validators import validate_dataset
    path = extraction[0] / 'data/teacher/qwen-fp8-sub1000/extraction.jsonl'
    write_jsonl(path, [dict(id='item-0',familyId='f0'),dict(id='item-1',familyId='f1'),dict(id='discarded',familyId='f2')])
    manifest_path = extraction[1] / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    assert 'source_count_mismatch' in codes(validate_dataset(extraction[0],'extraction'))


@pytest.mark.parametrize('location', ['user', 'system', 'completion', 'source_id'])
def test_unpaired_unicode_surrogate_produces_sanitized_json_and_junit(dataset, location):
    row = dataset[4][0]
    if location == 'source_id':
        write_jsonl(dataset[3], [dict(id='\ud800', familyId='family-0')])
    else:
        message = row['completion'][0] if location == 'completion' else row['prompt'][0 if location == 'system' else 1]
        message['content'] = '\ud800'
        write_jsonl(dataset[1] / 'train.jsonl', [row])
    out = dataset[0] / 'reports'
    run = subprocess.run([sys.executable, '-m', 'ci.preflight', '--root', str(dataset[0]), '--output-dir', str(out)], cwd=REPO, capture_output=True, text=True)
    assert run.returncode == 1
    result = json.loads((out / 'preflight.json').read_text())
    tagging = result['checks'][0]
    assert tagging['status'] == 'failed'
    assert 'invalid_jsonl' in codes(tagging)
    from xml.etree import ElementTree as ET
    assert ET.parse(out / 'preflight-junit.xml').findall('.//failure')
    assert 'Traceback' not in run.stderr
    assert '\\ud800' not in json.dumps(result)


def test_input_symlink_cycle_still_writes_failed_preflight_evidence(dataset):
    path = dataset[1] / 'train.jsonl'
    path.unlink()
    path.symlink_to(path.name)
    out = dataset[0] / 'reports'
    run = subprocess.run([sys.executable, '-m', 'ci.preflight', '--root', str(dataset[0]), '--output-dir', str(out)], cwd=REPO, capture_output=True, text=True)
    assert run.returncode == 1
    result = json.loads((out / 'preflight.json').read_text())
    assert result['checks'][0]['status'] == 'failed'
    assert (out / 'preflight-junit.xml').exists()
    assert 'Traceback' not in run.stderr
