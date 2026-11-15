"""Bounded streaming SFT audit; retain only ID/family/label hashes and counters."""
from __future__ import annotations

import hashlib
import json
import re
import stat
import time
from collections import Counter
from pathlib import Path

from ci.evidence import strict_json

TEACHER_DIR = 'data/teacher/qwen-fp8-sub1000'
INPUTS = {'tagging': f'{TEACHER_DIR}/tagging-sft-a0.8.jsonl', 'extraction': f'{TEACHER_DIR}/extraction.jsonl'}
EXTRACTION_SCORES = {'anxiety', 'avoidance', 'contact_rhythm', 'disclosure', 'engagement', 'ideal_vitality', 'openness', 'problem_solving', 'seriousness', 'assurances', 'compliance', 'ideal_status', 'ideal_warmth', 'positivity', 'withdrawal'}
MAX_JSONL_BYTES = 128 * 1024 * 1024
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_RECORDS = 100_000
MAX_MANIFEST_BYTES = 64 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def label_hash(labels):
    return digest(json.dumps(labels,sort_keys=True,separators=(',', ':'),ensure_ascii=False).encode('utf-8'))


def validate_dataset(base: Path, kind: str) -> dict:
    """Read fixed inputs in bounded lines; diagnostics are aggregate codes/counts."""
    start = time.perf_counter()
    result = dict(name=kind,status='failed',errors=[],warnings=[],skips=[],metrics={},files=[],training_ready=False,
                  readiness_reasons=['application_prompt_and_semantics_unavailable'])
    metrics, issues = result['metrics'], Counter()

    def error(code,count=1):
        issues[code] += count

    def open_input(relative,limit):
        path = base / relative
        try:
            if not path.resolve().is_relative_to(base.resolve()):
                error('unsafe_input_path')
                return None
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                error('nonregular_input')
                return None
            if info.st_size > limit:
                error('input_file_limit')
                return None
            return path.open('rb')
        except FileNotFoundError:
            error('missing_file')
        except OSError:
            error('unreadable_file')
        return None

    def manifest_input(relative):
        handle = open_input(relative,MAX_MANIFEST_BYTES)
        if handle is None:
            return {}
        with handle:
            data = handle.read(MAX_MANIFEST_BYTES+1)
        if len(data)>MAX_MANIFEST_BYTES:
            error('input_file_limit')
            return {}
        result['files'].append(dict(path=relative,bytes=len(data),sha256=digest(data)))
        try:
            parsed = strict_json(data.decode('utf-8'))
            if not isinstance(parsed,dict):
                raise ValueError('object required')
            return parsed
        except (ValueError,RecursionError):
            error('invalid_manifest')
            return {}

    def rows(relative):
        handle = open_input(relative,MAX_JSONL_BYTES)
        if handle is None:
            return
        file_hash, total_bytes, records, parsed_count, complete = hashlib.sha256(), 0, 0, 0, True
        with handle:
            while True:
                line = handle.readline(MAX_LINE_BYTES+1)
                if not line:
                    break
                total_bytes += len(line)
                if total_bytes>MAX_JSONL_BYTES:
                    error('input_file_limit')
                    complete = False
                    break
                if len(line)>MAX_LINE_BYTES:
                    error('input_line_limit')
                    complete = False
                    break
                file_hash.update(line)
                if not line.strip():
                    continue
                records += 1
                if records>MAX_RECORDS:
                    error('input_records_limit')
                    complete = False
                    break
                try:
                    text = line.decode('utf-8')
                except UnicodeError:
                    error('invalid_encoding')
                    continue
                try:
                    row = strict_json(text)
                    if not isinstance(row,dict):
                        raise ValueError('object required')
                except (ValueError,RecursionError):
                    error('invalid_jsonl')
                    continue
                parsed_count += 1
                yield row
        if complete:
            result['files'].append(dict(path=relative,bytes=total_bytes,sha256=file_hash.hexdigest()))
        if not parsed_count:
            error('empty_dataset')

    manifest = manifest_input(f'training/{kind}/out/manifest.json')
    counts, skipped, dropped = manifest.get('counts'), manifest.get('skipped',{}), manifest.get('dropped_by_max_primary',0)
    valid_manifest = (isinstance(counts,dict) and set(counts)=={'train','val'} and all(type(v) is int and v>0 for v in counts.values())
                      and manifest.get('input')==INPUTS[kind]
                      and all(isinstance(manifest.get(k),str) and re.fullmatch('[0-9a-f]{64}',manifest[k]) for k in ('input_sha256','prompt_sha256'))
                      and type(manifest.get('unreviewed_allowed')) is bool
                      and isinstance(manifest.get('sources'),list) and bool(manifest['sources'])
                      and all(isinstance(s,str) and bool(s.strip()) for s in manifest['sources'])
                      and isinstance(skipped,dict) and all(type(n) is int and n>=0 for n in skipped.values())
                      and type(dropped) is int and dropped>=0)
    if not valid_manifest:
        error('invalid_manifest')
    if manifest.get('unreviewed_allowed') is True:
        result['warnings'].append(dict(code='unreviewed_data',count=1))
        result['readiness_reasons'].append('unreviewed_data')
    if kind=='extraction':
        result['warnings'].append(dict(code='saved_extraction_rebuild_required',count=1))
        result['readiness_reasons'].append('saved_extraction_rebuild_required')
    result['skips'].append(dict(code='companion_application_contracts_unavailable',count=1))

    # Retain no source prompts, teacher responses or full row dictionaries.
    source_index, teacher_index = {}, {}
    source_count = teacher_count = 0
    for row in rows(INPUTS[kind]):
        source_count += 1
        key = row.get('id')
        if not isinstance(key,str) or not key.strip():
            error('invalid_source_ids')
            continue
        if key in source_index:
            error('duplicate_source_ids')
            continue
        labels = {k:row.get(k) for k in ('primary','secondary','off_topic')} if kind=='tagging' else row.get('output')
        source_index[key] = label_hash(labels) if isinstance(labels,dict) else None
        if kind=='extraction':
            family = row.get('familyId')
            teacher_index[key] = family if isinstance(family,str) and family.strip() else None
    source_file = next((f for f in result['files'] if f['path']==INPUTS[kind]),None)
    if valid_manifest and source_file and source_file['sha256']!=manifest['input_sha256']:
        error('input_hash_mismatch')
    if kind=='tagging':
        for row in rows(f'{TEACHER_DIR}/tagging.jsonl'):
            teacher_count += 1
            key = row.get('id')
            if not isinstance(key,str) or not key.strip():
                error('invalid_source_ids')
            elif key in teacher_index:
                error('duplicate_source_ids')
            else:
                family = row.get('familyId')
                teacher_index[key] = family if isinstance(family,str) and family.strip() else None
    else:
        teacher_count = source_count
    metrics.update(source_rows=source_count,teacher_rows=teacher_count)
    seen_ids, seen_content = Counter(), Counter()
    families = {'train':set(),'val':set()}
    prompt_sets = {'train':set(),'val':set()}
    unresolved_source = unresolved_family = joined = compared = mismatches = 0
    for split in ('train','val'):
        split_count = 0
        for row in rows(f'training/{kind}/out/{split}.jsonl'):
            split_count += 1
            key = row.get('id')
            valid_id = isinstance(key,str) and bool(key.strip())
            if not valid_id:
                error('invalid_sft_id')
            else:
                seen_ids[key] += 1
                if key not in source_index or key not in teacher_index:
                    unresolved_source += 1
                elif teacher_index[key] is None:
                    unresolved_family += 1
                else:
                    families[split].add(teacher_index[key])
                    joined += 1
            prompt, completion = row.get('prompt'), row.get('completion')
            if not _messages_valid(prompt,completion):
                error('invalid_messages')
                continue
            if digest(prompt[0]['content'].encode())!=manifest.get('prompt_sha256'):
                error('prompt_hash_mismatch')
            seen_content[label_hash([prompt,completion])] += 1
            user_text = '\n'.join(m['content'] for m in prompt if m['role']=='user')
            prompt_sets[split].add(digest(re.sub(r'[\W_]+','',user_text).lower().encode()))
            try:
                output = strict_json(completion[0]['content'])
                if not isinstance(output,dict) or not output:
                    raise ValueError('nonempty object required')
            except (ValueError,RecursionError):
                error('invalid_completion')
                continue
            if kind=='tagging' and not _tagging_valid(output):
                error('invalid_tagging_labels')
            if kind=='extraction' and not _extraction_valid(output):
                error('invalid_extraction_labels')
            if valid_id and key in source_index:
                compared += 1
                if label_hash(output)!=source_index[key]:
                    mismatches += 1
        metrics[f'{split}_rows'] = split_count
        metrics[f'{split}_families'] = len(families[split])
        if valid_manifest and counts[split]!=split_count:
            error('count_mismatch')
    metrics.update(joined_rows=joined,unresolved_source_ids=unresolved_source,unresolved_families=unresolved_family,
                   duplicate_ids=sum(n-1 for n in seen_ids.values()),duplicate_content=sum(n-1 for n in seen_content.values()),
                   family_overlap=len(families['train'] & families['val']),prompt_overlap=len(prompt_sets['train'] & prompt_sets['val']),
                   source_labels_compared=compared,source_label_mismatches=mismatches)
    for metric,code in [('unresolved_source_ids','unresolved_source_ids'),('unresolved_families','unresolved_families'),
                        ('duplicate_ids','duplicate_ids'),('duplicate_content','duplicate_content'),('family_overlap','family_leakage'),
                        ('prompt_overlap','prompt_leakage'),('source_label_mismatches','source_label_mismatch')]:
        if metrics[metric]:
            error(code,metrics[metric])
    if valid_manifest:
        metrics['declared_skipped_rows'] = sum(skipped.values())+dropped
        if source_count!=metrics['train_rows']+metrics['val_rows']+metrics['declared_skipped_rows']:
            error('source_count_mismatch')
    result['errors'] = [dict(code=code,count=count) for code,count in sorted(issues.items()) if count]
    if result['errors']:
        result['readiness_reasons'].append('artifact_validation_failed')
    result['status'] = 'failed' if result['errors'] else 'passed'
    result['duration_seconds'] = round(time.perf_counter()-start,6)
    return result


def _messages_valid(prompt, completion):
    if not isinstance(prompt, list) or len(prompt) < 2 or not isinstance(completion, list) or len(completion) != 1:
        return False
    messages = prompt + completion
    if not all(isinstance(m, dict) and isinstance(m.get('content'), str) and m['content'].strip() for m in messages):
        return False
    return (prompt[0].get('role') == 'system' and all(m.get('role') == ('user' if i % 2 else 'assistant') for i, m in enumerate(prompt[1:], start=1))
            and prompt[-1].get('role') == 'user' and completion[0].get('role') == 'assistant')


def _tagging_valid(output):
    if set(output) != {'primary', 'secondary', 'off_topic'} or type(output.get('off_topic')) is not bool:
        return False
    for key in ('primary', 'secondary'):
        labels = output[key]
        if not isinstance(labels, list) or not all(isinstance(v, str) and bool(v.strip()) for v in labels) or len(set(labels)) != len(labels):
            return False
    return not set(output['primary']) & set(output['secondary']) and (not output['off_topic'] or not (output['primary'] or output['secondary']))


def _extraction_valid(output):
    if set(output) - (EXTRACTION_SCORES | {'narrative', 'summaries'}):
        return False
    if not isinstance(output.get('narrative'), dict) or not output['narrative'] or not isinstance(output.get('summaries'), list):
        return False
    return all(value is None or (type(value) is int and 0 <= value <= 100) for key, value in output.items() if key in EXTRACTION_SCORES)
