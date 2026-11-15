"""Standard-library checks for saved SFT artifacts, with sanitized diagnostics.

This validates saved structures and provenance, not application semantics or model quality.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

TEACHER_DIR = 'data/teacher/qwen-fp8-sub1000'
INPUTS = {'tagging': f'{TEACHER_DIR}/tagging-sft-a0.8.jsonl', 'extraction': f'{TEACHER_DIR}/extraction.jsonl'}
EXTRACTION_SCORES = {'anxiety', 'avoidance', 'contact_rhythm', 'disclosure', 'engagement', 'ideal_vitality', 'openness', 'problem_solving', 'seriousness', 'assurances', 'compliance', 'ideal_status', 'ideal_warmth', 'positivity', 'withdrawal'}


def strict_json(text):
    def reject_constant(_):
        raise ValueError('nonfinite JSON')
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError('nonfinite JSON')
        return parsed
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(text, parse_constant=reject_constant, parse_float=finite_float, object_pairs_hook=reject_duplicates)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_dataset(base: Path, kind: str) -> dict:
    """Audit one fixed repository dataset; diagnostics contain codes/counts, never data."""
    start = time.perf_counter()
    result = dict(name=kind, status='failed', errors=[], warnings=[], skips=[], metrics={}, files=[], training_ready=False,
                  readiness_reasons=['application_prompt_and_semantics_unavailable'])
    metrics = result['metrics']

    def error(code, count=1):
        result['errors'].append(dict(code=code, count=count))

    def read(relative):
        path = base / relative
        # Never follow a repository input symlink outside the audit root.
        if not path.resolve().is_relative_to(base.resolve()):
            error('unsafe_input_path')
            return None
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            error('missing_file')
            return None
        except OSError:
            error('unreadable_file')
            return None
        result['files'].append(dict(path=str(relative), bytes=len(data), sha256=digest(data)))
        try:
            return data.decode('utf-8')
        except UnicodeError:
            error('invalid_encoding')
            return None

    def rows(relative):
        text = read(relative)
        if text is None:
            return []
        parsed = []
        bad = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = strict_json(line)
                if not isinstance(row, dict):
                    raise ValueError('object required')
            except (ValueError, RecursionError):
                bad += 1
                continue
            parsed.append(row)
        if bad:
            error('invalid_jsonl', bad)
        if not parsed:
            error('empty_dataset')
        return parsed

    manifest_text = read(f'training/{kind}/out/manifest.json')
    manifest = {}
    if manifest_text is not None:
        try:
            manifest = strict_json(manifest_text)
            if not isinstance(manifest, dict):
                raise ValueError('object required')
        except (ValueError, RecursionError):
            error('invalid_manifest')
            manifest = {}
    counts = manifest.get('counts')
    skipped = manifest.get('skipped', {})
    dropped = manifest.get('dropped_by_max_primary', 0)
    valid_manifest = (isinstance(counts, dict) and set(counts) == {'train', 'val'}
                      and all(type(v) is int and v > 0 for v in counts.values())
                      and manifest.get('input') == INPUTS[kind]
                      and all(isinstance(manifest.get(k), str) and re.fullmatch('[0-9a-f]{64}', manifest[k]) for k in ('input_sha256', 'prompt_sha256'))
                      and type(manifest.get('unreviewed_allowed')) is bool
                      and isinstance(manifest.get('sources'), list) and bool(manifest['sources'])
                      and all(isinstance(s, str) and bool(s.strip()) for s in manifest['sources'])
                      and isinstance(skipped, dict) and all(type(n) is int and n >= 0 for n in skipped.values())
                      and type(dropped) is int and dropped >= 0)
    if not valid_manifest:
        error('invalid_manifest')
    if manifest.get('unreviewed_allowed') is True:
        result['warnings'].append(dict(code='unreviewed_data', count=1))
        result['readiness_reasons'].append('unreviewed_data')
    # The repository's base commit explicitly marks these saved extraction files stale.
    if kind == 'extraction':
        result['warnings'].append(dict(code='saved_extraction_rebuild_required', count=1))
        result['readiness_reasons'].append('saved_extraction_rebuild_required')
    result['skips'].append(dict(code='companion_application_contracts_unavailable', count=1))

    source_rows = rows(INPUTS[kind])
    source_file = next((f for f in result['files'] if f['path'] == INPUTS[kind]), None)
    if valid_manifest and source_file and source_file['sha256'] != manifest['input_sha256']:
        error('input_hash_mismatch')
    family_relative = f'{TEACHER_DIR}/{kind}.jsonl'
    teacher_rows = source_rows if family_relative == INPUTS[kind] else rows(family_relative)

    def index(source):
        mapping = {}
        duplicates = invalid = 0
        for row in source:
            key = row.get('id')
            if not isinstance(key, str) or not key.strip():
                invalid += 1
            elif key in mapping:
                duplicates += 1
            else:
                mapping[key] = row
        if duplicates:
            error('duplicate_source_ids', duplicates)
        if invalid:
            error('invalid_source_ids', invalid)
        return mapping

    source_index = index(source_rows)
    teacher_index = source_index if teacher_rows is source_rows else index(teacher_rows)
    metrics['source_rows'] = len(source_rows)
    metrics['teacher_rows'] = len(teacher_rows)
    seen_ids, seen_content = Counter(), Counter()
    families = {'train': set(), 'val': set()}
    prompt_sets = {'train': set(), 'val': set()}
    unresolved_source = unresolved_family = joined = 0
    for split in ('train', 'val'):
        split_rows = rows(f'training/{kind}/out/{split}.jsonl')
        metrics[f'{split}_rows'] = len(split_rows)
        if valid_manifest and counts[split] != len(split_rows):
            error('count_mismatch')
        for row in split_rows:
            key = row.get('id')
            if not isinstance(key, str) or not key.strip():
                error('invalid_sft_id')
            else:
                seen_ids[key] += 1
                if key not in source_index or key not in teacher_index:
                    unresolved_source += 1
                else:
                    family = teacher_index[key].get('familyId')
                    if not isinstance(family, str) or not family.strip():
                        unresolved_family += 1
                    else:
                        families[split].add(family)
                        joined += 1
            prompt, completion = row.get('prompt'), row.get('completion')
            if not _messages_valid(prompt, completion):
                error('invalid_messages')
                continue
            if digest(prompt[0]['content'].encode()) != manifest.get('prompt_sha256'):
                error('prompt_hash_mismatch')
            content_key = json.dumps([prompt, completion], sort_keys=True, ensure_ascii=False)
            seen_content[digest(content_key.encode())] += 1
            # Ignore whitespace/punctuation when checking repeated user inputs across splits.
            user_text = '\n'.join(m['content'] for m in prompt if m['role'] == 'user')
            prompt_sets[split].add(digest(re.sub(r'[\W_]+', '', user_text).lower().encode()))
            try:
                output = strict_json(completion[0]['content'])
                if not isinstance(output, dict) or not output:
                    raise ValueError('nonempty object required')
            except (ValueError, RecursionError):
                error('invalid_completion')
                continue
            if kind == 'tagging' and not _tagging_valid(output):
                error('invalid_tagging_labels')
            if kind == 'extraction' and not _extraction_valid(output):
                error('invalid_extraction_labels')
        metrics[f'{split}_families'] = len(families[split])
    metrics.update(joined_rows=joined, unresolved_source_ids=unresolved_source, unresolved_families=unresolved_family,
                   duplicate_ids=sum(n-1 for n in seen_ids.values()), duplicate_content=sum(n-1 for n in seen_content.values()),
                   family_overlap=len(families['train'] & families['val']), prompt_overlap=len(prompt_sets['train'] & prompt_sets['val']))
    for metric, code in [('unresolved_source_ids','unresolved_source_ids'), ('unresolved_families','unresolved_families'),
                         ('duplicate_ids','duplicate_ids'), ('duplicate_content','duplicate_content'),
                         ('family_overlap','family_leakage'), ('prompt_overlap','prompt_leakage')]:
        if metrics[metric]:
            error(code, metrics[metric])
    if valid_manifest:
        metrics['declared_skipped_rows'] = sum(skipped.values()) + dropped
        if metrics['source_rows'] != metrics['train_rows'] + metrics['val_rows'] + metrics['declared_skipped_rows']:
            error('source_count_mismatch')
    if result['errors']:
        result['readiness_reasons'].append('artifact_validation_failed')
    result['status'] = 'failed' if result['errors'] else 'passed'
    result['duration_seconds'] = round(time.perf_counter() - start, 6)
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
