"""Independent standard-library evidence helpers; never import audit or fixture tools."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

DATASETS = {'tagging', 'extraction'}
CONTRACTS = {'python_syntax', 'train_lora_argparse_help'}


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
    parsed = json.loads(text, parse_constant=reject_constant, parse_float=finite_float, object_pairs_hook=reject_duplicates)
    pending = [parsed]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            # Escaped lone surrogates decode as strings but are not Unicode scalar text.
            value.encode('utf-8')
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return parsed


def identity(root):
    def safe(value):
        return value if re.fullmatch(r'[A-Za-z0-9_./:@+-]{1,200}', value) else 'unavailable'
    try:
        commit = subprocess.run(['git','-C',str(root),'rev-parse','HEAD'], capture_output=True,text=True,check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    return dict(repository=safe(os.environ.get('GITHUB_REPOSITORY', '100-hours-a-week/KTB4-17th-AI_Training')),
                commit=safe(commit), run_id=safe(os.environ.get('GITHUB_RUN_ID','local')),
                run_attempt=safe(os.environ.get('GITHUB_RUN_ATTEMPT','1')),
                generated_at=dt.datetime.now(dt.timezone.utc).isoformat())
