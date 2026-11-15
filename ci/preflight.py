"""Audit saved SFT files and always write sanitized JSON and JUnit evidence."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from ci.validators import validate_dataset


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'), help='repository root')
    parser.add_argument('--output-dir', type=Path, default=Path('reports'))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    checks = [validate_dataset(args.root / 'Lena-Training', kind) for kind in ('tagging','extraction')]
    failed = any(check['status'] == 'failed' for check in checks)
    result = dict(schema_version=1, **identity(args.root), status='failed' if failed else 'passed', checks=checks,
                  training_executed=False, external_model_downloads=0, gpu_used=False, training_ready=False,
                  readiness_reasons=sorted({reason for check in checks for reason in check['readiness_reasons']}))
    (args.output_dir / 'preflight.json').write_text(json.dumps(result, indent=2)+'\n',encoding='utf-8')
    suite = ET.Element('testsuite', name='saved_sft_preflight', tests=str(len(checks)), failures=str(sum(c['status']=='failed' for c in checks)), errors='0', skipped='0', time=str(sum(c['duration_seconds'] for c in checks)))
    for check in checks:
        case = ET.SubElement(suite,'testcase',classname='ci.validators',name=check['name'],time=str(check['duration_seconds']))
        if check['status'] == 'failed':
            ET.SubElement(case,'failure',message='saved artifact validation failed').text = json.dumps(check['errors'])
    ET.ElementTree(suite).write(args.output_dir / 'preflight-junit.xml',encoding='utf-8',xml_declaration=True)
    print(json.dumps(dict(status=result['status'], training_ready=False, checks=[dict(name=c['name'], status=c['status'],metrics=c['metrics']) for c in checks])))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
