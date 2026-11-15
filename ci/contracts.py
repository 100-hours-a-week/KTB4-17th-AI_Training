"""CPU syntax and safe argparse help contracts; never invoke a training command."""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ci.preflight import identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--output-dir',type=Path,default=Path('reports'))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    files = sorted(path for folder in ('ci','tests/pretraining','Lena-Training') for path in (args.root / folder).rglob('*.py'))
    invalid = 0
    for path in files:
        try:
            ast.parse(path.read_text(encoding='utf-8'),filename=str(path.relative_to(args.root)))
        except (OSError,UnicodeError,SyntaxError,RecursionError):
            invalid += 1
    checks = [dict(name='python_syntax',status='failed' if invalid or not files else 'passed',duration_seconds=round(time.perf_counter()-start,6),metrics=dict(python_files=len(files),invalid_files=invalid),errors=[dict(code='syntax_invalid_or_missing')] if invalid or not files else [],warnings=[],skips=[])]
    start = time.perf_counter()
    exit_code = 1
    try:
        run = subprocess.run([sys.executable,str(args.root / 'Lena-Training/training/tagging/train_lora.py'),'--help'],capture_output=True,timeout=20,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':''})
        exit_code = run.returncode
        if not run.stdout or b'--max-length' not in run.stdout:
            exit_code = 1
    except (OSError,subprocess.TimeoutExpired):
        pass
    checks.append(dict(name='train_lora_argparse_help',status='failed' if exit_code else 'passed',duration_seconds=round(time.perf_counter()-start,6),metrics=dict(exit_code=exit_code),errors=[dict(code='safe_help_failed')] if exit_code else [],warnings=[],skips=[]))
    failed = any(c['status'] != 'passed' for c in checks)
    result = dict(schema_version=1,**identity(args.root),status='failed' if failed else 'passed',checks=checks,training_executed=False,external_model_downloads=0,gpu_used=False)
    (args.output_dir / 'contracts.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],checks=checks)))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
