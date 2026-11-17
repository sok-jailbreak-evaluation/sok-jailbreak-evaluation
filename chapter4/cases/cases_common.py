"""Versioned, atomic records for the evaluator case study (no credentials)."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LEGACY = ROOT.parent/'batch/reference'
PYTHON = 'python3'

def sha(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()

def read(path: Path):
    return json.loads(path.read_text())

def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)

def sample_rows():
    return [json.loads(s) for s in (ROOT / 'data/unharmful25.jsonl').read_text().splitlines()]

def identity(names: list[str], extra=None):
    payload = {'code': names, 'extra': extra}
    key = sha(json.dumps(payload, sort_keys=True))[:16]
    return key, payload

def versioned(path: Path, key: str, payload: dict):
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / 'version.json'
    write(manifest, {'created_unix': time.time(), **payload})

def case_record(item, evaluator, sample, original, modified, baseline, final,
                construction, elapsed=0, **extra):
    return {'item': item, 'evaluator': evaluator, 'sample_id': sample,
            'original_object': original, 'modified_object': modified,
            'baseline': baseline, 'final': final, 'construction': construction,
            'elapsed_seconds': elapsed,
            'human_review': {'status': 'pending', 'reviewer': None},
            'semantic_screen': {'status': 'pending', 'type': 'not_human'}, **extra}
