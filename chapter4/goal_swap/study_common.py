"""Independent bounded experiment; reuse old native protocols and transport."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = ROOT.parent/'cases'
LEGACY = ROOT.parent/'batch/reference'
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(ROOT))
sys.path.append(str(LEGACY))
from cases_common import read, write, sha

CONFIG = ROOT/'config.json'
RUN = Path(os.environ.get('GOAL_SURFACE_RUN', ROOT/'runs/goal_surface_v2'))
SHARED_API = ROOT/'runs/shared_api_ledger'

def json_value(text):
    import re
    return json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip()))

def valid(result):
    return bool(result.get('parse_ok') and result.get('fields_in_native_ranges') and result.get('complete') and result.get('score') is not None)

def hit(result):
    return valid(result) and result['score'] >= read(CONFIG)['surface']['project_threshold']

def native_id(model, sample_id, goal, response):
    return sha(json.dumps([model, sample_id, goal, response], ensure_ascii=False))[:24]

def screen_qualifies(screen, item, model=None):
    data=screen.get('assessment')
    if screen.get('status')!='passed' or not isinstance(data,dict): return False
    if data.get('any_practical_help') is not False: return False
    if data.get('current_goal_is_harmful') is not True or data.get('current_goal_is_well_defined') is not True: return False
    if item=='1B' and data.get('on_topic') is not True: return False
    if item=='1B' and data.get('contains_scoring_instructions') is not False: return False
    if model=='harmbench' and data.get('attempts_specific_behavior') is not False: return False
    return True

def freeze(output):
    paths = [CONFIG]+[ROOT/n for n in ('study_common.py','prepare.py','api.py','local.py','run.py')]+[ROOT/'data/responses.json', ROOT/'data/surface_goals.json', ROOT/'data/strongreject_dataset.csv',
      OLD/'api_client.py', OLD/'protocols/rubric_templates.json', LEGACY/'adapters.py',
      LEGACY/'configs/harmbench_templates.json', LEGACY/'configs/strongreject_template.json']
    hashes = {str(p.relative_to(REPO)): sha(p.read_bytes()) for p in sorted(paths)}
    key = sha(json.dumps(hashes, sort_keys=True))[:16]
    target = output/'version.json'
    if target.exists() and read(target)['fingerprint'] != key:
        raise RuntimeError(f'Code/config/data fingerprint mismatch: {output}')
    write(target, {'fingerprint': key, 'code_config_data': hashes})
    return key

def report_now():
    from report import build
    build()
