"""Prepare fresh evaluation inputs from selected attack attempts; preserve the submitted text."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()

def freeze(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:
            raise ValueError(f'Frozen input differs at {path}')
    else:
        with path.open('xb') as output:
            output.write(raw)

def lines(rows):
    return b''.join(canonical(r)+b'\n' for r in rows)

def prepare(source, root):
    goals, attempts, prompts = {}, {}, {}
    for line in source.read_bytes().splitlines():
        if not line.strip():
            continue
        r=json.loads(line)
        for k in ('target','method','goal_id','goal'):
            if not isinstance(r.get(k),str) or not r[k].strip():
                raise ValueError(f'Missing {k}')
        ids=[r[k] for k in ('target','method','goal_id')]
        if any('|' in item for item in ids):
            raise ValueError('Identifiers may not contain a vertical bar')
        key='|'.join(ids)
        if key in attempts or goals.setdefault(r['goal_id'],r['goal'])!=r['goal']:
            raise ValueError(f'Duplicate attempt or conflicting goal: {key}')
        response=r['response'] if r['response'] is not None else ''
        if not isinstance(response,str):
            raise ValueError('Expected response text or null')
        status='valid' if response.strip() else 'missing_response'
        prompt=r['attack_prompt']
        sig=None
        if prompt is not None:
            if not isinstance(prompt,str) or not prompt.strip():
                raise ValueError('Expected the submitted prompt or null')
            if r['attack_prompt_scope']!='single_submitted_user_turn':
                turns=json.loads(prompt)
                if not isinstance(turns,dict) or not any(k.startswith('submitted_user') for k in turns):
                    raise ValueError('Missing submitted user turn')
            item={'goal':r['goal'],'attack_prompt':prompt}
            sig=sha(canonical(item))
            prompts[sig]=item
        attempts[key]={'attempt_id':key,**dict(zip(('target','method','goal_id'),ids)), 'goal':r['goal'],
            'goal_sha256':sha(r['goal'].encode()), 'context':r.get('context') or '',
            'attack_prompt':prompt, 'attack_prompt_scope':r.get('attack_prompt_scope'),
            'prompt_digest':sig,'response':response,'response_sha256':sha(response.encode()),'response_status':status}
    if not attempts:
        raise ValueError('No selected attempts')
    raw=lines(attempts.values())
    for name in ('all.jsonl','attempts.jsonl'):
        freeze(root/'input'/name,raw)
    freeze(root/'input/goals.jsonl',lines({'goal_id':k,'goal':v} for k,v in goals.items()))
    freeze(root/'input/manifest.json',canonical({'outputs_sha256':{'attempts.jsonl':sha(raw)}}))
    sem=root/'comparators/semantic'
    unique=gzip.compress(canonical(prompts),mtime=0)
    freeze(sem/'unique.json.gz',unique)
    freeze(sem/'input_manifest.json',canonical({'unique_sha256':sha(unique)}))
    guard=root/'comparators/guard'
    freeze(guard/'input.jsonl.gz',gzip.compress(raw,mtime=0))
    freeze(guard/'manifest.json',canonical({'model':'../../models/Llama-Guard-3-8B','content_sha256':sha(raw)}))

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('source',type=Path)
    cli.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    args=cli.parse_args()
    prepare(args.source,args.root)
