from pathlib import Path
from types import SimpleNamespace
import asyncio,json,sys
sys.path[:0]=[str(Path('condition_code').resolve()),str(Path('code').resolve())]
from evidence_core import path_for,sha,conjunction,save
from review_driver import goal_run,response_run,API
from protocol import digest
root=Path('conditions')
rows=list(map(json.loads,Path('input/attempts.jsonl').open()))
def read(f):return json.loads(f.read_bytes())
async def main():
    jobs=[]; out={}
    for row in rows:
        aid=row['attempt_id'];gid=row['goal_id']
        fp=path_for(Path('results/hybrid'),aid);e=read(fp);s=read(path_for(Path('results/static'),aid))
        spec=read(path_for(Path('specifications'),gid))
        assert spec['goal_sha256']==sha(row['goal']) and s['response_sha256']==row['response_sha256']
        if row['response_status']=='valid': assert e.get('response_sha256',row['response_sha256'])==row['response_sha256']
        req=spec['requirements']; eligible=(row['response_status']=='valid' and e['status']=='valid' and (not e['review_triggers'] or bool(e.get('review_record') and e.get('review_verification'))) and 'policy' in {c['id'] for c in req} and len(req)>=2)
        cond=e.get('conditions',{})
        policy=cond.get('policy',{}).get('status','?') if eligible else '?'
        task=conjunction([cond.get(c['id'],{}).get('status','?') for c in req if c['id']!='policy']+
                [c['outcome'] for c in s.get('checks',[]) if c['applicable'] and c['required_for_completion']]) if eligible and 'checks' in s else '?'
        a=e.get('assistance','unresolved') if eligible else 'unresolved'
        out[aid]=dict(h=int(policy=='1'),p=int(policy=='1' and a=='partial'),policy_status=policy,requirements_status=task,assistance=a,response_sha256=row['response_sha256'],condition_review='not_routed')
        if not (eligible and (task=='1' or (policy=='1' and e['properties']['correctness']['value']=='contradicted'))):continue
        prev=dict(requirements_status=task,policy_status=policy,assistance=a)
        jobs.append(dict(row=row,previous=prev,requirements=req,static=s,prior_evidence=e,source_hashes=dict(response=sha(row['response']),goal=sha(row['goal']),evidence=sha(fp.read_text()))))
    args=SimpleNamespace(env='.env',root=root,model='gpt-6-sol',budget=10000,concurrency=32)
    root.mkdir(exist_ok=True)
    api=API(args)
    try:
        goals={r['row']['goal_id']:r for r in jobs}
        await asyncio.gather(*(goal_run(api,r,root) for r in goals.values()))
        for gid,r in goals.items():
            plan=read(path_for(root/'goal_plans',gid))
            assert plan['goal_sha256']==digest(r['row']['goal']) and plan['original_requirements_sha256']==digest([x for x in r['requirements'] if x['id']!='policy'])
        await asyncio.gather(*(response_run(api,r,root) for r in jobs))
    finally:await api.client.aclose()
    from recovery_format import make_api_class,recover_one
    import review_driver as driver
    import protocol
    recover_args=SimpleNamespace(**{**vars(args),'root':Path('conditions/format_recovery'),'base_root':root})
    recover_args.root.mkdir(exist_ok=True)
    recover_api=make_api_class(protocol,driver)(recover_args)
    try:
        fixes=[]
        for r in jobs:
            old=read(path_for(root/'results',r['row']['attempt_id']))
            if old['status'] in ('unavailable_verification','unavailable_review_verification'):
                fixes.append(recover_one(recover_args,protocol,driver,recover_api,old,r))
        await asyncio.gather(*fixes)
    finally:await recover_api.client.aclose()
    for r in jobs:
        aid=r['row']['attempt_id']; fixed=path_for(recover_args.root/'results',aid);item=read(fixed if fixed.exists() else path_for(root/'results',aid));assert item['terminal']
        out[aid].update({c:item[c] for c in ['h','p','assistance','policy_status']},requirements_status=item['verified_requirements_status'],condition_review=item['status'])
    assert len(out)==len(rows) and all(i['p']<=i['h'] for i in out.values())
    save(Path('new_final.json'),out)
    print('records',len(out),'routed',len(jobs),'goals',len(goals))
asyncio.run(main())
