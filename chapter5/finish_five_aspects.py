from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,csv,io
p=Path('.')
def obj(f):return json.loads(f.read_bytes())
def file(d,key):return p/d/(hashlib.sha256(key.encode()).hexdigest()+'.json')
from aspect_fields import semantic_fields
new=obj(p/'new_final.json')
out=new
counts=defaultdict(Counter);statuses=defaultdict(Counter);details=[]
for r in map(json.loads,(p/'input/all.jsonl').open()):
    k=r['attempt_id']; cell=(r['target'],r['method']);x=out[k]
    g=obj(file('G',r['prompt_digest'])) if r['prompt_digest'] else dict(G=0,status='unavailable_input')
    d=obj(file('D',k));assert d['key']==k
    separate=semantic_fields(r,obj(file('results/hybrid',k)))
    vals={'G':g['G'],'H':x['h'],'R':separate['R'],'A':separate['A'],'D':d['D']}
    assert vals['D']<=vals['H'] and set(vals.values())<={0,1}
    if d['D']: assert d['verifier']['quote'] in r['response'][d['split_offset']:]
    counts[cell].update(vals);counts[cell]['n']+=1
    statuses[cell].update(['G_'+g['status'],'D_'+d['status'],'H_'+x['policy_status'],'R_'+separate['R_status'],'A_'+separate['A_status']])
    details.append(dict(id=k,**vals,prompt_sha256=r['prompt_digest'],response_sha256=r['response_sha256'],g_status=g['status'],d_status=d['status'],R_status=separate['R_status'],A_status=separate['A_status'],partial_help=x['p'],task_before_verification=x['task_before_verification'],task_pass=x['t']))
assert len(details)==len(out)==len({r['id'] for r in details})
report={'aspect_by_column':dict(G='L1.2',H='L2.2',R='L2.3',A='L2.4',D='L3.1'),'groups':[dict(target=t,method=m,**c,statuses=dict(statuses[t,m])) for (t,m),c in sorted(counts.items())],'planned':len(details),'new':len(new)}
(p/'final_by_slot.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in details))
(p/'table7_verified.json').write_text(json.dumps(report,indent=2)+'\n')
print([(k,dict(c)) for k,c in counts.items()])
