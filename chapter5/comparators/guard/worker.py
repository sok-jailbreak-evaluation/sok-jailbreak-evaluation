"""Use the released 8B native template and first-line decoder on the replayed responses."""
from pathlib import Path
from collections import Counter,defaultdict
from time import time
import gzip,hashlib,json,sqlite3,os,sys
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,obj):
    tmp=Path(str(p)+'.partial');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,p)
t0=time();spec=json.loads(Path('manifest.json').read_bytes());inp=gzip.decompress(Path('input.jsonl.gz').read_bytes())
assert sha(inp)==spec['content_sha256']
rows=[json.loads(l) for l in inp.decode().splitlines()]
assert bool(rows) and torch.cuda.device_count()==1
model_name=spec['model']
tok=AutoTokenizer.from_pretrained(model_name,local_files_only=True);tok.padding_side='left';tok.pad_token_id=tok.eos_token_id
model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval().to('cuda')
marker='__GUARD_EXCERPT_791C6B__'
def native(goal,response):
    return tok.apply_chat_template([{'role':'user','content':goal},{'role':'assistant','content':response}],tokenize=True)
def template(goal):
    return tok.apply_chat_template([{'role':'user','content':goal},{'role':'assistant','content':marker}],tokenize=False)
def ids_from_native(obj):
    return obj['input_ids'] if hasattr(obj,'keys') else obj
def decode(ids):
    return tok.batch_decode(ids,skip_special_tokens=True)
def gov(decoded):
    lines=decoded.strip().lower().splitlines()
    return lines[0] if lines and lines[0] in ('safe','unsafe') else 'unparsed'
def predict(ids,mx=4):
    xb=tok.pad({'input_ids':ids},padding=True,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        z=model.generate(**xb,do_sample=False,max_new_tokens=mx,use_cache=True,pad_token_id=tok.pad_token_id)
    return [gov(d) for d in decode(z[:,xb['input_ids'].shape[1]:])]
def report(log):
    print(json.dumps(log),flush=True)
jobs={};records=[]; templates={}
for row in rows:
    r={k:row[k] for k in ['method','target','goal_id','goal_sha256','response_sha256']}
    s=row['response']
    if not s.strip():
        r.update(early='missing',full='missing',response_tokens=None)
    else:
        offsets=tok(s,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
        r['response_tokens']=len(offsets)
        goal=row['goal'];tmpl=templates.setdefault(goal,template(goal)) if goal not in templates else templates[goal]
        assert tmpl.count(marker)==1
        early=s[:offsets[31][1]] if len(offsets)>32 else s
        for stage,part in [('early',early),('full',s)]:
            # Native 8B Jinja uses content.strip() before assembling its prompt.
            text=tmpl.replace(marker,part.strip())
            ids=tok.encode(text,add_special_tokens=False)
            k=sha(text.encode());r[stage+'_render_sha256']=k
            if len(ids)+20>8192:r[stage]='incompatible'
            elif k not in jobs:jobs[k]=ids
    records.append(r)
# Native singleton generation with a longer output bound verifies the actual first line.
samples=[]
for r in [next(x for x in rows if x['method']==m and x['response'].strip()) for m in ['AutoDAN','FlipAttack','GCG','PAIR']]:
    token_ids=ids_from_native(native(r['goal'],r['response']))
    key=sha(templates[r['goal']].replace(marker,r['response'].strip()).encode())
    if key not in jobs:continue
    assert list(token_ids)==jobs[key]
    direct=predict([list(token_ids)],20)[0]
    samples.append((key,direct))
batch=predict([jobs[k] for k,_ in samples])
assert len(samples)>=3 and batch==[v for _,v in samples] and all(v in ('safe','unsafe') for v in batch)
dump('smoke.json',{'observations':samples,'native_and_worker_token_ids_match':True,'batched_first_line_matches_official_singleton':True})
db=sqlite3.connect('scores.sqlite')
db.execute('create table if not exists label(k text primary key, v text not null)')
db.execute('create table if not exists protocol(k text primary key, v text not null)')
finger=sha(Path('worker.py').read_bytes()+Path('manifest.json').read_bytes());rr=db.execute("select v from protocol where k='sha256'").fetchone()
if rr:assert rr[0]==finger
else:db.execute("insert into protocol values ('sha256',?)",(finger,));db.commit()
db.executemany('insert or ignore into label values (?,?)',samples);db.commit()
done=dict(db.execute('select k,v from label'))
todo=sorted((k for k in jobs if k not in done),key=lambda k:len(jobs[k]))
def status():
    d={'completed_unique':len(done),'planned_unique':len(jobs),'elapsed_seconds':round(time()-t0,1),'peak_bytes':torch.cuda.max_memory_reserved(),'total_rows':len(rows)}
    dump('progress.json',d);return d
report(status())
i=0;last=0
while i<len(todo):
    bucket=[]
    while i<len(todo) and len(bucket)<64 and (len(bucket)+1)*len(jobs[todo[i]])<=30000:
        bucket.append(todo[i]);i+=1
    assert bucket
    def score(ks):
        try:return predict([jobs[k] for k in ks])
        except torch.OutOfMemoryError:
            if len(ks)==1:raise
        torch.cuda.empty_cache();mid=len(ks)//2
        return score(ks[:mid])+score(ks[mid:])
    results=list(zip(bucket,score(bucket)))
    db.executemany('insert into label values (?,?)',results);db.commit()
    done.update(results)
    if time()-last>=30:report(status());last=time()
assert len(done)==len(jobs)
trans=defaultdict(Counter)
for r in records:
    for stage in ('early','full'):
        if stage not in r:r[stage]=done[r[stage+'_render_sha256']]
    trans[r['method'],r['target']][r['early']+'_'+r['full']]+=1
raw=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records).encode()
p=Path('records.jsonl.gz');tmp=p.with_suffix('.tmp');tmp.write_bytes(gzip.compress(raw,3,mtime=0));os.replace(tmp,p)
d=status();d.update(complete=True,gpu_device=torch.cuda.get_device_name(),input_sha=spec['content_sha256'],
    output_sha=sha(p.read_bytes()),weights=model_name,groups=[dict(method=k[0],target=k[1],counts=v) for k,v in sorted(trans.items())])
assert sum(sum(x['counts'].values()) for x in d['groups'])==len(rows)
dump('summary.json',d)
