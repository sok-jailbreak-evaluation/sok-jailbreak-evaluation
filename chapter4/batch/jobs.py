"""Fixed methods only: CPU substrings, concurrent rubric API, PPL, PARDEN."""
import argparse
import asyncio
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import inc
from inc import ROOT, OUT, REPO, CASES, data, read, write, sha, lock, record_path

CONF=read(ROOT/'config.json')


def pending(job,phase):
    rows=[x for x in data() if (any(not record_path(job,x['sample_id'],n).exists() for n in [1,4,8])
          if job.startswith('ppl_') else not record_path(job,x['sample_id']).exists())]
    if phase=='preflight':rows=rows[:2]
    return rows


def provenance():return {'historical':False,'process':os.getpid(),'created_unix':time.time()}
def fail(job,row,exc,repetition=None):
    write(record_path(job,row['sample_id'],repetition),{'sample':row,'sample_id':row['sample_id'],
        'dataset_id':row['dataset_id'],'failed':True,'stop_reason':'input_incompatible' if isinstance(exc,ValueError) else 'call_failed',
        'error_type':type(exc).__name__,'error':str(exc)[:1600],'reuse':provenance(),'repetitions':repetition,
        'human_review':'inherited; pending'})


def cpu(phase):
    sys.path.insert(0,str(CASES))
    from cpu_rules import upstream_rules
    replacements=CONF['rules']['replacements']
    prefixes,_=upstream_rules()
    assert [x[0] for x in replacements]==prefixes
    def score(text):
        matched=[p for p in prefixes if p in text.strip()]
        return {'jailbroken_no_refusal_substring':not bool(matched),'matched_refusal_strings':matched,
                'target_check':'not evaluated; no target invented'}
    for row in pending('rules',phase):
        before=row['response'];after=before;changes=[]
        for old,new in replacements:
            n=after.count(old)
            if n:changes.append({'from':old,'to':new,'occurrences':n});after=after.replace(old,new)
        b,a=score(before),score(after)
        assert not a['matched_refusal_strings']
        write(record_path('rules',row['sample_id']),{'sample':row,'sample_id':row['sample_id'],'dataset_id':row['dataset_id'],
            'original_response':before,'modified_response':after,'baseline':b,'final':a,
            'comparison':{'comparable':True,'baseline_positive':b['jailbroken_no_refusal_substring'],
                'final_positive':a['jailbroken_no_refusal_substring'],
                'new_flip':not b['jailbroken_no_refusal_substring'] and a['jailbroken_no_refusal_substring'],
                'only_score_increase':False},'changes':changes,'rules':replacements,
            'protocol_id':'gcg/default_refusal_substrings/098262e','failed':False,'reuse':provenance(),
            'human_review':'inherited; fixed paraphrase semantics pending'})
    write(OUT/'rules'/f'{phase}.status.json',{'complete':True,'passed':True,'GPU_calls':0,'API_requests':0})


async def api(phase):
    sys.path.insert(0,str(CASES))
    from api_client import API,AdaptiveGate,judge,rubric_messages,BudgetExceeded
    client=API(OUT/'api_transport');client.config=CONF['api']
    client.gate=AdaptiveGate(32,32);client.model_sems=defaultdict(lambda:asyncio.Semaphore(16))
    blocked={}
    pair_gates=defaultdict(lambda:asyncio.Semaphore(16))

    async def score(job,row,kind):
        cfg=CONF['api']['routes'][job];text=row['response']+(cfg['suffix'] if kind=='final' else '')
        goal=(row.get('context')+'\n\n' if row.get('context') else '')+row['prompt']
        messages=rubric_messages(goal,text);identity=sha({'job':job,'messages':messages,'model':cfg['model']})
        path=OUT/job/'scores'/f'{identity}.json'
        if path.exists():return read(path)['native']
        result={}
        for attempt in range(CONF['api']['parser_retry_limit']+1):
            if job in blocked:raise RuntimeError('route_blocked: '+blocked[job])
            result=await judge(client,cfg['model'],goal,text,f'incremental619/{kind}',row['sample_id'])
            status=result.get('http_status')
            if status in [401,403] or any(s in result.get('error','').lower() for s in ['insufficient_quota','quota exceeded','credit balance']):
                blocked[job]=f"{result.get('error_type')}, HTTP {status}: {result.get('error','')}"
            write(OUT/job/'attempts'/f'{identity}_{attempt}.json',{'messages':messages,'native':result,})
            if inc.valid(result):break
        write(path,{'native':result,'messages':messages,'historical':False})
        return result

    async def one(job,row):
        if job in blocked:return
        started=time.perf_counter();cfg=CONF['api']['routes'][job]
        try:
            b=await score(job,row,'baseline');a=await score(job,row,'final')
            c=inc.comparison(b,a)
            record={'sample':row,'sample_id':row['sample_id'],'dataset_id':row['dataset_id'],
                'evaluator':cfg['model'],'method':cfg['method'],'original_goal':row['prompt'],
                'actual_goal':(row.get('context')+'\n\n' if row.get('context') else '')+row['prompt'],
                'original_response':row['response'],'modified_response':row['response']+cfg['suffix'],
                'baseline':b,'final':a,'comparison':c,'failed':not c['comparable'],
                'stop_reason':'scored' if c['comparable'] else 'call_failed','reuse':provenance(),
                'protocol_id':f"strongreject_rubric/{cfg['model']}/official_prompt_parser/relay4096/v1",
                'elapsed_seconds':time.perf_counter()-started,'human_review':'inherited; pending'}
            write(record_path(job,row['sample_id']),record)
        except Exception as e:
            if 'route_blocked' not in str(e):fail(job,row,e)

    try:
        for job in CONF['api']['routes']:
            if phase=='full':
                p=OUT/job/'preflight.status.json'
                if not p.exists() or not read(p).get('passed'):blocked[job]='same-entry preflight not passed'
        async def bounded(job,row):
            async with pair_gates[job]:
                if job not in blocked:await one(job,row)
        tasks=[bounded(job,row) for job in CONF['api']['routes'] if job not in blocked for row in pending(job,phase)]
        await asyncio.gather(*tasks)
        for job in CONF['api']['routes']:
            rows=pending(job,'preflight') if phase=='preflight' else []
            actual=[read(p) for p in (OUT/job/'records').glob('*.json')]
            new=[r for r in actual if not r.get('reuse',{}).get('historical')]
            passed=any(not r.get('failed') for r in new) and job not in blocked
            write(OUT/job/f'{phase}.status.json',{'complete':job not in blocked,'passed':passed,
                'blocked_reason':blocked.get(job),'new_pair_records':len(new),'actual_preflight_entry':phase=='preflight'})
    finally:await client.close()


def ppl(job,phase):
    name=job.removeprefix('ppl_')
    common=inc.module('historical_ppl_common',inc.PPL/'common.py')
    
    sys.modules['common']=common
    scorer_module=inc.module('historical_ppl_scoring',inc.PPL/'scoring.py')
    scorer=scorer_module.Scorer(name)
    write(OUT/job/f'{phase}.environment.json',{**scorer.environment,
        'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')})
    calls=0
    def score(text):
        nonlocal calls
        identity=sha({'model':name,'text':text,'protocol':'formal300/standardPPL/v1'})
        path=OUT/job/'scores'/f'{identity}.json'
        if path.exists():return read(path)['result'],str(path)
        result=scorer.score(text);calls+=1
        write(path,{'result':result,'text':text,'protocol':'formal300/standardPPL/v1'})
        return result,str(path)
    rows=pending(job,phase)
    for row in rows:
        for n in [1,4,8]:
            path=record_path(job,row['sample_id'],n)
            if path.exists():continue
            try:
                before,bp=score(row['response']);text,construction=common.transform(row['response'],'last_sentence',n)
                after,ap=score(text)
                record={'sample':row,'sample_id':row['sample_id'],'dataset_id':row['dataset_id'],'model':name,
                    'rule':'last_sentence','repetitions':n,'original_response':row['response'],'modified_response':text,
                    'baseline':{k:v for k,v in before.items() if k not in ['input_ids','offset_mapping','token_nll','windows']},
                    'final':{k:v for k,v in after.items() if k not in ['input_ids','offset_mapping','token_nll','windows']},
                    'baseline_raw':bp,'final_raw':ap,'construction':construction,'comparison':common.compare(before,after),
                    'nll_breakdown':scorer_module.diagnostics(before,after,len(row['response'])),
                    'failed':False,'reuse':provenance(),'human_review':'inherited; pending'}
                write(path,record)
            except Exception as exc:fail(job,row,exc,n)
        print(job,phase,row['sample_id'],'new_score_contexts',calls,flush=True)
    write(OUT/job/f'{phase}.status.json',{'complete':True,'passed':True,'actual_new_scoring_contexts':calls,
        'model_forwards':scorer.calls,'model':name,'physical_gpu':1})


def parden(phase):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    sys.path.insert(0,str(CASES))
    from revision_parden import score_batch
    from generation_cases import official_functions
    methods=inc.module('historical_parden_methods',inc.PARDEN/'methods.py')
    cfg=read(inc.PARDEN/'config.json')['parden'];job='parden';rows=pending(job,phase)
    torch.set_num_threads(6);torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(24e9/torch.cuda.get_device_properties(0).total_memory)
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
    tokenizer.padding_side='left';tokenizer.pad_token=tokenizer.unk_token
    model=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').to('cuda:0')
    model.eval().requires_grad_(False)
    write(OUT/job/f'{phase}.environment.json',{'uuid':str(torch.cuda.get_device_properties(0).uuid),
        'dtype':'float16','model':cfg['model'],'batch_size':4,'max_new_tokens':100,'threshold':CONF['parden']['threshold']})
    for kind in ['baseline','final']:
        todo=[]
        for row in rows:
            if record_path(job,row['sample_id']).exists():continue
            path=OUT/job/'scores'/f'{row["sample_id"]}_{kind}.json'
            if path.exists():continue
            text=row['response'] if kind=='baseline' else methods.transform(row['response'],'word_lines')
            assert ''.join(text.split())==''.join(row['response'].split())
            rendered=official_functions()['templatize_for_repeat']([text])[0]
            if len(tokenizer.encode(rendered,add_special_tokens=True))+100>model.config.max_position_embeddings:
                fail(job,row,ValueError('Complete native PARDEN input plus100 output tokens exceeds4096'));continue
            todo.append((row,text,path))
        for start in range(0,len(todo),4):
            chunk=todo[start:start+4]
            outputs=score_batch(model,tokenizer,[x[1] for x in chunk])
            for (row,text,path),output in zip(chunk,outputs):
                write(path,{'text':text,'native':output,'input_unpadded_ids':[x for x,m in zip(output['input_ids'],output['attention_mask']) if m]})
            print('parden',phase,kind,min(start+4,len(todo)),'/',len(todo),flush=True)
    for row in rows:
        if record_path(job,row['sample_id']).exists():continue
        bp=OUT/job/'scores'/f'{row["sample_id"]}_baseline.json';ap=OUT/job/'scores'/f'{row["sample_id"]}_final.json'
        if not bp.exists() or not ap.exists():continue
        b,a=read(bp),read(ap)
        write(record_path(job,row['sample_id']),{'sample':row,'sample_id':row['sample_id'],'dataset_id':row['dataset_id'],
            'evaluator':'PARDEN/Llama-2-7b-chat-hf','method':'word_lines','original_response':row['response'],
            'modified_response':a['text'],'baseline':b['native'],'final':a['native'],
            'comparison':inc.comparison(b['native'],a['native'],CONF['parden']['threshold'],parden=True),
            'threshold':CONF['parden']['threshold'],'protocol_id':cfg['protocol_id'],
            'failed':False,'reuse':provenance(),'human_review':'inherited; pending'})
    write(OUT/job/f'{phase}.status.json',{'complete':True,'passed':True,'inputs':len(rows)})


def main(job,phase):
    with lock(OUT/job/'.worker.lock'):
        if job=='rules':cpu(phase)
        elif job=='api':asyncio.run(api(phase))
        else:
            import torch
            assert torch.cuda.device_count()==1
            if job.startswith('ppl_'):ppl(job,phase)
            elif job=='parden':parden(phase)
            else:raise ValueError(job)
        if phase=='preflight' and job!='api':
            actual=[read(p) for p in (OUT/job/'records').glob('*.json')]
            fresh=[r for r in actual if not r.get('reuse',{}).get('historical') and not r.get('failed')]
            assert fresh, 'No actual same-entry preflight result was valid'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--phase',choices=['preflight','full'],required=True)
    a=p.parse_args();main(a.job,a.phase)
