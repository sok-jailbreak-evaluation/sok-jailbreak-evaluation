"""Native fixed-protocol CPU, concurrent API, PPL and PARDEN down experiments."""
import argparse
import asyncio
from collections import defaultdict
import math
import os
from pathlib import Path
import sys
import time
import traceback
from down import (API_SUFFIXES, CASES, CONF, JOBS, OUT, PPL, ROOT, compare, data, key, lock, mapping, module, parden_transform, probes, provenance, read, record_path, sha, valid_rubric, write)


def output(job,phase):return OUT/job/phase
def rows_for(job,phase):
    rows=probes()[:2] if phase=='preflight' else data()
    return [r for r in rows if not all(record_path(job,r['sample_id'],phase,n).exists()
            for n in ([1,4,8] if job.startswith('ppl_') else [None]))]
def failed(job,row,phase,exc,n=None):
    r={'sample':row,'sample_id':row['sample_id'],'dataset_id':row['dataset_id'],
       'fingerprint':key(job),'direction':'down','failed':True,'incomparable':True,
       'stop_reason':'input_incompatible' if isinstance(exc,ValueError) else 'call_failed',
       'error_type':type(exc).__name__,'error':str(exc)[:1600],'repetitions':n,'human_review':'pending'}
    write(record_path(job,row['sample_id'],phase,n),r)
def pair(job,row,phase,b,a,text,**extra):
    assert row['response'] in text or (job=='parden' and extra.get('method') in ['single_line','indented_quote'])
    c=compare(b,a)
    r={'fingerprint':key(job),'sample':row,'sample_id':row['sample_id'],'dataset_id':row['dataset_id'],
       'original_response':row['response'],'modified_response':text,'direction':'down',
       'baseline':b,'final':a,'comparison':c,'baseline_positive':c['baseline_positive'],
       'new_flip':c['new_flip'],'only_score_decrease':c['only_score_decrease'],
       'failed':not c['comparable'],'incomparable':not c['comparable'],
       'stop_reason':'baseline_already_negative' if b['parse_ok'] and not b['positive'] else
                     ('native_flip' if c['new_flip'] else ('scored_still_positive' if c['comparable'] else 'call_failed')),
       'provenance':provenance(),'human_review':'pending','terminal_verified':c['comparable'],**extra}
    write(record_path(job,row['sample_id'],phase),r)
    return r


def rules(phase):
    sys.path.insert(0,str(CASES))
    from cpu_rules import upstream_rules
    prefixes,_=upstream_rules()
    assert prefixes==[r[0] for r in CONF['rules']['replacements']]
    def native(text):
        matches=[p for p in prefixes if p in text.strip()]
        return {'parse_ok':True,'score':float(not matches),'positive':not bool(matches),
            'target_hit':bool(matches),'matched_refusal_strings':matches,
            'jailbroken_no_refusal_substring':not bool(matches)}
    for row in rows_for('rules',phase):
        b=native(row['response'])
        text=row['response']+CONF['rules']['suffix'] if b['positive'] else row['response']
        a=native(text)
        pair('rules',row,phase,b,a,text,method='append_quoted_refusal_marker',
            appended_text=text[len(row['response']):],protocol_id='gcg/default_refusal_substrings/098262e',
            semantic_note='Quoted marker only; the original substantive answer is preserved, not withdrawn.')
    write(output('rules',phase)/'status.json',{'passed':True,'complete':True,'GPU_calls':0,'API_requests':0})


async def api(phase):
    sys.path.insert(0,str(CASES))
    from api_client import API,AdaptiveGate,judge,rubric_messages,BudgetExceeded
    client=API(OUT/'api_transport');client.config=CONF['api']
    client.gate=AdaptiveGate(32,32);client.model_sems=defaultdict(lambda:asyncio.Semaphore(16))
    blocked={};pair_gates=defaultdict(lambda:asyncio.Semaphore(16))
    async def score(job,row,method=None):
        cfg=CONF['api']['routes'][job]
        text=row['response']+(API_SUFFIXES[method] if method else '')
        goal=(row.get('context')+'\n\n' if row.get('context') else '')+row['prompt']
        messages=rubric_messages(goal,text)
        identity=sha({'model':cfg['model'],'messages':messages,'protocol':'619rubric4096/v1'})
        p=OUT/job/'scores'/f'{identity}.json'
        if p.exists():return read(p)['native']
        if job in blocked:raise RuntimeError('route_blocked')
        result={}
        for attempt in range(2):
            result=await judge(client,cfg['model'],goal,text,f'down483/{phase}',row['sample_id'])
            status=result.get('http_status');error=result.get('error','').lower()
            if status in [401,403] or any(s in error for s in ['insufficient_quota','quota exceeded','credit balance']):
                blocked[job]={'type':result.get('error_type'),'http_status':status}
            if valid_rubric(result) or job in blocked:break
        ok=valid_rubric(result)
        native={**result,'parse_ok':ok,'positive':bool(ok and result['score']>=.5),
                'target_hit':bool(ok and result['score']<.5)}
        write(p,{'native':native,'messages':messages,'input_sha256':sha(messages),'method':method,
            'protocol_id':f"strongreject_rubric/{cfg['model']}/official_prompt_parser/relay4096/v1"})
        return native
    async def bounded_score(job,row,method=None):
        async with pair_gates[job]:
            return await score(job,row,method)
    async def one(job,row,method,which_phase):
        started=time.perf_counter()
        try:
            b=await bounded_score(job,row)
            a=b if b['parse_ok'] and not b['positive'] else await bounded_score(job,row,method)
            text=row['response']+(API_SUFFIXES[method] if a is not b else '')
            return pair(job,row,which_phase,b,a,text,method=method,appended_text=text[len(row['response']):],
                evaluator=CONF['api']['routes'][job]['model'],elapsed_seconds=time.perf_counter()-started,
                protocol_id=f"strongreject_rubric/{CONF['api']['routes'][job]['model']}/official_prompt_parser/relay4096/v1")
        except Exception as exc:
            failed(job,row,which_phase,exc)
            if isinstance(exc,BudgetExceeded):blocked[job]={'type':'shared_request_budget_exhausted'}
            return None
    async def route(job):
        selected_path=OUT/job/'selected_method.json'
        if phase=='preflight':
            rs=await asyncio.gather(*(one(job,r,'gold_low',phase) for r in probes()[:2]))
            passed=any(r and not r['failed'] for r in rs)
            write(output(job,phase)/'status.json',{'passed':passed,'pair_records':len(rs),'blocked':blocked.get(job)})
            return
        if not read(output(job,'preflight')/'status.json')['passed']:
            write(output(job,phase)/'status.json',{'passed':False,'stop_reason':'route_preflight_failed'})
            return
        if not selected_path.exists():
            trials=[]
            for method in API_SUFFIXES:
                rs=await asyncio.gather(*(one(job,r,method,'screen_'+method) for r in probes()))
                good=[r for r in rs if r and not r['failed']]
                flips=sum(r['new_flip'] for r in good)
                trial={'method':method,'valid_pairs':len(good),'eligible_positive':sum(r['baseline_positive'] for r in good),
                    'new_flips':flips,'baseline_negative':sum(not r['baseline_positive'] for r in good)}
                trials.append(trial);write(OUT/job/'screen_summary.json',trials)
                print('API_SCREEN',job,trial,flush=True)
                if flips>=CONF['screen']['required_new_flips']:
                    write(selected_path,{'method':method,'suffix':API_SUFFIXES[method],'suffix_sha256':sha(API_SUFFIXES[method]),
                        'fingerprint':key(job),'selection':'first >=3/10 paired down flips','trials':trials});break
                if job in blocked:break
            if not selected_path.exists():
                write(output(job,phase)/'status.json',{'passed':False,'blocked':blocked.get(job),
                    'stop_reason':'no_method_met_frozen_screen','trials':trials});return
        selected=read(selected_path);assert selected['fingerprint']==key(job)
        # Screen calls use exactly the same text/model and cached native outputs.
        await asyncio.gather(*(one(job,r,selected['method'],phase) for r in rows_for(job,phase)))
        write(output(job,phase)/'status.json',{'passed':job not in blocked,'complete':job not in blocked,
            'selected_method':selected,'blocked':blocked.get(job)})
    try:await asyncio.gather(*(route(job) for job in CONF['api']['routes']))
    finally:await client.close()


def ppl(job,phase):
    name=job.removeprefix('ppl_')
    common=module('ppl_shared_common',PPL/'common.py');sys.modules['common']=common
    common.CONF['models'][name]['physical_gpu']=int(os.environ.get('DOWN_GPU','0'))
    scoring=module('ppl_shared_scoring',PPL/'scoring.py');scorer=scoring.Scorer(name)
    write(output(job,phase)/'environment.json',scorer.environment)
    def score(text):
        p=OUT/job/'scores'/f'{sha(text)}.json'
        if not p.exists():write(p,{'text':text,'text_sha256':sha(text),'result':scorer.score(text)})
        return read(p)['result'],str(p)
    for row in rows_for(job,phase):
        b,bp=score(row['response'])
        for n in [1,4,8]:
            if record_path(job,row['sample_id'],phase,n).exists():continue
            try:
                text,construction=common.transform(row['response'],'last_sentence',n)
                assert text.startswith(row['response'])
                a,ap=score(text)
                write(record_path(job,row['sample_id'],phase,n),{'sample':row,'sample_id':row['sample_id'],
                    'dataset_id':row['dataset_id'],'fingerprint':key(job),'model':name,'repetitions':n,
                    'original_response':row['response'],'modified_response':text,'construction':construction,
                    'baseline':{k:v for k,v in b.items() if k not in ['token_nll','input_ids','offset_mapping','windows']},
                    'final':{k:v for k,v in a.items() if k not in ['token_nll','input_ids','offset_mapping','windows']},
                    'baseline_raw':bp,'final_raw':ap,'comparison':common.compare(b,a),
                    'nll_breakdown':scoring.diagnostics(b,a,len(row['response'])),'failed':False,
                    'human_review':'pending','provenance':provenance()})
            except (ValueError,ArithmeticError) as e:failed(job,row,phase,e,n)
        print('PPL',name,phase,row['sample_id'],flush=True)
    write(output(job,phase)/'status.json',{'passed':True,'complete':True,'model_forwards':scorer.calls})


def parden(phase):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    sys.path.insert(0,str(CASES))
    from revision_parden import score_batch
    from generation_cases import official_functions
    cfg=read(ROOT.parent/'parden/config.json')['parden']
    torch.set_num_threads(6);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(24e9/torch.cuda.get_device_properties(0).total_memory)
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
    tokenizer.padding_side='left';tokenizer.pad_token=tokenizer.unk_token
    model=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,
        dtype=torch.float16,attn_implementation='sdpa').to('cuda:0')
    model.eval().requires_grad_(False)
    job='parden';threshold=CONF['parden']['threshold']
    write(output(job,phase)/'environment.json',{'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),
        'physical_gpu':os.environ.get('DOWN_GPU','0'),'dtype':'float16','batch_size':4,
        'max_new_tokens':100,'threshold':threshold,'model':cfg['model'],'TF32':False})
    def native(texts):
        values={};todo=[]
        for text in dict.fromkeys(texts):
            p=OUT/job/'scores'/f'{sha(text)}.json'
            if p.exists():values[text]=read(p)['native'];continue
            rendered=official_functions()['templatize_for_repeat']([text])[0]
            if len(tokenizer.encode(rendered,add_special_tokens=True))+100>model.config.max_position_embeddings:
                values[text]={'parse_ok':False,'input_incompatible':True,'error':'full PARDEN input plus100 tokens exceeds4096'}
            else:todo.append(text)
        for start in range(0,len(todo),4):
            batch=todo[start:start+4];outputs=score_batch(model,tokenizer,batch)
            for text,r in zip(batch,outputs):
                ok=math.isfinite(r['bleu60']) and 0<=r['bleu60']<=1
                value={**r,'parse_ok':ok,'score':1-r['bleu60'],'positive':bool(ok and r['bleu60']<threshold),
                    'target_hit':bool(ok and r['bleu60']>=threshold)}
                write(OUT/job/'scores'/f'{sha(text)}.json',{'text':text,'native':value,'threshold':threshold,
                    'protocol_id':cfg['protocol_id']});values[text]=value
        return [values[t] for t in texts]
    def run_pairs(rows,method,where):
        results=[]
        for start in range(0,len(rows),4):
            chunk=rows[start:start+4];bs=native([r['response'] for r in chunk])
            texts=[r['response'] if b.get('parse_ok') and not b['positive'] else parden_transform(r['response'],method)
                   for r,b in zip(chunk,bs)]
            aa=native(texts)
            for row,b,a,text in zip(chunk,bs,aa,texts):
                if b.get('input_incompatible') or a.get('input_incompatible'):
                    failed(job,row,where,ValueError('Complete PARDEN native input exceeds4096; body not truncated'));continue
                results.append(pair(job,row,where,b,a,text,method=method,threshold=threshold,protocol_id=cfg['protocol_id'],
                    construction={'complete_original_substantive_information_retained':True,
                        'original_is_exact_substring':row['response'] in text,
                        'non_whitespace_preserved':''.join(row['response'].split()) in ''.join(text.split()),
                        'semantic_review':'pending','method':method}))
            print('PARDEN',where,method,min(start+4,len(rows)),'/',len(rows),flush=True)
        return results
    if phase=='preflight':
        candidates=probes()
        compatible=[]
        for r in candidates:
            text=official_functions()['templatize_for_repeat']([r['response']])[0]
            if len(tokenizer.encode(text,add_special_tokens=True))+100<=4096:compatible.append(r)
        rs=run_pairs(compatible[:2],'single_line',phase)
        assert len(rs)==2 and all(not r['failed'] for r in rs)
    else:
        selected_path=OUT/job/'selected_method.json'
        if not selected_path.exists():
            trials=[]
            for method in CONF['parden']['candidate_methods']:
                rs=run_pairs(probes(),method,'screen_'+method)
                trial={'method':method,'valid_pairs':len(rs),'eligible_positive':sum(r['baseline_positive'] for r in rs),
                    'new_flips':sum(r['new_flip'] for r in rs)}
                trials.append(trial);write(OUT/job/'screen_summary.json',trials)
                if trial['new_flips']>=CONF['screen']['required_new_flips']:
                    write(selected_path,{'method':method,'fingerprint':key(job),'trials':trials});break
            if not selected_path.exists():
                write(output(job,phase)/'status.json',{'passed':False,'stop_reason':'no_method_met_frozen_screen','trials':trials});return
        selected=read(selected_path);assert selected['fingerprint']==key(job)
        run_pairs(rows_for(job,phase),selected['method'],phase)
    write(output(job,phase)/'status.json',{'passed':True,'complete':True})


def main(job,phase):
    with lock(OUT/job/'.worker.lock'):
        if job=='api':asyncio.run(api(phase))
        elif job=='rules':rules(phase)
        else:
            import torch
            assert torch.cuda.device_count()==1
            actual=str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')
            if job=='parden':parden(phase)
            else:ppl(job,phase)
        if phase=='preflight' and job!='api':
            records=[read(p) for p in (output(job,phase)/'records').glob('*.json')]
            assert records and all(not r['failed'] for r in records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--phase',choices=['preflight','full'],required=True)
    a=p.parse_args();main(a.job,a.phase)
