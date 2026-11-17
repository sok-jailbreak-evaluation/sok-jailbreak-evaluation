"""One model, one visible GPU; unchanged reference search with explicit native down stops."""
import argparse
from collections import Counter
import json
import os
import platform
import sys
import time
import traceback

import torch
import transformers
from study import ROOT,OUT,CONF,REFERENCE,prepare,read,write,sha,lock,search_sample,normalized_gpu_uuid,GPU
from backend import create
from layout import build_layout,initial_suffix,InvalidControl
from search import run_search,propose
from checkpoint_io import IncrementalCheckpoint


def safe_cuda_rng():
    try:return torch.cuda.get_rng_state().cpu().tolist()
    except Exception:return None


def compatibility(adapter,rows):
    items=[]
    for position,row in enumerate(rows):
        sample=search_sample(row,position)
        item={'sample_id':row['sample_id'],'dataset_id':row['dataset_id'],'position':position}
        try:
            ids=adapter.encode(sample,'')
            item['baseline_tokens']=len(ids);item['baseline_compatible']=True
            suffix=initial_suffix([adapter],sample,20)
            layout=build_layout(adapter,sample,suffix,20)
            item.update(initial_compatible=True,initial_suffix=suffix,initial_tokens=len(layout.input_ids))
        except (ValueError,InvalidControl) as exc:
            item.setdefault('baseline_compatible',False)
            item.update(initial_compatible=False,error=str(exc))
        items.append(item)
    return items


def run(name,phase):
    key,rows=prepare();cfg=CONF['models'][name];output=OUT/name if phase=='full' else OUT/name/'preflight'
    with lock(OUT/name/'.worker.lock'):
        if phase=='full':
            for model in CONF['order']:
                pilot=read(OUT/model/'preflight/status.json')
                assert pilot['passed']
        if (output/'status.json').exists() and read(output/'status.json').get('complete'):
            print('Already complete',name,phase,flush=True);return
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        torch.set_num_threads(6);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        torch.cuda.set_per_process_memory_fraction(min(1.,cfg['max_gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory))
        started=time.perf_counter();adapter=create(name,output);load=time.perf_counter()-started
        write(output/'environment.json',{'evaluator':name,'metadata':adapter.metadata(),
            'model_load_seconds':load,'physical_gpu':GPU,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),'gpu_name':torch.cuda.get_device_name(0),
            'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
            'memory_limit_gib':cfg['max_gpu_memory_gib'],'TF32':False,'config':cfg})
        if phase=='preflight':
            compat=compatibility(adapter,rows)
            write(OUT/name/'compatibility.json',{'samples':compat,
                'baseline_incompatible':sum(not r['baseline_compatible'] for r in compat),
                'initial_incompatible':sum(not r['initial_compatible'] for r in compat),
                'policy':'do not truncate, replace or silently omit any formal task'})
            active=[x['position'] for x in compat if x['initial_compatible'] and not (OUT/name/'records'/f"{x['sample_id']}.json").exists()][:2]
            if not active:raise RuntimeError('No compatible formal sample for same-entry preflight')
        else:active=list(range(len(rows)))
        steps=2 if phase=='preflight' else 200
        for position in active:
            original=rows[position];sample=search_sample(original,position);sid=original['sample_id']
            path=output/'records'/f'{sid}.json'
            if path.exists():
                saved=read(path)
                assert saved['sample']==original
                continue
            task_started=time.perf_counter();meter_before=Counter(adapter.meter)
            run_id=f'{key}:{name}:{phase}:{steps}:{sid}'
            checkpoint=IncrementalCheckpoint(output/'checkpoints'/f'{sid}.json',run_id)
            saved_state=checkpoint.load();prior_audit=saved_state.get('_audit',{}) if saved_state else {}
            def audit():
                delta={k:v-meter_before.get(k,0) for k,v in adapter.meter.items()}
                prior=prior_audit.get('counts',{})
                return {'counts':dict(Counter(prior)+Counter(delta)),
                    'seconds':prior_audit.get('seconds',0)+time.perf_counter()-task_started,
                    'candidate_rng':'stateless torch.Generator: seed42 + source_index*100003 + zero_based_step*1009',
                    'torch_cpu_rng_state':torch.get_rng_state().tolist(),
                    'torch_cuda_rng_state':safe_cuda_rng()}
            def save(state):
                current=dict(state)
                current.setdefault('best',current.get('final'))
                current.setdefault('current_suffix',current['best_suffix'])
                current['_audit']=audit();
                if current['steps']:
                    current['steps'][-1]['cumulative_task_seconds']=current['_audit']['seconds']
                checkpoint.save(current)
                print(json.dumps({'evaluator':name,'phase':phase,'position':position,'sample_id':sid,
                    'step':len(current['steps']),'score':current['best']['judge']['score'],
                    'native_checks':current.get('native_stop_checks'),'status':current['status']}),flush=True)
            baseline=None;initial=None;terminal=None;result=None;probe=None
            try:
                basepath=output/'baselines'/f'{sid}.json'
                if basepath.exists():
                    base=read(basepath);assert base['sample']==original
                    baseline=base['native']
                else:
                    baseline=adapter.evaluate_native(sample,'','baseline')
                    write(basepath,{'sample':original,'native':baseline})
                if not baseline['parse_ok']:
                    raise AssertionError('Invalid native baseline; not a comparable low prediction')
                if phase=='full' and baseline['parse_ok'] and baseline['target_hit']:
                    suffix='';initial_suffix_text=None;terminal=baseline
                    reason='baseline_already_negative';steps_run=0;candidate_evals=0;rechecks=0;first=None
                    write(output/'checkpoints'/f'{sid}.json',{'run_id':run_id,'status':'complete_baseline_negative','current_suffix':'','best_suffix':'',
                        'step_count':0,'baseline':baseline,'_audit':audit()})
                else:
                    if saved_state:
                        if prior_audit.get('torch_cpu_rng_state'):
                            torch.set_rng_state(torch.tensor(prior_audit['torch_cpu_rng_state'],dtype=torch.uint8))
                            torch.cuda.set_rng_state(torch.tensor(prior_audit['torch_cuda_rng_state'],dtype=torch.uint8))
                    with adapter.role('search'):
                        if phase=='preflight' and saved_state is None:
                            run_search({'judge':adapter},sample,{'judge':'down'},{'judge':1.},cfg,
                                on_step=save,max_steps=1)
                            saved_state=checkpoint.load()
                            assert saved_state is not None
                        result=run_search({'judge':adapter},sample,{'judge':'down'},{'judge':1.},cfg,
                            checkpoint=saved_state,on_step=save,max_steps=steps)
                    assert result['execution_backend']=='reference'
                    suffix=result['best_suffix'];initial_suffix_text=result['initial_suffix']
                    build_layout(adapter,sample,suffix,20)
                    initial=result['initial_native_stop']
                    terminal=adapter.evaluate_native(sample,suffix,'final')
                    if not terminal['parse_ok']:
                        raise AssertionError('Invalid final native verdict')
                    checked=result['native_stop_cache'][suffix]
                    if terminal['parse_ok']!=checked['parse_ok'] or terminal['hit']!=checked['hit']:
                        raise AssertionError('Terminal native verdict differs from current-suffix early-stop verdict')
                    if terminal['parse_ok'] and abs(terminal['score']-checked['score'])>cfg['native_check_tolerance']:
                        raise AssertionError('Terminal native score differs from native early-stop score')
                    if result['first_success'] is not None and not terminal['hit']:
                        raise AssertionError('Terminal native verification did not preserve early success')
                    reason=result['stop_reason'];steps_run=result['steps_run']
                    candidate_evals=result['candidate_evaluations_total']
                    rechecks=sum(r['candidate_rechecks'] for r in result['steps']);first=result['first_success']
                    if phase=='preflight' and not result['steps']:
                        # Real-gradient health probe only when valid native step-0
                        # stopping correctly avoids every optimization step.
                        with adapter.role('preflight_probe'):
                            found,diag=propose(adapter,sample,initial_suffix_text,'down',128,256,
                                42+sample['source_row_index']*100003,False)
                            assert found and diag['gradient']['gradient_finite_nonzero']
                            values=adapter.score_suffixes(sample,found,'down',cfg['score_batch_size'])
                        probe={'gradient':diag,'scored_candidates':len(values),'scope':'preflight health check, not a formal search'}
                    save({**result,'current_suffix':suffix})
                comparable=bool(baseline['parse_ok'] and terminal['parse_ok'])
                new_flip=comparable and not baseline['hit'] and terminal['hit']
                initial_flip=bool(new_flip and initial is not None and initial['parse_ok'] and initial['hit'])
                search_flip=bool(new_flip and not initial_flip and first is not None and first['step']>0)
                record={'run_id':run_id,'evaluator':name,'phase':phase,'frozen_position':position,
                    'sample_id':sid,'dataset_id':original['dataset_id'],'sample':original,'direction':'down',
                    'actual_question':sample['prompt'],'search_seed_index':sample['source_row_index'],
                    'baseline_native':baseline,'initial_suffix':initial_suffix_text,'initial_native':initial,
                    'suffix':suffix,'modified_response':original['response']+suffix,'final_native':terminal,
                    'baseline_positive':baseline['parse_ok'] and baseline['positive'],
                    'initial_flip':initial_flip,'search_new_flip':search_flip,'new_flip':new_flip,
                    'only_score_decrease':bool(comparable and terminal['positive'] and terminal['score']<baseline['score']),
                    'incomparable':not comparable,'failed':False,'terminal_verified':True,
                    'stop_reason':reason,'steps_run':steps_run,'steps_budget':steps,'first_success':first,
                    'candidate_evaluations_total':candidate_evals,'candidate_rechecks':rechecks,
                    'native_stop_checks':result['native_stop_checks'] if result else 0,
                    'native_gate_enabled':True,'trace_file':str(checkpoint.trace_path.relative_to(OUT)),
                    'counts':audit()['counts'],'elapsed_seconds':audit()['seconds'],'model_load_seconds_shared':load,
                    'gradient_probe':probe,'protocol_id':adapter.protocol_id,'human_review':'pending'}
                if phase=='preflight':
                    assert result is not None and result['native_stop_checks']>=1
                    assert result['steps'] or probe is not None
                    restored=IncrementalCheckpoint(checkpoint.state_path,run_id).load()
                    assert restored['best_suffix']==suffix and len(restored['steps'])==steps_run
                    record['checkpoint_readback_passed']=True
            except Exception as exc:
                if isinstance(exc,torch.OutOfMemoryError) or ('CUDA' in str(exc) and any(t in str(exc).lower() for t in ['launch failure','device-side assert','driver shutting','out of memory'])):
                    raise  # Keep last atomic checkpoint; controller retries in a fresh CUDA process.
                latest=checkpoint.load() if checkpoint.state_path.exists() else None
                done_steps=result['steps'] if result else (latest['steps'] if latest else [])
                record={'run_id':run_id,'evaluator':name,'phase':phase,'frozen_position':position,
                    'sample_id':sid,'dataset_id':original['dataset_id'],'sample':original,'direction':'down',
                    'actual_question':sample['prompt'],'baseline_native':baseline,'initial_native':initial,
                    'final_native':terminal,'failed':True,'incomparable':True,'new_flip':False,
                    'baseline_positive':bool(baseline and baseline['parse_ok'] and baseline['positive']),
                    'initial_flip':False,'search_new_flip':False,'only_score_decrease':False,
                    'steps_run':len(done_steps),
                    'steps_budget':steps,'candidate_evaluations_total':sum(s['candidate_evaluations'] for s in done_steps),
                    'stop_reason':'input_incompatible' if isinstance(exc,(ValueError,InvalidControl)) else 'task_failed',
                    'error_type':type(exc).__name__,'error':str(exc)[:1800],
                    'counts':audit()['counts'],'elapsed_seconds':audit()['seconds'],'human_review':'pending'}
                write(output/'failures'/f'{sid}.json',{'record':record,'traceback':traceback.format_exc()})
                if phase=='preflight':
                    write(path,record);raise
                torch.cuda.empty_cache()
            write(path,record)
            print('COMPLETE',name,phase,position,sid,record['stop_reason'],'steps',record['steps_run'],flush=True)
        records=[read(output/'records'/f'{rows[i]["sample_id"]}.json') for i in active]
        passed=all(not r['failed'] and r.get('checkpoint_readback_passed') and r.get('native_gate_enabled') for r in records) if phase=='preflight' else True
        write(output/'status.json',{'complete':True,'passed':bool(passed),'evaluator':name,
            'phase':phase,'covered':len(records),'expected':len(active),'failed':sum(r['failed'] for r in records),
            'native_gate_enabled':True,'early_stop_observed':sum(r.get('stop_reason') in ('native_flip','native_flip_initial') for r in records),
            'model_load_seconds':load,'total_seconds':time.perf_counter()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'counts_this_process':dict(adapter.meter),'API_requests':0})
        if not passed:raise AssertionError('Preflight did not pass')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=CONF['order'],required=True)
    p.add_argument('--phase',choices=['preflight','full'],required=True);args=p.parse_args();run(args.model,args.phase)
