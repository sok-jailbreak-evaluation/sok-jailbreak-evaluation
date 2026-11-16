"""Read-only aggregation of old and incremental sample records into new reports."""
import csv
import json
from collections import Counter
from pathlib import Path
import statistics
import time
import inc
from inc import OUT, data, read, write


def main():
    source=data();ds_counts=Counter(x['dataset_id'] for x in source)
    all_rows=[];summaries={};failures=[];provenance=[]
    lines = ['# Evaluator results', '', '| Evaluator | Reused | New records | Pending | Incompatible | Invalid | Baseline positive | New flags | All inputs | Eligible inputs |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for job in inc.JOBS:
        rows=[read(p) for p in sorted((OUT/job/'records').glob('*.json'))]
        expected=len(source)*(3 if job.startswith('ppl_') else 1)
        def aggregate(rr,denominator):
            historical=sum(x.get('reuse',{}).get('historical',False) for x in rr)
            incompatible=sum(x.get('stop_reason')=='input_incompatible' for x in rr)
            failed=sum(bool(x.get('failed')) and x.get('stop_reason')!='input_incompatible' for x in rr)
            comps=[]
            for x in rr:
                if job in inc.SUFFIX_JOBS:
                    b,a=x.get('baseline_native') or {},x.get('final_native') or {}
                    valid=not x.get('failed') and b.get('parse_ok') and a.get('parse_ok')
                    comps.append({'baseline_positive':x.get('baseline_positive',False),
                        'new_flip':x.get('new_flip',False),'negative_valid':bool(valid and not b.get('hit'))})
                elif not job.startswith('ppl_'):
                    c=x.get('comparison',{});comps.append({'baseline_positive':c.get('baseline_positive') is True,
                        'new_flip':bool(c.get('new_flip')),'negative_valid':bool(c.get('comparable') and c.get('baseline_positive') is False)})
            flips=sum(x['new_flip'] for x in comps);eligible=sum(x['negative_valid'] for x in comps)
            return {'expected_records':denominator,'completed_records':len(rr),'historical_reused_records':historical,
                'new_records':len(rr)-historical,'pending_records':denominator-len(rr),'input_incompatible':incompatible,
                'call_or_parse_failed':failed,'baseline_positive':sum(x['baseline_positive'] for x in comps),
                'new_flips':flips,'new_flips_over_all_samples':flips/(denominator or 1),
                'valid_original_negative_pairs':eligible,'conditional_new_flip_rate':flips/eligible if eligible else None,
                'initial_flips':sum(bool(x.get('initial_flip')) for x in rr),'search_new_flips':sum(bool(x.get('search_new_flip')) for x in rr),
                'steps_exhausted':sum(x.get('stop_reason')=='steps_exhausted' for x in rr),
                'only_score_increase':sum(bool(x.get('only_score_increase',x.get('comparison',{}).get('only_score_increase'))) for x in rr)}
        summary=aggregate(rows,expected)
        summary['datasets']={d:aggregate([x for x in rows if x['dataset_id']==d],n*(3 if job.startswith('ppl_') else 1)) for d,n in ds_counts.items()}
        if job.startswith('ppl_'):
            strengths={}
            for n in [1,4,8]:
                rr=[x for x in rows if x.get('repetitions')==n and not x.get('failed')]
                def pplstats(xx):
                    return {'completed':len(xx),'ppl_decreased':sum(x['comparison']['ppl_decreased'] for x in xx),
                        'effect_at_least_10_percent':sum(x['comparison']['effect_at_least_10_percent'] for x in xx),
                        'median_relative_ppl_drop':statistics.median(x['comparison']['relative_ppl_drop'] for x in xx) if xx else None}
                strengths[n]={'all':pplstats(rr),'datasets':{d:pplstats([x for x in rr if x['dataset_id']==d]) for d in ds_counts}}
            summary['strengths']=strengths
        else:
            rate=summary['conditional_new_flip_rate']
            lines.append(f'| {job} | {summary["historical_reused_records"]} | {summary["new_records"]} | {summary["pending_records"]} | {summary["input_incompatible"]} | {summary["call_or_parse_failed"]} | {summary["baseline_positive"]} | {summary["new_flips"]} | {100*summary["new_flips_over_all_samples"]:.2f}% | {100*rate:.2f}%'+' |' if rate is not None else f'| {job} | {summary["historical_reused_records"]} | {summary["new_records"]} | {summary["pending_records"]} | {summary["input_incompatible"]} | {summary["call_or_parse_failed"]} | {summary["baseline_positive"]} | {summary["new_flips"]} | {100*summary["new_flips_over_all_samples"]:.2f}% | — |')
        summaries[job]=summary
        for x in rows:
            all_rows.append({'job':job,**x})
            if x.get('failed'):failures.append({'job':job,'sample_id':x['sample_id'],'stop_reason':x.get('stop_reason'),'error':x.get('error')})
            provenance.append({'job':job,'sample_id':x['sample_id'],'repetitions':x.get('repetitions'),'reuse':x.get('reuse'),
                               'checkpoint_source':str(OUT/job/'checkpoint_sources'/f'{x["sample_id"]}.json') if (OUT/job/'checkpoint_sources'/f'{x["sample_id"]}.json').exists() else None})
    lines+=['','## By dataset','','| Evaluator | Dataset | Reused | New records | Pending | Incompatible | Invalid | Baseline positive | New flags |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for job,s in summaries.items():
        if job.startswith('ppl_'):continue
        for d,x in s['datasets'].items():lines.append(f'| {job} | {d} | {x["historical_reused_records"]} | {x["new_records"]} | {x["pending_records"]} | {x["input_incompatible"]} | {x["call_or_parse_failed"]} | {x["baseline_positive"]} | {x["new_flips"]} |')
    lines+=['','## Perplexity','','| Model | Copies | Dataset | Scored | Decreased | Drop of at least 10% | Median relative drop |','|---|---:|---|---:|---:|---:|---:|']
    for job in ['ppl_gpt2','ppl_llama2']:
        for n,x in summaries[job]['strengths'].items():
            for d,v in [('all',x['all']),*x['datasets'].items()]:
                med=f'{100*v["median_relative_ppl_drop"]:.2f}%' if v['median_relative_ppl_drop'] is not None else '—'
                lines.append(f'| {job} | {n} | {d} | {v["completed"]} | {v["ppl_decreased"]} | {v["effect_at_least_10_percent"]} | {med} |')
    lanes={p.stem:read(p) for p in (OUT/'control').glob('*.status.json')}
    complete=all(s['pending_records']==0 for s in summaries.values())
    payload={'updated_unix':time.time(),'complete_all_records':complete,'sample_count':len(source),'datasets':dict(ds_counts),
             'jobs':summaries,'lanes':lanes,'manual_review':'inherited, no new full review'}
    write(OUT/'summary.json',payload);write(OUT/'failure_list.json',failures);write(OUT/'reuse_provenance.json',provenance)
    # Full text and native outputs are kept in this machine-readable export.
    temp=OUT/'all_records.jsonl.tmp'
    with temp.open('w') as f:
        for x in all_rows:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(OUT/'all_records.jsonl')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'covered_records':len(all_rows),'complete_all_records':complete}),flush=True)


if __name__=='__main__':
    try:
        with inc.lock(OUT/'.report.lock'):main()
    except BlockingIOError:print('Another atomic report writer is active',flush=True)
