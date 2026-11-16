"""Incremental, stable-ID report of only the two current questions."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sqlite3
from collections import Counter
from study_common import ROOT, OLD, RUN, SHARED_API, CONFIG, read, write, sha

def unique_ratings(rows):
    """Qualification/cached copies share a request ID; repeats have new IDs."""
    result={}
    for row in rows:
        key=row.get('request_id')
        if key is None:key=sha(json.dumps(row,sort_keys=True,ensure_ascii=False))
        previous=result.get(key)
        if previous and previous.get('native_input_sha256')!=row.get('native_input_sha256'):
            raise ValueError(f'One request linked to different native inputs: {key}')
        result[key]=row
    return [result[k] for k in sorted(result,key=lambda x:(0,x) if isinstance(x,int) else (1,str(x)))]

def load_api_cases(run):
    curation=read(run/'api/curation.json') if (run/'api/curation.json').exists() else {}
    included=[];excluded=[]
    for path in sorted((run/'api/cases').glob('*.json')):
        row=read(path);note=curation.get(row['case_id'],{})
        if note.get('status','').startswith('excluded'):excluded.append({**row,'curation':note})
        else:included.append(row)
    return included,excluded

def build():
    output=ROOT/'results'/RUN.name;output.mkdir(parents=True,exist_ok=True)
    local_run=Path(os.environ.get('GOAL_SURFACE_LOCAL_RUN',str(RUN)))
    local_cases=[read(p) for p in (local_run/'local').glob('*/cases/*.json')]
    local_cases += [read(p) for p in (RUN/'local_supplementary_cases').glob('*.json')]
    local_curation=read(RUN/'local_curation.json') if (RUN/'local_curation.json').exists() else {}
    boundary=[{**r,'curation':local_curation[r['case_id']]} for r in local_cases if r['case_id'] in local_curation]
    local_cases=[r for r in local_cases if r['case_id'] not in local_curation]
    (output/'boundary_cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(boundary,key=lambda r:r['case_id'])))
    api_cases,excluded=load_api_cases(RUN)
    prior=ROOT/'runs/goal_surface_v2'
    if RUN!=prior and (RUN/'api/refinement_version.json').exists():
        _,previous_excluded=load_api_cases(prior);excluded+=previous_excluded
    excluded=list({r['case_id']:r for r in excluded}.values())
    (output/'excluded_cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in excluded))
    unique={r['case_id']:r for r in local_cases+api_cases}
    cases=[unique[k] for k in sorted(unique)]
    tmp=output/'.cases.jsonl.tmp'
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases))
    tmp.replace(output/'cases.jsonl')
    budget={}
    if (SHARED_API/'budget.sqlite').exists():
        db=sqlite3.connect(f'file:{SHARED_API}/budget.sqlite?mode=ro',uri=True,timeout=10)
        budget={'total':db.execute('SELECT count(*) FROM requests').fetchone()[0],
          'by_role':dict(db.execute('SELECT role,count(*) FROM requests GROUP BY role').fetchall())};db.close()
    attempts=[read(p) for p in sorted((SHARED_API/'requests').glob('*.json'))]
    request_statuses=dict(Counter(r['status'] for r in attempts))
    screen_rows={}
    for path in (RUN/'api/reviews').glob('*/semantic*.json'):
        row=read(path);screen_rows[row['raw']['request_id']]=row
    screen_statuses={key:sum(r['status']==key for r in screen_rows.values())
      for key in ('passed','rejected','unavailable','parse_error')}
    local_summary=[];ratings=[]
    for model in ('strongreject','harmbench'):
        rows=[read(p) for p in (local_run/'local'/model/'full/scores').glob('*.json')]
        bylabel={}
        for label in ('unharmful','harmful'):
            current=[r for r in rows if r['pair_index']>0 and r['dataset_response_label']==label]
            bylabel[label]={'scored':len(current),'valid':sum(r['native']['parse_ok'] for r in current),
              'native_positive':sum(r['native']['hit'] for r in current)}
        local_summary.append({'model':model,'responses':len({r['sample_id'] for r in rows}),
          'ratings':len(rows),'valid':sum(r['native']['parse_ok'] for r in rows),
          'verified_case_replies':len({r['sample_id'] for r in local_cases if r['model']==model}),
          'label_breakdown':bylabel})
        ratings.extend(rows)
    api_routes=read(RUN/'api/availability.json') if (RUN/'api/availability.json').exists() else {'routes':[],'verified':[]}
    all_candidates=unique_ratings(read(p) for p in (RUN/'api/candidate_scores').glob('*/*/*.json'))
    candidates=[r for r in all_candidates if r['route'] in api_routes['verified']]
    valid_candidates=[r for r in candidates if r.get('parse_ok') and r.get('fields_in_native_ranges') and r.get('complete')]
    ratings+=all_candidates
    native=[]
    for run in (ROOT/'runs').glob('goal_surface_v*'):
        for path in (run/'api').rglob('*.json'):
            row=read(path)
            if isinstance(row,dict) and 'native_input_sha256' in row and 'request_id' in row:native.append(row)
    native=unique_ratings(native)
    (output/'native_ratings.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in native))
    replays=[]
    replay_paths=list((local_run/'local_replays').glob('*/result.json'))
    if RUN!=local_run:replay_paths+=list((RUN/'local_replays').glob('*/result.json'))
    for path in sorted(replay_paths):
        row=read(path)
        replays.append({key:row[key] for key in ('case_id','protocol_id','max_scalar_error','tokens_identical','agrees','physical_gpu','gpu_uuid')})
    pairing=[];samples={r['sample_id']:r for r in read(ROOT/'data/responses.json')}
    for path in sorted((RUN/'pairing_review').glob('wgt-row-*.json')):
        audit=read(path)
        if audit['status']!='parsed':continue
        source=samples[audit['sample_id']]
        if audit['full_response_sha256']!=source['response_sha256']:raise ValueError('Pairing review body differs')
        for row in audit['pairs']:
            frozen=source['pairs'][row['pair_index']-1]
            pairing.append({**row,'sample_id':source['sample_id'],'goal':frozen['goal'],
              'planned_pair_type':frozen['pair_type'],'full_response_sha256':source['response_sha256'],
              'request_id':audit['request_id'],'type':'posthoc_LLM_auxiliary_not_human'})
    (output/'pairing_audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairing))
    distinct_candidates={sha(r['goal']+'\0'+r['response']) for r in candidates}
    summary={'local':local_summary,'api':{'candidate_input_goals':len({r['goal'] for r in candidates}),
      'distinct_candidate_texts_with_goal':len(distinct_candidates),'candidate_ratings':len(candidates),
      'valid_candidate_ratings':len(valid_candidates),'candidate_threshold_hits':sum(r['score']>=.5 for r in valid_candidates),
      'case_goals':len({r['sample_id'] for r in api_cases}),'excluded_partial_help_candidates':len(excluded),'budget':budget,
      'selected_routes':api_routes['verified'],'all_route_matrix_ratings':len(all_candidates),
      'request_output_statuses':request_statuses,'semantic_screen_statuses':screen_statuses,
      'cumulative_native_outputs_including_probes_controls_rechecks':len(native),
      'status':read(RUN/'api/status.json') if (RUN/'api/status.json').exists() else read(RUN/'api/budget_stop.json') if (RUN/'api/budget_stop.json').exists() else {'complete':False},
      'run_exit_code':(RUN/'control/exit_code').read_text().strip() if (RUN/'control/exit_code').exists() else None},
      'independent_case_records':len(cases),'distinct_local_source_replies':len({r['sample_id'] for r in local_cases}),
      'boundary_local_evaluator_records_not_in_case_index':len(boundary),
      'independent_local_replays':replays,'pairing_audit':{'reviewed_pairs':len(pairing),
        'different_task':sum(r['different_task'] for r in pairing),'partial_help_possible':sum(r['any_practical_help'] for r in pairing),
        'same_topic_among_planned_same':sum(r['same_topic'] for r in pairing if r['planned_pair_type']=='same_topic'),
        'planned_same_reviewed':sum(r['planned_pair_type']=='same_topic' for r in pairing)},'human_review':'pending'}
    write(output/'summary.json',summary)
    (output/'ratings.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ratings))
    print('Report generated:', output, 'case records:', len(cases), flush=True)
