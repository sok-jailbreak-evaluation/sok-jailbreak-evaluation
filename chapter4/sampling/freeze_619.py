"""Extend the exact frozen 300 negatives to WG100 + all HB329 + all JBB190."""
import csv
import json
from collections import Counter
from pathlib import Path
from build import ROOT, REPO, read, digest, base_record, majority, jbb_majority, as_json, as_jsonl, frozen_write


def main():
    out=ROOT/'data_619'
    old_path=ROOT/'data_jbb600/unharmful_up.jsonl'
    original=[json.loads(x) for x in old_path.read_text().split('\n') if x]
    assert len(original)==300 and all(x['sampling_group']=='unharmful' for x in original)
    old={x['sample_id']:x for x in original}
    cfg=read(ROOT/'config.json')
    main_meta=ROOT/'sources/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv'
    metadata={x['BehaviorID']:x for x in csv.DictReader(main_meta.open())}
    recovery_path=ROOT/'sources/harmbench_historical_metadata/recovered_metadata.json'
    recovery=read(recovery_path)
    historical={x['BehaviorID']:x for x in recovery['behaviors']}
    hb_path=ROOT/'sources/harmbench/data/classifier_val_sets/text_behaviors_val_set.json'
    source=read(hb_path);hb=[];raw_index=-1;restored=[]
    for behavior_id,items in source.items():
        for index,item in enumerate(items):
            raw_index+=1
            if majority(item):continue
            sid=f'harmbench_response:{behavior_id}:{index}'
            if sid in old:
                x=old[sid]
                assert x['response']==item['generation'] and x['original_record']==item
                assert x['prompt']==metadata[behavior_id]['Behavior'] and x.get('context','')==metadata[behavior_id]['ContextString']
                hb.append(x);continue
            meta=metadata.get(behavior_id,historical.get(behavior_id));assert meta is not None
            recovered=behavior_id not in metadata
            semantic=meta.get('SemanticCategory',meta.get('Category','not_provided'))
            functional=meta.get('FunctionalCategory','not_provided_in_historical_metadata')
            x=base_record('harmbench_response',f'{behavior_id}:{index}',meta['Behavior'],item['generation'],
                          'unharmful',[semantic,functional],meta.get('ContextString',''))
            x.update(behavior_id=behavior_id,item_index=index,source_row_index=raw_index,
                test_case=item['test_case'],test_case_sha256=digest(item['test_case']),
                source_dataset='centerforaisafety/HarmBench',source_split='classifier_val_sets/text_behaviors_val_set',
                source_revision=cfg['harmbench_revision'],semantic_category=semantic,functional_category=functional,
                model=item['model'],method=item['method'],label_scope='goal_conditioned_harmful_behavior_success',
                label_source='three_human_votes_majority',native_label=0,response_harm_label=None,
                global_response_harmfulness='not_separately_annotated',human_votes=[item[f'human_{i}'] for i in range(3)],
                behavior_metadata=meta,original_record=item,
                behavior_metadata_provenance={'revision':recovery['revision'] if recovered else cfg['harmbench_revision'],
                    'path':recovery['upstream_path'] if recovered else 'data/behavior_datasets/harmbench_behaviors_text_all.csv',
                    'recovered_from_official_history':recovered})
            hb.append(x)
            if recovered:restored.append(sid)
    assert len(hb)==329 and len(restored)==3
    jbb_path=ROOT/'sources/jailbreakbench/data/judge-comparison.csv';jbb=[]
    for index,item in enumerate(csv.DictReader(jbb_path.open())):
        if jbb_majority(item):continue
        sid=f'jailbreakbench_response:row-{index:04d}'
        if sid in old:
            x=old[sid];assert x['response']==item['target_response'] and x['prompt']==item['goal'] and x['original_record']==item
        else:
            x=base_record('jailbreakbench_response',f'row-{index:04d}',item['goal'],item['target_response'],
                          'unharmful',['no_official_category'])
            x.update(source_row_index=index,original_index=item['Index'],test_case=item['prompt'],test_case_sha256=digest(item['prompt']),
                source_dataset='JailbreakBench/JBB-Behaviors',source_configuration='judge_comparison',source_split='test',
                source_revision=cfg['jailbreakbench_revision'],native_label=0,human_votes=[int(item[f'human{i}']) for i in range(1,4)],
                native_human_majority=int(item['human_majority']),label_source='three_human_votes_verified_published_majority',
                label_scope='goal_conditioned_jailbreak_success',response_harm_label=None,
                global_response_harmfulness='not_separately_annotated',category=None,
                category_source='not_provided_in_judge_comparison',original_record=item)
        jbb.append(x)
    assert len(jbb)==190
    added=[x for x in hb+jbb if x['sample_id'] not in old]
    assert len(added)==319 and Counter(x['dataset_id'] for x in added)=={'harmbench_response':229,'jailbreakbench_response':90}
    rows=original+added;assert len(rows)==len({x['sample_id'] for x in rows})==619
    assert Counter(x['dataset_id'] for x in rows)=={'wildguardtest':100,'harmbench_response':329,'jailbreakbench_response':190}
    mapping=[]
    for position,x in enumerate(rows):
        index=x.get('source_row_index')
        try:index=int(index) if index is not None else position
        except (ValueError,TypeError):index=position
        mapping.append({'sample_id':x['sample_id'],'dataset_id':x['dataset_id'],'position_619':position,
            'original_300_position':position if position<300 else None,'is_new_319':position>=300,
            'source_row_index':x.get('source_row_index'),'search_seed_index':index,'evaluator_index':0,
            'base_seed':42,'first_proposal_seed':42+index*100003,
            'candidate_seed_formula':'42 + search_seed_index*100003 + zero_based_step*1009; evaluator_index=0',
            'input_pair_sha256':x['input_pair_sha256']})
    # Preserve the exact bytes and order of the original queue as the first300.
    combined=old_path.read_bytes()+as_jsonl(added).encode()
    files={'unharmful_up.jsonl':combined,'original_300.jsonl':old_path.read_bytes(),
           'new_319.jsonl':as_jsonl(added).encode(),'mapping.jsonl':as_jsonl(mapping).encode()}
    for ds in ['wildguardtest','harmbench_response','jailbreakbench_response']:
        files[f'{ds}/unharmful.jsonl']=as_jsonl([x for x in rows if x['dataset_id']==ds]).encode()
    manifest={'schema_version':1,'total':619,'counts':dict(Counter(x['dataset_id'] for x in rows)),
        'original_300_count':300,'new_319_count':319,'new_counts':dict(Counter(x['dataset_id'] for x in added)),
        'old300_sha256':digest(old_path.read_bytes()),'data_sha256':digest(combined),
        'original300_retained_byte_for_byte':True,'original300_retained_order':True,
        'restored_missing_harmbench_negatives':restored,'recovery':recovery,
        'source_files':{str(p.relative_to(REPO)):digest(p.read_bytes()) for p in [old_path,hb_path,main_meta,jbb_path,recovery_path]},
        'files':{k:{'sha256':digest(v),'bytes':len(v)} for k,v in files.items()},
        'mapping':'mapping.jsonl','sampling':'WG retains exact original100; HB and JBB take all majority-negative records, no model score filtering',
        'label_scopes':{d:sorted({x['label_scope'] for x in rows if x['dataset_id']==d}) for d in {x['dataset_id'] for x in rows}},
        'rng':'old source_row_index or original frozen300 position retained; new source row index frozen; evaluator_index0',
        'positive_group_answers_inspected':False,'human_review':'inherited source labels; no new annotations'}
    for name,content in files.items():frozen_write(out/name,content)
    frozen_write(out/'manifest.json',as_json(manifest))
    print(json.dumps({'data':str((out/'unharmful_up.jsonl').resolve()),'manifest':str((out/'manifest.json').resolve()),
        'counts':manifest['counts'],'new_counts':manifest['new_counts'],'data_sha256':manifest['data_sha256'],
        'restored':restored,'mapping':str((out/'mapping.jsonl').resolve())},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
