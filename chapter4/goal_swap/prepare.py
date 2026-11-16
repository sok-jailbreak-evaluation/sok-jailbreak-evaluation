"""Freeze complete public replies and deterministic mismatched harmful goals."""
from __future__ import annotations
from collections import Counter
import math
import csv
import random
import re
import pyarrow.parquet as pq
from study_common import ROOT, REPO, CONFIG, read, write, sha

STOP = set('a an the to of in on for how can i is are and or with by from as you your it this that be do my me what please'.split())
FICTION = r'fictional|hypothetical|imaginary|novel|imagine|pretend|role.?play|story|Santa|Harry Potter|Hogwarts|Katniss|Batman|Superman|Vader|Frodo|Homer Simpson|Tyrion|Mickey|Gandalf|Tony Stark|Sherlock|James Bond|Edward Cullen|Hermione|Wakanda|SpongeBob|Bruce Wayne|Iron Man'
TOPICS = {
 'cyber': r'password|network|cyber|malware|ransom|hack|encrypt|decrypt|authentication|software|computer|SQL|firewall|login',
 'privacy': r'personal information|social security|address|phone number|contact information|medical histor|patient|privacy|private information',
 'finance': r'bank|credit card|financial|money|fraud|tax|invoice|account number|insider trad',
 'physical': r'weapon|explosive|bomb|firearm|chemical|firework|synthes|poison|assault',
 'health': r'medical|health|drug|disease|treatment|vaccine|doctor',
 'social': r'stereotyp|racis|sexist|propaganda|disinformation|misinformation|election|religio|hate',
 'copyright': r'copyright|magazine|movie|song|lyrics|novel|painting|book',
}

def topic(row):
    text = row['prompt']
    category = row.get('subcategory', '')
    if category == 'cyberattack': return 'cyber'
    if category == 'private_information_individual': return 'privacy'
    counts = {k: len(re.findall(pattern,text,re.I)) for k,pattern in TOPICS.items()}
    return max(counts, key=counts.get) if max(counts.values()) else 'misc'

def tokens(text):
    return Counter(w for w in re.findall(r'[a-z]{3,}', text.lower()) if w not in STOP)

def main():
    target = ROOT/'data/responses.json'
    if target.exists():
        print('Frozen data already exists; preserved', target); return
    config = read(CONFIG)
    source = REPO/'wildguard_test.parquet'
    raw = pq.read_table(source).to_pylist()
    labels = {'harmful': [], 'unharmful': []}
    seen = set()
    for index, row in enumerate(raw):
        response = row.get('response'); label = row.get('response_harm_label')
        if label not in labels or not isinstance(response,str) or not row.get('prompt'): continue
        if not config['local']['minimum_response_words'] <= len(response.split()) <= config['local']['maximum_response_words']: continue
        text_id = sha(response)
        if text_id in seen: continue
        seen.add(text_id)
        labels[label].append({'sample_id': f'wgt-row-{index:04d}', 'source_row': index,
          'source': 'allenai/wildguardmix/wildguardtest, existing local parquet',
          'prompt': row['prompt'], 'response': response, 'response_sha256': text_id,
          'dataset_response_label': label, 'dataset_prompt_label': row['prompt_harm_label'],
          'subcategory': row['subcategory'], 'topic': topic(row)})
    rng = random.Random(config['seed'])
    for pool in labels.values(): rng.shuffle(pool)
    selected = [r for pair in zip(labels['unharmful'][:100],labels['harmful'][:100]) for r in pair]
    assert len(selected) == 200
    benchmark=list(csv.DictReader((ROOT/'data/strongreject_dataset.csv').open()))
    goals=[{'prompt':row['forbidden_prompt'],'source_row':index,
       'source':'StrongREJECT author-released 313 prompts','category':row['category'],
       'topic':topic({'prompt':row['forbidden_prompt']})} for index,row in enumerate(benchmark)]
    vectors = [tokens(g['prompt']) for g in goals]
    df = Counter(w for vector in vectors for w in vector)
    def vec(text):
        v = tokens(text)
        v = {w: n*math.log(1+len(goals)/(1+df[w])) for w,n in v.items()}
        norm = math.sqrt(sum(x*x for x in v.values())) or 1
        return {w:x/norm for w,x in v.items()}
    normalized = [vec(g['prompt']) for g in goals]
    for row in selected:
        original = vec(row['prompt'])
        ordered = sorted(range(len(goals)), key=lambda i:(-sum(v*normalized[i].get(w,0) for w,v in original.items()),i))
        legal = [i for i in ordered if goals[i]['prompt'] != row['prompt']]
        same = [i for i in legal if goals[i]['topic'] == row['topic']][:3]
        if len(same) < 3: same += [i for i in legal if i not in same][:3-len(same)]
        cross = [i for i in reversed(legal) if goals[i]['topic'] != row['topic'] and i not in same][:2]
        assert len(same)==3 and len(cross)==2
        row['pairs'] = [{'pair_type': 'same_topic' if i in same and goals[i]['topic']==row['topic'] else
                        'nearest_topic_fallback' if i in same else 'cross_topic',
          'goal': goals[i]['prompt'], 'goal_source_row': goals[i]['source_row'], 'goal_topic': goals[i]['topic'],
          'mismatch_hypothesis': 'Original response answers a different request; whether any substantive assistance transfers must be checked against the entire response.'}
          for i in same+cross]
    # Objective technical/privacy goals from the paper's curated dataset.
    # False-prose generation tasks are unsuitable for "false facts => no help".
    eligible=[g for g in goals if g['category']=='Non-violent crimes'
      and re.search(r'malware|trojan|rootkit|virus|computer|software|database|SQL|information|personal|private|account|financial|records|credit card|phone|SIM|cryptomin|digital|hijack|service|WiFi|Wi-Fi|data|bank',g['prompt'],re.I)
      and not re.search(r'fake|false|phish|scam|catfish|harass|misinform|rumor|diploma|stock|insider|tax|charity',g['prompt'],re.I)]
    rng.shuffle(eligible)
    source_material=labels['harmful']+labels['unharmful']
    surface=[]
    for g in eligible[:25]:
        query=vec(g['prompt'])
        material=max(source_material,key=lambda r:sum(v*vec(r['prompt']).get(w,0) for w,v in query.items()))
        surface.append({**g,'sample_id':f'sr-goal-{g["source_row"]:04d}',
          'response':material['response'],'degradation_source_prompt':material['prompt'],
          'degradation_source_sample_id':material['sample_id'],'degradation_source_response_label':material['dataset_response_label']})
    assert len(surface)==25
    write(target,selected)
    write(ROOT/'data/surface_goals.json',surface)
    write(ROOT/'data/manifest.json',{'seed':42,'source_file':str(source),'source_sha256':sha(source.read_bytes()),
      'initial_response_count':50,'balanced_initial_labels':{'harmful':25,'unharmful':25},
      'maximum_response_count':200,'surface_goals':25,'no_evaluator_scores_used_for_sampling':True,
      'response_word_range':[60,550],'pairing':'3 same inferred-topic TF-IDF neighbours, 2 least-similar cross-topic harmful prompts; explicit nearest-topic fallback retained',
      'frozen_responses_sha256':sha(target.read_bytes()),'surface_goals_sha256':sha((ROOT/'data/surface_goals.json').read_bytes()),
      'mismatched_goals_source':read(ROOT/'data/strongreject_source.json'),
      'same_topic_is_a_retrieval_hypothesis':'not a label of non-assistance; full response semantic verification required'})
    print('Frozen',len(selected),'replies; first 50 balanced; 25 surface goals')

if __name__=='__main__': main()
