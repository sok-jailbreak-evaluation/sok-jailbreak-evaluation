import json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CASES=ROOT.parent/'cases'
CONF=json.loads((ROOT/'config.json').read_text())



def read(path):
    return json.loads(Path(path).read_text())

def unit(text, rule):
    words = list(re.finditer(r'\S+', text))
    if not words: raise ValueError('Empty response')
    end = words[-1].end()
    if rule == 'tail_word': start = words[-1].start()
    elif rule == 'tail_four_words': start = words[max(0, len(words)-4)].start()
    elif rule == 'last_sentence':
        stripped = text[:end]
        boundaries = list(re.finditer(r'(?<=[.!?。！？])\s+', stripped))
        start = boundaries[-1].end() if boundaries else words[0].start()
    else: raise ValueError(rule)
    value = text[start:end]
    assert value and value in text
    return {'text': value, 'source_start_char': start, 'source_end_char': end}

def transform(text, rule, repetitions):
    if repetitions < 1: raise ValueError('At least one additional copy')
    part = unit(text, rule)
    modified = text + (CONF['separator']+part['text'])*repetitions
    assert modified[:len(text)] == text
    return modified, {**part, 'rule': rule, 'additional_copies': repetitions,
        'original_chars': len(text), 'modified_chars': len(modified),
        'added_chars': len(modified)-len(text), 'original_words': len(text.split()),
        'modified_words': len(modified.split()), 'unit_words': len(part['text'].split()),
        'original_is_exact_prefix': True, 'new_domain_facts': False,
        'quality_evidence': 'unchanged original body plus exact redundant copies of an existing tail span',
        'human_review': 'pending'}

def windows(length, context, stride):
    if length < 2 or not (1 <= stride < context): raise ValueError('Invalid scoring dimensions')
    previous_end = 1
    for start in range(0, length, stride):
        end = min(start+context, length)
        target_start = max(previous_end, start+1)
        if target_start < end:
            yield {'start': start, 'end': end, 'target_start': target_start, 'target_end': end}
        previous_end = end
        if end == length: break

def compare(before, after):
    relative = 1-after['ppl']/before['ppl']
    return {'ppl_decreased': after['ppl'] < before['ppl'], 'relative_ppl_drop': relative,
        'effect_at_least_10_percent': relative >= CONF['relative_ppl_drop_effect'],
        'mean_nll_delta': after['mean_nll']-before['mean_nll'],
        'safety_label': None}
