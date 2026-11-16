"""Locate a response suffix using structural character spans and verified token IDs."""
from __future__ import annotations

from dataclasses import dataclass

class InvalidControl(ValueError):
    pass


@dataclass
class ControlLayout:
    text: str
    input_ids: list[int]
    control_start: int
    control_stop: int
    prefix_ids: list[int]
    tail_ids: list[int]
    control_ids: list[int]
    suffix: str
    character_start: int
    character_stop: int
    response_start: int
    original_response_stop: int


def build_layout(adapter, sample: dict, suffix: str, expected_length: int | None = None) -> ControlLayout:
    if getattr(adapter, 'input_format', None) == 'text_pair':
        return _build_text_pair_layout(adapter, sample, suffix, expected_length)
    prefix, tail, response_start = adapter.parts(sample)
    text = prefix + suffix + tail
    start, stop = len(prefix), len(prefix) + len(suffix)
    encoded = adapter.tokenizer(text, truncation=False, return_offsets_mapping=True,
                                add_special_tokens=getattr(adapter, 'add_special_tokens', True))
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    if len(ids) + adapter.output_reserve > adapter.context_limit:
        raise InvalidControl(f'Complete input exceeds {adapter.name} context: {len(ids)} tokens')
    overlapping = [i for i, (a, b) in enumerate(offsets) if b > start and a < stop]
    if not overlapping or overlapping != list(range(overlapping[0], overlapping[-1] + 1)):
        raise InvalidControl('Suffix token region is empty or noncontiguous')
    if any(offsets[i][0] < start or offsets[i][1] > stop for i in overlapping):
        raise InvalidControl('A token overlaps fixed text and suffix')
    isolated = adapter.tokenizer.encode(suffix, add_special_tokens=False)
    first, last = overlapping[0], overlapping[-1] + 1
    if ids[first:last] != isolated:
        raise InvalidControl('Standalone and in-context suffix token IDs differ')
    if expected_length is not None and len(isolated) != expected_length:
        raise InvalidControl('Suffix does not have the required token length')
    if any(token in adapter.tokenizer.all_special_ids for token in isolated):
        raise InvalidControl('Suffix contains a special token')
    if text[response_start:start] != sample['response'] or text[start:stop] != suffix:
        raise AssertionError('Original response or suffix field changed')
    if text[stop:] != tail:
        raise AssertionError('Fixed input integrity failure')
    return ControlLayout(text, ids, first, last, ids[:first], ids[last:], isolated, suffix,
                         start, stop, response_start, start)


def _build_text_pair_layout(adapter, sample: dict, suffix: str,
                            expected_length: int | None) -> ControlLayout:
    """Locate suffix IDs in the second field of an official tokenizer text pair."""
    prompt, response = sample['prompt'], sample['response']
    full_response = response + suffix
    encoded = adapter.tokenizer(prompt, full_response, add_special_tokens=True,
                                truncation=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    sequence_ids = encoded.sequence_ids()
    if len(ids) + adapter.output_reserve > adapter.context_limit:
        raise InvalidControl(f'Complete input exceeds {adapter.name} context: {len(ids)} tokens')
    start, stop = len(response), len(full_response)
    overlapping = [i for i, (a, b) in enumerate(offsets)
                   if sequence_ids[i] == 1 and b > start and a < stop]
    if not overlapping or overlapping != list(range(overlapping[0], overlapping[-1] + 1)):
        raise InvalidControl('Second-field suffix token region is empty or noncontiguous')
    if any(offsets[i][0] < start or offsets[i][1] > stop for i in overlapping):
        raise InvalidControl('A token overlaps fixed response text and suffix')
    isolated = adapter.tokenizer.encode(suffix, add_special_tokens=False)
    first, last = overlapping[0], overlapping[-1] + 1
    if ids[first:last] != isolated:
        raise InvalidControl('Standalone and in-context suffix token IDs differ')
    if expected_length is not None and len(isolated) != expected_length:
        raise InvalidControl('Suffix does not have the required token length')
    if any(token in adapter.tokenizer.all_special_ids for token in isolated):
        raise InvalidControl('Suffix contains a special token')
    if full_response[:start] != response or full_response[start:stop] != suffix:
        raise AssertionError('Original response or suffix field changed')
    return ControlLayout(full_response, ids, first, last, ids[:first], ids[last:], isolated,
                         suffix, start, stop, 0, start)


def initial_suffix(adapters, sample: dict, length: int) -> str:
    choices = [' !' * length, ' x' * length,
               '\n' + ' !' * (length - 1), '\n!' + ' !' * (length - 2)]
    for suffix in choices:
        try:
            for i, adapter in enumerate(adapters):
                build_layout(adapter, sample, suffix, length if i == 0 else None)
            return suffix
        except InvalidControl:
            continue
    raise InvalidControl('No canonical initial suffix layout for all evaluators')


def filter_candidates(adapter, sample: dict, layout: ControlLayout, controls: list[list[int]]) -> tuple[list[str], dict]:
    candidates, seen = [], {layout.suffix}
    counts = {'proposed': len(controls), 'roundtrip_or_boundary_rejected': 0, 'duplicates_or_current': 0}
    for ids in controls:
        raw = adapter.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # Llama tokenizers may drop a leading space while decoding. Keep it only
        # when both the token-ID roundtrip and the full context verify exactly.
        decoded_choices = [raw, ' ' + raw] if not raw.startswith(' ') else [raw]
        chosen = None
        for text in decoded_choices:
            if adapter.tokenizer.encode(text, add_special_tokens=False) != ids:
                continue
            try:
                candidate = build_layout(adapter, sample, text, len(layout.control_ids))
            except InvalidControl:
                continue
            if candidate.prefix_ids != layout.prefix_ids or candidate.tail_ids != layout.tail_ids or candidate.control_ids != ids:
                continue
            chosen = text
            break
        if chosen is None:
            counts['roundtrip_or_boundary_rejected'] += 1
        elif chosen in seen:
            counts['duplicates_or_current'] += 1
        else:
            seen.add(chosen)
            candidates.append(chosen)
    counts['valid_unique'] = len(candidates)
    return candidates, counts
