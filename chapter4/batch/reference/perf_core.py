"""Optional, semantics-preserving execution helpers for evaluator search.

The search policy lives in search.py.  This module only carries already
validated inputs and schedules independent native scoring tasks.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Protocol, runtime_checkable

from layout import ControlLayout, InvalidControl, build_layout


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Rendering and native interpretation of an evaluator input."""

    name: str

    def render(self, sample: dict, suffix: str) -> str: ...
    def preprocess(self, sample: dict, suffix: str) -> list[int]: ...
    def generation_config(self) -> dict: ...
    def parse(self, raw_output: str) -> dict: ...
    def aggregate(self, parsed: dict) -> float: ...


@runtime_checkable
class ProposalBackend(Protocol):
    def gradient(self, layout: ControlLayout, direction: str): ...
    def disallowed_tokens(self, allow_non_ascii: bool): ...


@runtime_checkable
class NativeJudgeBackend(Protocol):
    def score_suffixes(self, sample: dict, suffixes: list[str], direction: str,
                       batch_size: int) -> list[dict]: ...
    def native_check(self, sample: dict, suffix: str, direction: str,
                     tolerance: float) -> dict: ...


@dataclass
class ValidatedCandidate:
    candidate_id: int
    text: str
    layouts: dict[str, ControlLayout] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreTask:
    candidate_id: int
    evaluator: str
    direction: str
    layout: ControlLayout





def require_versioned_output(output: Path, config: dict, data: Path,
                             code_root: Path) -> str | None:
    if not config.get('performance'):
        return None
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'version.json'
    if not manifest.exists():
        manifest.write_text(json.dumps({'config': config, 'data': str(data)}, indent=2) + '\n')
    return output.name


def filter_with_layouts(adapter, sample: dict, anchor: ControlLayout,
                        controls: list[list[int]]) -> tuple[list[ValidatedCandidate], dict]:
    """Match layout.filter_candidates, retaining the successful full validation."""
    result: list[ValidatedCandidate] = []
    seen = {anchor.suffix}
    counts = {'proposed': len(controls), 'roundtrip_or_boundary_rejected': 0,
              'duplicates_or_current': 0}
    for ids in controls:
        raw = adapter.tokenizer.decode(ids, skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
        choices = [raw, ' ' + raw] if not raw.startswith(' ') else [raw]
        chosen = None
        for text in choices:
            if adapter.tokenizer.encode(text, add_special_tokens=False) != ids:
                continue
            try:
                layout = build_layout(adapter, sample, text, len(anchor.control_ids))
            except InvalidControl:
                continue
            if (layout.prefix_ids != anchor.prefix_ids or
                    layout.tail_ids != anchor.tail_ids or layout.control_ids != ids):
                continue
            chosen = (text, layout)
            break
        if chosen is None:
            counts['roundtrip_or_boundary_rejected'] += 1
        elif chosen[0] in seen:
            counts['duplicates_or_current'] += 1
        else:
            seen.add(chosen[0])
            result.append(ValidatedCandidate(len(result), chosen[0],
                                             {adapter.name: chosen[1]}))
    counts['valid_unique'] = len(result)
    return result, counts


def validate_all(candidates: list[ValidatedCandidate], adapters: dict,
                 sample: dict, length: int) -> list[ValidatedCandidate]:
    valid = []
    for candidate in candidates:
        try:
            for index, (name, adapter) in enumerate(adapters.items()):
                layout = candidate.layouts.get(name)
                if layout is None:
                    layout = build_layout(adapter, sample, candidate.text,
                                          length if index == 0 else None)
                    candidate.layouts[name] = layout
                elif index == 0 and len(layout.control_ids) != length:
                    raise InvalidControl('Suffix does not have the required token length')
            candidate.candidate_id = len(valid)
            valid.append(candidate)
        except InvalidControl:
            continue
    return valid


def score_tasks(adapters: dict, sample: dict, candidates: list[ValidatedCandidate],
                directions: dict, batch_size: int, asynchronous: bool = False,
                score_layouts: bool = True, cache: dict | None = None
                ) -> tuple[dict[str, list[dict]], dict]:
    """Score all tasks, then restore original candidate order by candidate_id.

    Independent evaluators can run concurrently.  No search state is updated
    until all futures have completed successfully.
    """
    names = list(adapters)
    tasks = [ScoreTask(c.candidate_id, name, directions[name], c.layouts[name])
             for name in names for c in candidates]
    grouped = {name: [t for t in tasks if t.evaluator == name] for name in names}
    started = time.perf_counter()
    cache_hits = 0

    def run(name):
        nonlocal cache_hits
        adapter = adapters[name]
        selected = grouped[name]
        keys = [(name, directions[name], tuple(task.layout.input_ids)) for task in selected]
        missing = [task for task, key in zip(selected, keys)
                   if cache is None or key not in cache]
        cache_hits += len(selected) - len(missing)
        if not missing:
            return {task.candidate_id: cache[key] for task, key in zip(selected, keys)}
        if score_layouts and hasattr(adapter, 'score_layouts'):
            rows = adapter.score_layouts([t.layout for t in missing],
                                         directions[name], batch_size)
        else:
            rows = adapter.score_suffixes(sample, [t.layout.suffix for t in missing],
                                          directions[name], batch_size)
        if len(rows) != len(missing):
            raise AssertionError('Native judge omitted a scoring task')
        fresh = {task.candidate_id: row for task, row in zip(missing, rows)}
        if cache is not None:
            for task, row in zip(missing, rows):
                cache[(name, directions[name], tuple(task.layout.input_ids))] = row
        return {task.candidate_id: (fresh[task.candidate_id]
                                    if task.candidate_id in fresh else cache[key])
                for task, key in zip(selected, keys)}

    if asynchronous and len(names) > 1:
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = {name: pool.submit(run, name) for name in names}
            scored = {name: futures[name].result() for name in names}
    else:
        scored = {name: run(name) for name in names}
    ordered = {name: [scored[name][i] for i in range(len(candidates))] for name in names}
    return ordered, {'score_tasks': len(tasks),
                     'native_score_seconds': time.perf_counter() - started,
                     'deterministic_cache_hits': cache_hits,
                     'async_evaluators': asynchronous and len(names) > 1}
