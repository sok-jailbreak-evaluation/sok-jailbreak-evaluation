"""Incremental trajectory and atomic minimal search state for optimized runs."""
from __future__ import annotations

import json
import os
from pathlib import Path

from common import read_json, read_jsonl, write_json, write_jsonl


class IncrementalCheckpoint:
    def __init__(self, state_path: Path, run_id: str):
        self.state_path = state_path
        self.trace_path = state_path.with_suffix('.steps.jsonl')
        self.run_id = run_id
        self.count = 0

    def load(self) -> dict | None:
        if not self.state_path.exists():
            return None
        state = read_json(self.state_path)
        if state.get('run_id') != self.run_id:
            raise ValueError('Checkpoint version does not match this run')
        if 'step_count' not in state:
            self.count = len(state['steps'])
            return state
        count = int(state['step_count'])
        trace = read_jsonl(self.trace_path) if self.trace_path.exists() else []
        if len(trace) < count:
            raise ValueError('Checkpoint trajectory is incomplete')
        if len(trace) > count:
            write_jsonl(self.trace_path, trace[:count])
            trace = trace[:count]
        state['steps'] = trace
        self.count = count
        return state

    def save(self, state: dict) -> None:
        steps = state['steps']
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        if len(steps) == self.count + 1:
            with self.trace_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(steps[-1], ensure_ascii=False,
                                        allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        elif len(steps) == self.count and self.count and read_jsonl(self.trace_path) != steps:
            write_jsonl(self.trace_path, steps)
        elif len(steps) != self.count:
            # A reference fallback may restart a step after a failed optimized path.
            write_jsonl(self.trace_path, steps)
        compact = {key: value for key, value in state.items() if key != 'steps'}
        compact['run_id'] = self.run_id
        compact['step_count'] = len(steps)
        if 'current_suffix' not in compact and 'best_suffix' in compact:
            compact['current_suffix'] = compact['best_suffix']
        if 'best' not in compact and 'final' in compact:
            compact['best'] = compact['final']
        write_json(self.state_path, compact)
        self.count = len(steps)
