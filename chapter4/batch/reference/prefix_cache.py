"""Optional immutable-prefix KV reuse for causal candidate scoring.

Only verified suffix layouts enter this module. The cache contains tokens
strictly before the suffix; the suffix and all following template tokens are
recomputed for every candidate. Search gradients and final native checks keep
their existing full-forward paths.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import time

import torch


def _cache_nbytes(cache) -> int:
    if hasattr(cache, 'layers'):
        return sum(t.numel() * t.element_size()
                   for layer in cache.layers
                   for t in (getattr(layer, 'keys', None), getattr(layer, 'values', None))
                   if isinstance(t, torch.Tensor))
    if isinstance(cache, (tuple, list)):
        return sum(t.numel() * t.element_size() for layer in cache for t in layer
                   if isinstance(t, torch.Tensor))
    raise TypeError(f'Unsupported cache type: {type(cache).__name__}')


class PrefixKVScorer:
    def __init__(self, adapter, tolerance: float = 1e-5, max_entries: int = 4,
                 max_memory_mib: int = 512, microbatch_size: int | None = None,
                 loss_tolerance: float | None = None):
        if max_entries < 1 or max_memory_mib < 1:
            raise ValueError('Prefix KV cache limits must be positive')
        if microbatch_size is not None and microbatch_size < 1:
            raise ValueError('KV microbatch size must be positive')
        self.adapter = adapter
        self.tolerance = tolerance
        self.loss_tolerance = tolerance if loss_tolerance is None else loss_tolerance
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_mib * 1024**2
        self.microbatch_size = microbatch_size
        self.entries = OrderedDict()
        self.verified_prefixes = set()
        self.disabled_reason = None
        self.verified = False
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.prefill_forwards = 0
        self.score_forwards = 0
        self.prefill_seconds = 0.0
        self.score_seconds = 0.0
        self.cache_bytes_peak = 0
        self.max_verify_score_error = 0.0
        self.max_verify_loss_error = 0.0
        self.last_backend = 'not_used'

    def _key(self, prefix: list[int]) -> tuple:
        return (self.adapter.prefix_kv_identity(), tuple(prefix))

    def _get(self, prefix: list[int]):
        key = self._key(prefix)
        if not prefix:
            raise ValueError('No fixed prefix before suffix')
        if key in self.entries:
            self.cache_hits += 1
            self.entries.move_to_end(key)
            return self.entries[key], key
        self.cache_misses += 1
        ids = torch.tensor([prefix], dtype=torch.long, device=self.adapter.device)
        if ids.is_cuda:
            torch.cuda.synchronize(ids.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = self.adapter.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        use_cache=True, return_dict=True,
                                        logits_to_keep=1)
        if ids.is_cuda:
            torch.cuda.synchronize(ids.device)
        self.prefill_seconds += time.perf_counter() - started
        self.prefill_forwards += 1
        cache = output.past_key_values
        if cache is None or cache.get_seq_length() != len(prefix):
            raise ValueError('Model returned missing or incomplete prefix KV')
        size = _cache_nbytes(cache)
        if size > self.max_memory_bytes:
            raise MemoryError(f'Prefix KV requires {size} bytes; limit is '
                              f'{self.max_memory_bytes} bytes')
        self.entries[key] = cache
        while (len(self.entries) > self.max_entries or
               sum(_cache_nbytes(value) for value in self.entries.values()) >
               self.max_memory_bytes):
            old_key, _ = self.entries.popitem(last=False)
            self.verified_prefixes.discard(old_key)
            self.cache_evictions += 1
        self.cache_bytes_peak = max(self.cache_bytes_peak,
                                    sum(_cache_nbytes(value) for value in self.entries.values()))
        return cache, key

    @staticmethod
    def _repeat(cache, batch: int):
        # DynamicLayer appends with torch.cat: distinct layer objects can share
        # immutable prefix tensor storage without mutating the saved cache.
        try:
            from transformers import DynamicCache
            from transformers.cache_utils import DynamicLayer
            if type(cache) is DynamicCache and all(
                    type(layer) is DynamicLayer for layer in cache.layers):
                result = copy.copy(cache)
                result.layers = []
                for layer in cache.layers:
                    replica = copy.copy(layer)
                    replica.keys = layer.keys.expand(batch, *layer.keys.shape[1:])
                    replica.values = layer.values.expand(batch, *layer.values.shape[1:])
                    result.layers.append(replica)
                return result
        except (ImportError, AttributeError):
            pass
        result = copy.deepcopy(cache)
        if batch != 1:
            if hasattr(result, 'batch_repeat_interleave'):
                result.batch_repeat_interleave(batch)
            elif isinstance(result, (tuple, list)):
                return tuple(tuple(t.repeat_interleave(batch, dim=0)
                                   for t in layer) for layer in result)
            else:
                raise TypeError(f'Unsupported cache type: {type(result).__name__}')
        return result

    def logits(self, prefix: list[int], rest: list[list[int]],
               rest_embeds: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple]:
        if not rest or any(len(row) != len(rest[0]) for row in rest):
            raise ValueError('KV scoring requires a nonempty equal-length batch')
        stored, key = self._get(prefix)
        before_length = stored.get_seq_length()
        past = self._repeat(stored, len(rest))
        prefix_length, rest_length = len(prefix), len(rest[0])
        positions = torch.arange(prefix_length, prefix_length + rest_length,
                                 device=self.adapter.device)
        started = time.perf_counter()
        kwargs = {'past_key_values': past, 'use_cache': False, 'logits_to_keep': 1,
                  'attention_mask': torch.ones((len(rest), prefix_length + rest_length),
                                               dtype=torch.long, device=self.adapter.device),
                  'position_ids': positions.unsqueeze(0).expand(len(rest), -1),
                  'cache_position': positions}
        if rest_embeds is None:
            kwargs['input_ids'] = torch.tensor(rest, dtype=torch.long,
                                               device=self.adapter.device)
        else:
            kwargs['inputs_embeds'] = rest_embeds
        output = self.adapter.model(**kwargs)
        if self.adapter.device.type == 'cuda':
            torch.cuda.synchronize(self.adapter.device)
        self.score_seconds += time.perf_counter() - started
        self.score_forwards += 1
        if stored.get_seq_length() != before_length:
            raise AssertionError('Candidate forward mutated stored prefix KV')
        return output.logits[:, -1], key

    def score(self, layouts, direction: str, batch_size: int):
        sequences = [layout.input_ids for layout in layouts]
        self.last_backend = 'not_used'
        if self.disabled_reason is not None:
            self.last_backend = 'native_fallback'
            return self.adapter.score_ids(sequences, direction, batch_size)

        try:
            groups = {}
            for index, layout in enumerate(layouts):
                prefix = tuple(layout.prefix_ids)
                rest_length = len(layout.input_ids) - len(prefix)
                if layout.input_ids[:len(prefix)] != list(prefix):
                    raise AssertionError('Layout prefix does not match full input IDs')
                if rest_length < len(layout.control_ids):
                    raise AssertionError('Suffix or following template tokens are missing')
                groups.setdefault(len(layout.input_ids), []).append((index, layout))
            results = [None] * len(layouts)
            width = min(batch_size, self.microbatch_size or batch_size)
            for indexed in groups.values():
                for offset in range(0, len(indexed), width):
                    chunk = indexed[offset:offset + width]
                    prefixes = {tuple(layout.prefix_ids) for _, layout in chunk}
                    if len(prefixes) != 1:
                        native = self.adapter.score_ids(
                            [layout.input_ids for _, layout in chunk], direction,
                            len(chunk))
                        for (index, _), row in zip(chunk, native):
                            results[index] = row
                        self.last_backend = 'mixed_prefix_native'
                        continue
                    prefix = next(iter(prefixes))
                    rest = [layout.input_ids[len(prefix):] for _, layout in chunk]
                    with torch.no_grad():
                        logits, key = self.logits(list(prefix), rest)
                        metrics = self.adapter.metrics_tensor(logits)
                        losses = self.adapter.loss_tensor(logits, direction)
                        if key not in self.verified_prefixes:
                            ids = torch.tensor([layout.input_ids for _, layout in chunk],
                                               dtype=torch.long, device=self.adapter.device)
                            reference = self.adapter.last_logits(
                                input_ids=ids, attention_mask=torch.ones_like(ids))
                            reference_metrics = self.adapter.metrics_tensor(reference)
                            reference_losses = self.adapter.loss_tensor(reference, direction)
                            score_error = float((metrics['score'].float() -
                                                 reference_metrics['score'].float()).abs().max())
                            loss_error = float((losses.float() -
                                                reference_losses.float()).abs().max())
                            self.max_verify_score_error = max(self.max_verify_score_error,
                                                              score_error)
                            self.max_verify_loss_error = max(self.max_verify_loss_error,
                                                             loss_error)
                            if score_error > self.tolerance or loss_error > self.loss_tolerance:
                                raise ArithmeticError(
                                    f'KV/full first-batch difference: score {score_error}, '
                                    f'loss {loss_error}, tolerances '
                                    f'{self.tolerance}/{self.loss_tolerance}')
                            self.verified_prefixes.add(key)
                            self.verified = True
                    metric_lists = {name: value.detach().float().cpu().tolist()
                                    for name, value in metrics.items()}
                    loss_list = losses.detach().cpu().tolist()
                    for position, (index, layout) in enumerate(chunk):
                        row = {name: values[position] for name, values in metric_lists.items()}
                        row.update({'loss': loss_list[position],
                                    'input_tokens': len(layout.input_ids),
                                    'input_truncated': False,
                                    'score_batch_size': len(chunk)})
                        results[index] = row
            if self.last_backend != 'mixed_prefix_native':
                self.last_backend = 'prefix_kv'
            return results
        except Exception as exc:
            self.disabled_reason = f'{type(exc).__name__}: {exc}'
            self.entries.clear()
            self.verified_prefixes.clear()
            self.last_backend = 'native_fallback'
            return self.adapter.score_ids(sequences, direction, batch_size)

    def gradient_logits(self, layout, control: torch.Tensor,
                        tail_embeds: torch.Tensor) -> torch.Tensor:
        rest_ids = layout.input_ids[layout.control_start:]
        embeds = torch.cat([control, tail_embeds], dim=1)
        logits, _ = self.logits(layout.prefix_ids, [rest_ids], embeds)
        return logits
