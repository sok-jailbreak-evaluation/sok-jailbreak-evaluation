"""Standard full-vocabulary causal PPL, reusing the frozen Llama token loss."""
import math
import os
from pathlib import Path
import platform
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

from common import CASES, CONF, ROOT, read, windows
sys.path.insert(0, str(CASES))
from revision_ppl import token_losses, summarize


def regions(losses, offsets, original_chars):
    bins = {'original': [], 'appended': [], 'boundary': []}
    assert len(losses) == len(offsets)
    for loss, (start, end) in zip(losses, offsets):
        if end <= original_chars: name = 'original'
        elif start >= original_chars: name = 'appended'
        else: name = 'boundary'
        bins[name].append(loss)
    return {name: {'scored_tokens': len(values), 'sum_nll': math.fsum(values),
                  'mean_nll': math.fsum(values)/len(values) if values else None}
            for name, values in bins.items()}


class Scorer:
    def __init__(self, name):
        self.name = name
        self.config = CONF['models'][name]
        if name == 'gpt2':
            path, revision = os.environ.get('GPT2_MODEL','models/gpt2'), self.config.get('revision')
        else: path, revision = self.config['model'], self.config['revision']
        if torch.cuda.device_count() != 1: raise RuntimeError('One visible GPU per worker required')
        torch.set_num_threads(6); torch.manual_seed(CONF['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., self.config['memory_limit_bytes']/total))
        free_before, _ = torch.cuda.mem_get_info()
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True, legacy=False)
        dtype = {'float16': torch.float16, 'float32': torch.float32}[self.config['dtype']]
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
            dtype=dtype, attn_implementation='sdpa').to('cuda:0')
        self.model.eval().requires_grad_(False)
        self.context = int(getattr(self.model.config, 'max_position_embeddings', None) or self.model.config.n_positions)
        assert self.context == self.config['expected_context']
        self.stride = self.context//2
        if self.tokenizer.bos_token_id is None: raise ValueError('Explicit model-native BOS required')
        self.calls = 0
        self.environment = {'python': platform.python_version(), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'model_path': path, 'revision': revision,
            'model_type': self.model.config.model_type, 'dtype': str(next(self.model.parameters()).dtype),
            'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid), 'gpu_name': torch.cuda.get_device_name(0),
            'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'visible_device_count': torch.cuda.device_count(),
            'free_bytes_before_load': free_before, 'memory_limit_bytes': self.config['memory_limit_bytes'],
            'context': self.context, 'stride': self.stride, 'bos_token_id': self.tokenizer.bos_token_id,
            'bos_token': self.tokenizer.bos_token, 'EOS_appended': False, 'chat_template': False,
            'attention': self.model.config._attn_implementation, 'API_requests': 0}

    def encode(self, text):
        encoded = self.tokenizer(text, add_special_tokens=False, truncation=False,
                                 return_offsets_mapping=True, verbose=False)
        ids = [self.tokenizer.bos_token_id]+encoded['input_ids']
        offsets = [(-1, -1)]+[tuple(x) for x in encoded['offset_mapping']]
        if len(ids) < 2: raise ValueError('No response tokens to score')
        return ids, offsets

    @torch.inference_mode()
    def score(self, text):
        started = time.perf_counter()
        ids, offsets = self.encode(text)
        losses, pieces = [], []
        cursor = 1
        for window in windows(len(ids), self.context, self.stride):
            assert window['target_start'] == cursor
            x = torch.tensor([ids[window['start']:window['end']]], device='cuda:0')
            labels = x.clone()
            local_target = window['target_start']-window['start']
            labels[:, :local_target] = -100
            expected = window['target_end']-window['target_start']
            actual = int((labels[:, 1:] != -100).sum())
            assert actual == expected
            torch.cuda.synchronize(); before = time.perf_counter()
            output = self.model(input_ids=x, attention_mask=torch.ones_like(x), labels=labels, use_cache=False)
            selected = token_losses(output.logits, x)[local_target-1:].double().cpu().tolist()
            assert len(selected) == expected
            explicit = math.fsum(selected)/len(selected)
            native = float(output.loss)
            error = abs(native-explicit)
            if error > CONF['scoring']['mean_nll_agreement_atol']:
                raise RuntimeError(f'Native/explicit mean NLL disagreement {error}')
            torch.cuda.synchronize()
            pieces.append({**window, 'target_count': expected, 'sum_nll': math.fsum(selected),
                'mean_nll': explicit, 'native_loss': native, 'native_loss_abs_error': error,
                'seconds': time.perf_counter()-before})
            losses.extend(selected); cursor = window['end']; self.calls += 1
            del output, x, labels
        assert cursor == len(ids) and len(losses) == len(ids)-1
        return {**summarize(losses), 'input_ids': ids, 'offset_mapping': offsets,
            'input_tokens_including_bos': len(ids),
            'all_target_tokens_scored_exactly_once': True, 'context': self.context, 'stride': self.stride,
            'windows': pieces, 'model_forwards': len(pieces),
            'max_native_loss_abs_error': max(p['native_loss_abs_error'] for p in pieces),
            'elapsed_seconds': time.perf_counter()-started}


def diagnostics(before, after, original_chars):
    breakdown = regions(after['token_nll'], after['offset_mapping'][1:], original_chars)
    prefix = after['input_ids'][:len(before['input_ids'])] == before['input_ids']
    result = {'regions': breakdown, 'prefix_token_ids_unchanged': prefix,
        'all_original_characters_preserved': True}
    if prefix:
        count = before['scored_tokens']
        diffs = [abs(a-b) for a,b in zip(before['token_nll'],after['token_nll'][:count])]
        result.update({'original_token_nll_in_modified_mean': math.fsum(after['token_nll'][:count])/count,
                       'original_token_nll_max_abs_change': max(diffs),
                       'original_token_nll_mean_change': math.fsum(after['token_nll'][:count])/count-before['mean_nll']})
    assert sum(v['scored_tokens'] for v in breakdown.values()) == after['scored_tokens']
    assert abs(math.fsum(v['sum_nll'] for v in breakdown.values())-after['sum_nll']) < 1e-7
    return result
