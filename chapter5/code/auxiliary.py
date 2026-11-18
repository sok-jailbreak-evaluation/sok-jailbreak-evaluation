"""Score unchanged archived responses with pinned PPL and PARDEN protocols."""

import argparse
import ast
from collections import Counter
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'code/auxiliary_reference'
MODELS = Path('models')
MODEL_SPECS = {
    'ppl_gpt2': ('gpt2', '607a30d783dfa663caf39e06633721c8d4cfcd7e', 'float32', 1024),
    'ppl_llama2': ('Llama-2-7b-chat-hf', 'f5db02db724555f92da89c216ac04704f23d4590', 'float16', 4096),
    'parden': ('Llama-2-7b-chat-hf', 'f5db02db724555f92da89c216ac04704f23d4590', 'float16', 4096),
}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_data(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.split(b'\n') if line.strip()]
    if not rows or len({r['attempt_id'] for r in rows}) != len(rows):
        raise ValueError('Empty input or duplicate attempt IDs')
    for row in rows:
        for key in ('attempt_id', 'goal_id', 'goal', 'method', 'target'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError('Invalid archived row identity')
    return raw, rows


def valid_response(row):
    return (row.get('response_status') in (None, 'valid')
            and isinstance(row.get('response'), str) and bool(row['response'].strip()))


def input_text(row, scoring_object):
    if scoring_object == 'response':
        return row.get('response') if valid_response(row) else None
    prompt = row.get('attack_prompt')
    return prompt if isinstance(prompt, str) and prompt.strip() else None


def windows(length, context, stride):
    previous_end = 1
    for start in range(0, length, stride):
        end = min(start + context, length)
        target_start = max(previous_end, start + 1)
        if target_start < end:
            yield start, end, target_start
        previous_end = end
        if end == length:
            break


def official_functions(torch):
    source = REFERENCE / 'parden_utils.py'
    names = {'flush_memory', 'templatize', 'templatize_for_repeat', 'generate_mini_batch', 'get_windowed_bleu'}
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError('Incomplete frozen PARDEN source')
    namespace = {'torch': torch, 'gc': gc}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), 'exec'), namespace)
    return namespace


class Scorer:
    def __init__(self, name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one physical GPU must be visible')
        self.torch = torch
        self.name = name
        directory, revision, precision, context = MODEL_SPECS[name]
        self.path = MODELS / directory
        torch.set_num_threads(6)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True,
                                                       use_fast=True, legacy=False)
        if name == 'parden':
            self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, local_files_only=True, dtype=getattr(torch, precision),
            attn_implementation='sdpa').to('cuda:0')
        self.model.eval().requires_grad_(False)
        self.context = int(getattr(self.model.config, 'max_position_embeddings', None)
                           or self.model.config.n_positions)
        if self.context != context or self.tokenizer.bos_token_id is None:
            raise ValueError('Pinned model context or BOS does not match the protocol')
        self.functions = official_functions(torch) if name == 'parden' else None
        calibration = json.loads((ROOT.parent / 'chapter4/batch/config.json').read_text())['parden']
        self.threshold = calibration['threshold'] if name == 'parden' else None
        self.metadata = {
            'model_path': str(self.path), 'revision': revision, 'dtype': precision,
            'context': self.context, 'attention': 'sdpa', 'seed': 42,
            'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'visible_gpus': torch.cuda.device_count(), 'gpu': torch.cuda.get_device_name(0),
            'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
            'torch': torch.__version__,
        }

    def ppl(self, text):
        torch = self.torch
        ids = [self.tokenizer.bos_token_id] + self.tokenizer(
            text, add_special_tokens=False, truncation=False, verbose=False)['input_ids']
        if len(ids) < 2:
            raise ValueError('No response tokens to score')
        losses, pieces = [], []
        with torch.inference_mode():
            for start, end, target_start in windows(len(ids), self.context, self.context // 2):
                tensor = torch.tensor([ids[start:end]], device='cuda:0')
                labels = tensor.clone()
                local_target = target_start - start
                labels[:, :local_target] = -100
                output = self.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                    labels=labels, use_cache=False)
                selected = torch.nn.functional.cross_entropy(
                    output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                    tensor[:, 1:].reshape(-1), reduction='none')[local_target - 1:].double().cpu().tolist()
                if len(selected) != end - target_start:
                    raise ValueError('Sliding window token accounting mismatch')
                explicit = math.fsum(selected) / len(selected)
                native = float(output.loss)
                error = abs(native - explicit)
                if error > 1e-5:
                    raise ValueError('Native and explicit mean NLL differ')
                pieces.append({'start': start, 'end': end, 'target_start': target_start,
                               'target_end': end, 'scored_tokens': len(selected),
                               'sum_nll': math.fsum(selected), 'native_loss': native,
                               'native_loss_abs_error': error})
                losses.extend(selected)
                del output, labels, tensor
        if len(losses) != len(ids) - 1 or not all(math.isfinite(v) and v >= 0 for v in losses):
            raise ValueError('Invalid token NLL accounting')
        total = math.fsum(losses)
        mean = total / len(losses)
        return {'score': math.exp(mean), 'ppl': math.exp(mean), 'mean_nll': mean, 'sum_nll': total,
                'scored_tokens': len(losses), 'input_tokens': len(ids), 'input_ids': ids,
                'token_nll': losses, 'windows': pieces, 'model_forwards': len(pieces),
                'all_response_tokens_scored_once': True, 'native_bos': True,
                'eos_appended': False, 'chat_template': False,
                'threshold': None, 'decision': None, 'outcome': '?'}

    def parden(self, texts):
        scorer = self

        class Recorder:
            @property
            def device(self):
                return scorer.model.device

            def generate(self, **kwargs):
                output = scorer.model.generate(**kwargs)
                width = kwargs['input_ids'].shape[1]
                self.saved = [{'input_ids': x.tolist(), 'attention_mask': mask.tolist(),
                               'generated_ids': y[width:].tolist()}
                              for x, mask, y in zip(kwargs['input_ids'], kwargs['attention_mask'], output)]
                return output

        functions = self.functions
        templates = functions['templatize_for_repeat'](texts)
        lengths = [len(self.tokenizer.encode(template, add_special_tokens=True)) for template in templates]
        if any(length + 100 > self.context for length in lengths):
            raise ValueError('Complete PARDEN input exceeds context')
        recorder = Recorder()
        repeats = functions['generate_mini_batch'](recorder, self.tokenizer, templates,
                                                  num_samples=1, max_new_tokens=30, do_sample=False)
        results = []
        for text, repeat, length, saved in zip(texts, repeats, lengths, recorder.saved):
            scores, effective = functions['get_windowed_bleu'](text, repeat)
            if not saved['generated_ids'] or not all(math.isfinite(x) and 0 <= x <= 1 for x in scores):
                raise ValueError('Invalid PARDEN generation or score')
            decision = int(scores[3] < self.threshold)
            results.append({'score': scores[3], 'bleu60': scores[3], 'bleu': scores,
                            'threshold': self.threshold, 'decision': decision, 'outcome': str(decision),
                            'input_tokens': length, 'output_tokens': len(saved['generated_ids']),
                            'effective_windows': effective, 'window_sizes': [30,40,50,60,70,80,90,100],
                            'actual_repeat': repeat, 'model_forwards': None, **saved})
        return results


def base_result(row, name, scoring_object):
    result = {key: row[key] for key in ('attempt_id', 'goal_id', 'method', 'target')}
    response = row.get('response')
    result.update(model=name, status='missing_response', decision=None, outcome='?', score=None,
                  reason=None, response_characters=len(response) if isinstance(response, str) else None)
    text = input_text(row, scoring_object)
    result.update(evaluation_object=scoring_object,
                  input_scope='unchanged_archived_response' if scoring_object == 'response' else row.get('attack_prompt_scope'),
                  text_characters=len(text) if text is not None else None)
    if scoring_object == 'attack_prompt':
        result['status'] = 'missing_input'
    return result


def run(args):
    raw, rows = load_data(args.data)
    if args.limit:
        rows = rows[:args.limit]
    if args.target:
        rows = [row for row in rows if row['target'] == args.target]
    if not rows:
        raise ValueError('No rows selected')
    manifest = {'model': args.model, 'rows': len(rows), 'limit': args.limit,
                'evaluation_object': args.object, 'target': args.target,
                'model_spec': MODEL_SPECS[args.model],
                'input_policy': ('Complete unchanged raw response; no text truncation or substitutions'
                                 if args.object == 'response' else
                                 'Complete saved plain-text prompt or documented reconstruction; structured chats omitted'),
                'ppl_protocol': 'Native BOS, no EOS/chat, each response token once; context/2 sliding windows',
                'parden_protocol': 'Official greedy 100-token repeat, character crop and BLEU60, existing full calibration',
                'batch_size': 4 if args.model == 'parden' else 1,
                'api_requests': 0}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / 'manifest.json'
        manifest = json.loads(json.dumps(manifest))
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError('Existing output has different data, code, or protocol')
        else:
            if any(args.out.glob('*.json')):
                raise ValueError('Existing results have no manifest')
            write(manifest_path, manifest)
        pending, counts = [], Counter()
        for row in rows:
            path = args.out / (sha(row['attempt_id'].encode()) + '.json')
            if path.exists():
                saved = json.loads(path.read_text())
                if any(saved[key] != row[key] for key in ('attempt_id', 'goal_id', 'target', 'method')) or saved['model'] != args.model:
                    raise ValueError('Stored result identity mismatch')
                counts[saved['status']] += 1
            else:
                pending.append((row, path))
        scorer = Scorer(args.model) if any(input_text(row, args.object) is not None for row, _ in pending) else None
        if scorer:
            environment_path = args.out / 'environment.json'
            if environment_path.exists():
                old = json.loads(environment_path.read_text())
                if any(old[k] != scorer.metadata[k] for k in ('revision', 'dtype', 'context', 'attention', 'torch')):
                    raise ValueError('Scoring environment changed across resume')
            else:
                write(environment_path, scorer.metadata)
        started = time.perf_counter()
        batch_size = 4 if args.model == 'parden' else 1
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            todo = []
            for row, path in chunk:
                base = base_result(row, args.model, args.object)
                text = input_text(row, args.object)
                if text is None:
                    base['reason'] = ('Archived response is missing or invalid' if args.object == 'response'
                                      else 'No plain-text archived or reconstruction-rule prompt; structured chats are not serialized for PPL')
                    write(path, base)
                    counts[base['status']] += 1
                    continue
                if args.model == 'parden':
                    template = scorer.functions['templatize_for_repeat']([text])[0]
                    tokens = len(scorer.tokenizer.encode(template, add_special_tokens=True))
                    if tokens + 100 > scorer.context:
                        base.update(status='input_incompatible', reason='Complete input plus 100 generation tokens exceeds context',
                                    input_tokens=tokens, context=scorer.context)
                        write(path, base)
                        counts[base['status']] += 1
                        continue
                todo.append((row, path, base))
            if todo:
                scorer.torch.cuda.reset_peak_memory_stats()
                scorer.torch.cuda.synchronize()
                before = time.perf_counter()
                if args.model == 'parden':
                    scored = scorer.parden([input_text(row, args.object) for row, _, _ in todo])
                else:
                    scored = [scorer.ppl(input_text(todo[0][0], args.object))]
                scorer.torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                allocated = scorer.torch.cuda.max_memory_allocated()
                reserved = scorer.torch.cuda.max_memory_reserved()
                for (_, path, base), result in zip(todo, scored):
                    base.update(result, status='valid', elapsed_seconds=elapsed / len(todo),
                                batch_elapsed_seconds=elapsed, batch_size=len(todo),
                                peak_allocated_bytes=allocated, peak_reserved_bytes=reserved)
                    write(path, base)
                    counts['valid'] += 1
            progress = {'model': args.model, 'completed': sum(counts.values()), 'planned': len(rows),
                        'status_counts': dict(counts), 'seconds': time.perf_counter() - started}
            write(args.out / 'progress.json', progress)
            if start == 0 or sum(counts.values()) % 40 < batch_size:
                print(json.dumps(progress), flush=True)
        status = {'complete': sum(counts.values()) == len(rows), 'model': args.model,
                  'rows': len(rows), 'status_counts': dict(counts),
                  'seconds_this_process': time.perf_counter() - started}
        write(args.out / 'status.json', status)
        print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--object', choices=('response', 'attack_prompt'), default='response')
    parser.add_argument('--target')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.object == 'attack_prompt' and args.model == 'parden':
        parser.error('PARDEN evaluates responses')
    if args.model == 'parden' and args.data.name == 'extension.jsonl' and args.out.name == 'parden':
        from parden_queue import execute
        execute()
    else:
        run(args)


if __name__ == '__main__':
    main()
