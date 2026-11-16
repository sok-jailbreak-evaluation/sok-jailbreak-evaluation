import math
import torch
import torch.nn.functional as F

def token_losses(logits, ids):
    """BOS predicts first text token; last logit has no target and is excluded."""
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           ids[:, 1:].reshape(-1), reduction='none')

def summarize(losses):
    values = [float(x) for x in losses]
    if not values or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError('Empty or non-finite token NLL')
    total = math.fsum(values)
    mean = total / len(values)
    return {'scored_tokens': len(values), 'sum_nll': total, 'mean_nll': mean,
            'ppl': math.exp(mean), 'token_nll': values, 'native_pass': None}
