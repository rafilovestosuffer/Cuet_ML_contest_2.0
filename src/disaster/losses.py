import math

import torch
import torch.nn.functional as F


def soft_ce(logits, soft_targets, smoothing: float = 0.05):
    n = logits.size(1)
    lp = F.log_softmax(logits.float(), dim=-1)
    soft = soft_targets * (1 - smoothing) + smoothing / n
    return -(soft * lp).sum(-1).mean()


def cosine_warmup(optim, n_warmup: int, n_total: int):
    def fn(step):
        if step < n_warmup:
            return step / max(1, n_warmup)
        prog = (step - n_warmup) / max(1, n_total - n_warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(optim, fn)
