"""Loss functions and the cosine schedule used by the trainable branches.

`soft_ce` and `cosine_warmup` are TRANSCRIBED FROM notebooks/Final_notebook.ipynb.
`FocalLabelSmoothingLoss` is RECONSTRUCTED FROM SPEC (the paper describes focal
loss + label smoothing; the notebook's shown loss is the label-smoothed soft CE
below) — reconcile with the original training cells before relying on it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TRANSCRIBED FROM NOTEBOOK
# ---------------------------------------------------------------------------
def soft_ce(logits, soft_targets, smoothing: float = 0.05):
    """Label-smoothed cross-entropy against soft targets (notebook verbatim)."""
    n = logits.size(1)
    lp = F.log_softmax(logits.float(), dim=-1)
    soft = soft_targets * (1 - smoothing) + smoothing / n
    return -(soft * lp).sum(-1).mean()


def cosine_warmup(optim, n_warmup: int, n_total: int):
    """Linear warmup then cosine decay LR schedule (notebook verbatim)."""
    def fn(step):
        if step < n_warmup:
            return step / max(1, n_warmup)
        prog = (step - n_warmup) / max(1, n_total - n_warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(optim, fn)


# ---------------------------------------------------------------------------
# RECONSTRUCTED FROM SPEC
# ---------------------------------------------------------------------------
class FocalLabelSmoothingLoss(nn.Module):
    """Multiclass focal loss combined with label smoothing.

    L = (1 - p_t)^gamma * CE_smoothed

    RECONSTRUCTED FROM PAPER SPEC — the notebook's shown objective is `soft_ce`
    (label-smoothed CE on soft targets). The paper text describes focal + label
    smoothing; this implements that description. Reconcile before use.
    """

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.05,
                 num_classes: int = 8, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        with torch.no_grad():
            true_dist = torch.full_like(
                logp, self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        ce = -(true_dist * logp).sum(dim=-1)
        pt = (true_dist * p).sum(dim=-1)
        focal = (1.0 - pt).clamp(min=1e-6) ** self.gamma * ce
        if self.weight is not None:
            focal = focal * self.weight[target]
        return focal.mean()
