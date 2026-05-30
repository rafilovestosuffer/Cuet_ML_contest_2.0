"""Focal loss with label smoothing (used by all trainable branches)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLabelSmoothingLoss(nn.Module):
    """Multiclass focal loss combined with label smoothing.

    L = (1 - p_t)^gamma * CE_smoothed
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
        # smoothed one-hot
        with torch.no_grad():
            true_dist = torch.full_like(logp, self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        ce = -(true_dist * logp).sum(dim=-1)
        pt = (true_dist * p).sum(dim=-1)
        focal = (1.0 - pt).clamp(min=1e-6) ** self.gamma * ce
        if self.weight is not None:
            focal = focal * self.weight[target]
        return focal.mean()
