"""Text-anchored cross-attention fusion (JointMM).

RECONSTRUCTED FROM PAPER SPEC — reconcile with your original notebook before use.

Image encoder -> v (1024); MuRIL pooled -> t (1024). Both projected to d_f=512.
Text is the Query, image the Key/Value into 8-head attention. The classifier
sees concat[attn_out, text_proj]. A `late_fusion` baseline is also provided so
the ablation (late vs cross-attention) is reproducible.
"""
import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim=1024, img_dim=1024, d_f=512, n_heads=8,
                 num_classes=8, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim),
                                       nn.Linear(text_dim, d_f), nn.GELU())
        self.img_proj = nn.Sequential(nn.LayerNorm(img_dim),
                                      nn.Linear(img_dim, d_f), nn.GELU())
        self.attn = nn.MultiheadAttention(d_f, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * d_f), nn.Dropout(dropout),
                                  nn.Linear(2 * d_f, num_classes))

    def forward(self, text_feat, img_feat):
        q = self.text_proj(text_feat).unsqueeze(1)   # (B,1,d_f)
        kv = self.img_proj(img_feat).unsqueeze(1)     # (B,1,d_f)
        a, _ = self.attn(q, kv, kv)                   # text queries image
        z = torch.cat([a.squeeze(1), q.squeeze(1)], dim=-1)
        return self.head(z)


class LateFusion(nn.Module):
    """Baseline: weighted average of independent text/image posteriors."""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, text_prob, img_prob):
        return self.alpha * text_prob + (1 - self.alpha) * img_prob
