"""HuggingFace encoder + masked mean pooling + linear head.

RECONSTRUCTED FROM PAPER SPEC — confirm the exact checkpoint IDs from your
notebooks. Defaults: csebuetnlp/banglabert, google/muril-large-cased.
The multilingual variant ('bb_multi') MUST be confirmed (TODO).
"""
import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class TextClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 8, dropout: float = 0.1):
        super().__init__()
        assert AutoModel is not None, "pip install transformers"
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        return self.head(pooled), pooled
