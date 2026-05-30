# RECONSTRUCTED REFERENCE IMPLEMENTATION — generated from specification.
# This is NOT the original training code and is NOT verified to reproduce the
# released OOF arrays. Reconcile with the author's original notebooks.

"""HuggingFace encoder + masked mean pooling + linear head.

Confirmed checkpoint (from the notebook): google/muril-large-cased.
Text-branch checkpoints are not shown in the final notebook; defaults follow the
spec: csebuetnlp/banglabert (base) and a multilingual variant
(TODO: confirm — likely csebuetnlp/banglishbert or bert-base-multilingual-cased).
"""
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
