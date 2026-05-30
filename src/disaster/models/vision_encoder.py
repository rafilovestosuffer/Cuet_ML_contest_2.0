"""timm backbone + linear head.

RECONSTRUCTED FROM PAPER SPEC — confirm the exact timm model strings from your
notebooks (EVA-02-Large @448, ConvNeXt V2-base @384).
"""
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None


class VisionClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 8,
                 drop_path_rate: float = 0.10, pretrained: bool = True):
        super().__init__()
        assert timm is not None, "pip install timm"
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            drop_path_rate=drop_path_rate)
        feat = self.backbone.num_features
        self.head = nn.Linear(feat, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat), feat
