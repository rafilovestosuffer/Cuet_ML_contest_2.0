# RECONSTRUCTED REFERENCE IMPLEMENTATION — generated from specification.
# This is NOT the original training code and is NOT verified to reproduce the
# released OOF arrays. Reconcile with the author's original notebooks.

"""timm backbone + linear head.

Confirmed timm strings (from the notebook):
  * convnextv2_base.fcmae_ft_in22k_in1k_384   (ConvNeXt V2-base @384)
  * eva02_large_patch14_448.mim_m38m_ft_in22k_in1k   (EVA-02-Large @448)
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
