# TRANSCRIBED FROM notebooks/Final_notebook.ipynb — faithful to the original
# code that produced the released artifacts. Not re-executed in CI (needs raw
# images), but copied verbatim from the notebook rather than reconstructed.

"""Albumentations pipelines + aspect-preserving pad-resize.

Image preprocessing for every vision / multimodal branch:
  * pad_resize: resize the long side to `size`, then zero-pad to a square so the
    aspect ratio is preserved (no distortion).
  * build_train_tfms / build_val_tfms: ImageNet-normalized tensors; train adds
    the augmentation stack from the paper.
"""
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:  # pragma: no cover - albumentations optional at import time
    A = None
    ToTensorV2 = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pad_resize(img_pil, size: int) -> np.ndarray:
    """Resize preserving aspect ratio, then zero-pad to (size, size, 3)."""
    img = np.array(img_pil.convert("RGB"))
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    img = A.Resize(nh, nw)(image=img)["image"]
    pad_h, pad_w = size - nh, size - nw
    return np.pad(
        img,
        ((pad_h // 2, pad_h - pad_h // 2),
         (pad_w // 2, pad_w - pad_w // 2),
         (0, 0)),
        constant_values=0,
    )


def build_train_tfms(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Training augmentation stack (exact pipeline from the notebook)."""
    assert A is not None, "pip install albumentations"
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ShiftScaleRotate(0.05, 0.10, 12, p=1.0),
            A.Affine(scale=(0.92, 1.08), translate_percent=0.04,
                     rotate=(-10, 10), p=1.0),
        ], p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(0.15, 0.15, p=1.0),
            A.HueSaturationValue(10, 15, 10, p=1.0),
        ], p=0.4),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])


def build_val_tfms(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Validation / inference transform: normalize + to-tensor only."""
    assert A is not None, "pip install albumentations"
    return A.Compose([A.Normalize(mean, std), ToTensorV2()])
