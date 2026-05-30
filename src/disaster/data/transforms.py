"""Albumentations pipelines + aspect-preserving pad-resize (stub).

Train aug (paper): HorizontalFlip(.5); OneOf[ShiftScaleRotate, Affine](.5);
OneOf[RandomBrightnessContrast, HueSaturationValue](.4); CoarseDropout(.2);
ImageNet normalization. Provide build_train_tfms(size) / build_val_tfms(size).
"""
