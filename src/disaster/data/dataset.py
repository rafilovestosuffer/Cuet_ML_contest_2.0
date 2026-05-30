# TRANSCRIBED FROM notebooks/Final_notebook.ipynb — faithful to the original.

"""PyTorch Datasets for text / image / multimodal inputs.

The multimodal dataset (`MMDataset`) is copied from the notebook; the text-only
and image-only datasets are thin specializations used by the unimodal branches.

Each row needs an `img_path` (resolved by `resolve_img`) and a `context` string;
training rows additionally carry an integer `label`.
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from disaster.data.transforms import pad_resize


def resolve_img(img_id, base_dir: str | Path):
    """Find the image file for an id under base_dir, trying common extensions
    and one level of subdirectories (matches the notebook's resolver)."""
    base = Path(base_dir)
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
        p = base / f"{img_id}{ext}"
        if p.exists():
            return p
    for sub in base.iterdir():
        if sub.is_dir():
            for ext in [".jpg", ".jpeg", ".png"]:
                p = sub / f"{img_id}{ext}"
                if p.exists():
                    return p
    raise FileNotFoundError(img_id)


class MMDataset(Dataset):
    """Multimodal dataset: returns image tensor + tokenized text (+ label)."""

    def __init__(self, df, tokenizer, max_len, img_size, tfm, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        self.img_size = img_size
        self.tfm = tfm
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = pad_resize(Image.open(row["img_path"]), self.img_size)
        img = self.tfm(image=img)["image"]
        enc = self.tok(str(row["context"]), max_length=self.max_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["image"] = img
        if not self.is_test:
            item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item


class TextDataset(Dataset):
    """Text-only dataset for the BanglaBERT / MuRIL branches."""

    def __init__(self, df, tokenizer, max_len=256, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tok(str(row["context"]), max_length=self.max_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if not self.is_test:
            item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item


class ImageDataset(Dataset):
    """Image-only dataset for the EVA-02 / ConvNeXt branches."""

    def __init__(self, df, img_size, tfm, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.tfm = tfm
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = pad_resize(Image.open(row["img_path"]), self.img_size)
        img = self.tfm(image=img)["image"]
        item = {"image": img}
        if not self.is_test:
            item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item
