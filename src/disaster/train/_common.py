# RECONSTRUCTED REFERENCE IMPLEMENTATION — generated from specification.
# This is NOT the original training code and is NOT verified to reproduce the
# released OOF arrays. Reconcile with the author's original notebooks.

"""Shared 5-fold OOF training driver.

Implements the paper's shared recipe so each branch entry point stays thin:
  * AdamW with layer-wise LR (backbone vs head), weight_decay 0.01
  * cosine schedule with 10% linear warmup (disaster.losses.cosine_warmup)
  * AMP + gradient accumulation (micro-batch 4 × accum 8 = effective 32)
  * focal + label-smoothing loss (disaster.losses.FocalLabelSmoothingLoss)
  * writes OOF (aligned to folds_canonical.csv) + averaged test probabilities

The training loop body is reconstructed; it has not been run end-to-end here.
"""
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None

from disaster.losses import FocalLabelSmoothingLoss, cosine_warmup
from disaster.seed import set_seed


def build_optimizer(model, backbone_modules, lr_backbone, lr_head, weight_decay):
    """AdamW with a low LR on pretrained backbones and a high LR on new heads."""
    backbone_params, head_params = [], []
    backbone_ids = set()
    for m in backbone_modules:
        for p in m.parameters():
            backbone_ids.add(id(p))
            if p.requires_grad:
                backbone_params.append(p)
    for p in model.parameters():
        if id(p) not in backbone_ids and p.requires_grad:
            head_params.append(p)
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=weight_decay)


def train_kfold_oof(df, cfg, model_factory, backbone_getter, dataset_factory,
                    batch_to_inputs, n_classes=8, test_df=None,
                    device="cuda"):
    """Generic k-fold OOF trainer.

    Parameters
    ----------
    df : training frame with a `fold` and `label` column
    cfg : dict of hyperparameters (see configs/*.yaml)
    model_factory : () -> nn.Module
    backbone_getter : (model) -> list[nn.Module] to receive the low LR
    dataset_factory : (frame, is_test: bool) -> torch Dataset
    batch_to_inputs : (batch, device) -> (kwargs_for_model, labels_or_None)
    test_df : optional test frame for averaged test-probability output

    Returns
    -------
    (oof_probs[N, C], test_probs[M, C] | None)
    """
    assert torch is not None, "pip install torch"
    set_seed(cfg.get("seed", 42))
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    oof = np.zeros((len(df), n_classes), np.float32)
    test_acc = (np.zeros((len(test_df), n_classes), np.float32)
                if test_df is not None else None)
    folds = sorted(df["fold"].unique())
    loss_fn = FocalLabelSmoothingLoss(
        gamma=cfg.get("gamma", 2.0), smoothing=cfg.get("label_smooth", 0.05),
        num_classes=n_classes)

    for fold in folds:
        tr = df[df.fold != fold].reset_index(drop=True)
        va = df[df.fold == fold]
        va_idx = va.index.values
        va = va.reset_index(drop=True)

        model = model_factory().to(dev)
        opt = build_optimizer(
            model, backbone_getter(model),
            lr_backbone=cfg.get("lr_backbone", cfg.get("lr_text", 8e-6)),
            lr_head=cfg.get("lr_head", 5e-4),
            weight_decay=cfg.get("weight_decay", 0.01))

        micro = cfg.get("batch_size", 4)
        accum = cfg.get("grad_accum", 8)
        epochs = cfg.get("epochs", 5)
        tr_loader = DataLoader(dataset_factory(tr, False), batch_size=micro,
                               shuffle=True, num_workers=2, pin_memory=True)
        va_loader = DataLoader(dataset_factory(va, False), batch_size=micro * 2,
                               shuffle=False, num_workers=2, pin_memory=True)

        n_steps = (len(tr_loader) // accum) * epochs
        sched = cosine_warmup(opt, int(cfg.get("warmup_ratio", 0.10) * n_steps),
                              n_steps)
        scaler = GradScaler()

        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            for i, batch in enumerate(tr_loader):
                inputs, labels = batch_to_inputs(batch, dev)
                with autocast(dtype=torch.float16):
                    logits = model(**inputs)
                    loss = loss_fn(logits, labels) / accum
                scaler.scale(loss).backward()
                if (i + 1) % accum == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    sched.step()

        # OOF predictions for this fold's validation rows
        model.eval()
        preds = []
        with torch.no_grad(), autocast(dtype=torch.float16):
            for batch in va_loader:
                inputs, _ = batch_to_inputs(batch, dev)
                preds.append(F.softmax(model(**inputs).float(), -1).cpu().numpy())
        oof[va_idx] = np.concatenate(preds)

        if test_df is not None:
            te_loader = DataLoader(dataset_factory(test_df, True),
                                   batch_size=micro * 2, shuffle=False,
                                   num_workers=2, pin_memory=True)
            tp = []
            with torch.no_grad(), autocast(dtype=torch.float16):
                for batch in te_loader:
                    inputs, _ = batch_to_inputs(batch, dev)
                    tp.append(F.softmax(model(**inputs).float(), -1).cpu().numpy())
            test_acc += np.concatenate(tp) / len(folds)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return oof, test_acc


def save_outputs(oof, test, out_dir, run_id):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{run_id}_oof.npy", oof)
    if test is not None:
        np.save(out / f"{run_id}_test_probs.npy", test)
    print(f"Saved {run_id}_oof.npy"
          + (f" and {run_id}_test_probs.npy" if test is not None else "")
          + f" → {out}")
