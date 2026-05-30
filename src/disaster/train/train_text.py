"""train_text — TRAINING ENTRY POINT (stub).

TODO (Claude Code): port the corresponding logic from notebooks/ here.
Shared recipe (see README / paper): FocalLabelSmoothingLoss(gamma=2, smoothing=0.05),
AdamW (layer-wise LR: text 8e-6 / image 1e-5 / head 5e-4, wd 0.01), cosine schedule
with 10% warmup, AMP, gradient checkpointing, grad accumulation (micro 4 x 8 = 32),
5-fold OOF using data/folds/folds_canonical.csv. Save OOF + test probabilities to
artifacts/ with the names in artifacts/README.md.
"""

def main():
    raise NotImplementedError(
        "Port train_text from notebooks/ — see module docstring for the recipe.")

if __name__ == "__main__":
    main()
