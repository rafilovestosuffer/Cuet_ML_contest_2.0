# Artifacts

Frozen out-of-fold (OOF) probability arrays, each shape `(6323, 8)`, aligned to
`data/folds/folds_canonical.csv` (row order = CSV order). Column order follows
`disaster.labels.LABELS`.

| File | Branch | OOF macro-F1 |
|---|---|---|
| `oof_banglabert_base.npy`  | BanglaBERT-base (text)     | 0.96587 |
| `oof_banglabert_multi.npy` | BanglaBERT multilingual    | 0.96656 |
| `oof_muril_large.npy`      | MuRIL-Large (text)         | 0.97556 |
| `oof_eva02_large.npy`      | EVA-02-Large @448 (image)  | 0.96714 |
| `oof_fusion_eva_muril.npy` | EVA-02 x MuRIL fusion      | 0.99212 |
| `oof_pl_muril.npy`         | PL-MuRIL (pseudo-labeled)  | 0.97525 |
| `oof_stack.npy`            | LightGBM stack (s3e)       | 0.99351 |

These are NOT committed to git (see `.gitignore`). Distribute via **Git LFS** or
a **GitHub Release**. To regenerate them, retrain each branch (see `src/disaster/train/`).
