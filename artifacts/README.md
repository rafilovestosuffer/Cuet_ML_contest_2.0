# Artifacts

Frozen probability arrays from trained models, aligned to
`data/folds/folds_canonical.csv` (row order = CSV order).
Column order follows `disaster.labels.LABELS` (8 classes).

These files are **NOT committed to git** (`artifacts/*.npy` is gitignored).
Distribute via **Git LFS** or a **GitHub Release**.

---

## Out-of-fold arrays — shape `(6323, 8)`

| File | Branch | OOF macro-F1 |
|---|---|---|
| `oof_banglabert_base.npy`  | BanglaBERT-base (multi-seed avg, seeds 123+456) | 0.96587 |
| `oof_banglabert_multi.npy` | BanglaBERT multilingual                         | 0.96656 |
| `oof_muril_large.npy`      | MuRIL-Large                                     | 0.97556 |
| `oof_eva02_large.npy`      | EVA-02-Large @448                               | 0.96714 |
| `oof_fusion_eva_muril.npy` | EVA-02 × MuRIL cross-attention fusion           | 0.99212 |
| `oof_pl_muril.npy`         | PL-MuRIL (pseudo-labeled re-fine-tune)          | 0.97525 |
| `oof_stack.npy`            | LightGBM stack (s3e)                            | 0.99351 |
| `final_oof.npy`            | Pre-calibration blend output (s2)               | 0.99135 |

## Specialist OOF — shape `(2403, 3)`

| File | Description |
|---|---|
| `oof_specialist.npy` | 3-class {Earthquake, Flood, Landslides} OOF; rows = the subset of `folds_canonical.csv` where `label ∈ {1,2,4}` |

## Test probability arrays — shape `(1580, 8)`

| File | Branch |
|---|---|
| `test_banglabert_base.npy`  | BanglaBERT-base |
| `test_banglabert_multi.npy` | BanglaBERT multilingual |
| `test_muril_large.npy`      | MuRIL-Large |
| `test_eva02_large.npy`      | EVA-02-Large @448 |
| `test_fusion_eva_muril.npy` | EVA-02 × MuRIL fusion |
| `test_pl_muril.npy`         | PL-MuRIL |
| `test_stack.npy`            | LightGBM stack (s3e) |
| `final_test.npy`            | Final submission test array (pre-calibration) |

## Specialist test — shape `(1580, 3)`

| File | Description |
|---|---|
| `test_specialist.npy` | 3-class specialist predictions on the test set |

## Calibration & blend scalars — shape `(8,)`

| File | Description |
|---|---|
| `bias_s3e.npy` | Per-class log-bias vector (bootstrap-stable, 15 resamples); values: `[-0.003, -0.168, 0.892, 0.457, 0.010, 0.069, 0.070, -0.129]` |
| `blend_weights.npy` | Uniform placeholder weights `[0.125 × 8]`; real optimized weights are computed at runtime by `ensemble/blend.py` |

---

## Regenerating

To retrain all branches from scratch, follow `REPRODUCE.md`.
The frozen arrays are the source of truth for all reported metrics.
