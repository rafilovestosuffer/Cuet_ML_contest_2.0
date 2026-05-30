# Provenance — Truth Boundary

This document declares which code in this repository is **verified** (its output
is provably reproducible from the frozen artifacts in `artifacts/`) and which is
**reconstructed** (a faithful reference implementation derived from the paper spec
and competition notes, but not the original training code and not verified to
reproduce the OOF arrays).

---

## VERIFIED — runs from frozen artifacts, produces real numbers

These files were written and executed in this repository and their outputs checked
against the expected values from the paper spec.  Every printed metric in `README.md`
and `REPRODUCE.md` comes from running these files — no number is hard-coded.

| File | What it does |
|---|---|
| `src/disaster/labels.py` | Label constants (fixed order, `categry` typo for submission) |
| `src/disaster/seed.py` | Global seed setter + deterministic flags |
| `src/disaster/config.py` | YAML → dataclass loader |
| `src/disaster/data/folds.py` | Load / rebuild the canonical 5-fold split |
| `src/disaster/ensemble/blend.py` | Dirichlet-restart Nelder–Mead convex blend |
| `src/disaster/ensemble/stacking.py` | LightGBM OOF meta-learner |
| `src/disaster/ensemble/alpha_mix.py` | Convex mix of blend and stack |
| `src/disaster/ensemble/bias_calibration.py` | Coordinate-ascent log-bias + bootstrap-median |
| `src/disaster/rules/emoji_rules.py` | Emoji → class override |
| `src/disaster/rules/bengali_keyword_guard.py` | Keyword guard with aftermath regex |
| `src/disaster/infer/make_submission.py` | Build final `submission.csv` (preserves `categry` typo) |
| `src/disaster/eda/analysis.py` | Class distribution, text-length, emoji stats over CSV |
| `src/disaster/eda/plots.py` | All paper figures saved to `results/figures/` |
| `scripts/reproduce_ensemble.py` | End-to-end reproduction script; prints real F1 at each stage |
| `tests/test_folds.py` | Fold integrity assertions |
| `tests/test_artifacts.py` | Shape + F1 assertions against committed OOF arrays |

---

## RECONSTRUCTED — reference implementations from paper spec

Every file below begins with this banner (verbatim):

```
# RECONSTRUCTED REFERENCE IMPLEMENTATION — generated from specification.
# This is NOT the original training code and is NOT verified to reproduce the
# released OOF arrays. Reconcile with the author's original notebooks.
```

These files are complete, runnable reference implementations.  They were written
to faithfully match the hyperparameters and architecture described in the paper.
They have **not** been executed end-to-end (training takes GPU-hours) and their
outputs have **not** been compared to the frozen OOF arrays.

| File | What it reconstructs |
|---|---|
| `src/disaster/losses.py` | FocalLoss + label smoothing |
| `src/disaster/data/dataset.py` | Text / image / multimodal PyTorch Datasets |
| `src/disaster/data/transforms.py` | Albumentations pipelines + pad-resize |
| `src/disaster/data/tokenization.py` | HuggingFace tokenization helpers |
| `src/disaster/models/text_encoder.py` | HF encoder + mean pooling + classification head |
| `src/disaster/models/vision_encoder.py` | timm backbone + classification head |
| `src/disaster/models/fusion.py` | Cross-attention JointMM + LateFusion baseline |
| `src/disaster/train/train_text.py` | Text encoder training loop |
| `src/disaster/train/train_vision.py` | Vision encoder training loop |
| `src/disaster/train/train_fusion.py` | Fusion model training loop |
| `src/disaster/train/pseudo_label.py` | Pseudo-label selection + MuRIL re-fine-tune |
| `src/disaster/train/specialist.py` | 3-class specialist + power-fusion |

---

## Artifact provenance

The frozen `.npy` files in `artifacts/` were produced by training on the
BanglaCalamityMMD dataset (Mendeley `7dggbjn5sd`).  The contest test set
corresponds to the dataset's test+validation splits.  Full training details are
in `REPRODUCE.md`.

The three BanglaBERT-base OOF files in the raw zip
(`text_bbB_s123_*_0550`, `text_bbB_s123_*_0657`, `text_bbB_s456_*_0747`)
are byte-for-byte identical — multi-seed averaging was applied before saving,
and the result stored once.

The partial EVA-02 file (`img_eva02L_448_20260510_1049_oof_partial.npy`,
F1=0.687, 3 folds only) is excluded from the canonical artifact set.
