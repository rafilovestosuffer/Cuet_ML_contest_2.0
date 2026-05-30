# Provenance — Truth Boundary

This repository keeps a strict, visible boundary between code whose output is
**provable from the frozen artifacts** and code that is a **reference
implementation**. Three tiers:

1. **VERIFIED** — runs in this repo on the frozen `.npy` arrays and reproduces
   the reported numbers. Every metric in `README.md` / `REPRODUCE.md` comes from
   running these. Covered by the test suite.
2. **TRANSCRIBED** — copied faithfully from `notebooks/Final_notebook.ipynb`
   (the author's original code). Not re-executed in CI (needs raw images / GPU),
   but it is the real code, not a guess.
3. **RECONSTRUCTED** — written from the paper/spec because the original is not in
   the notebook (the training loops that produced the OOF arrays are not in the
   shipped notebook). Each such file begins with the banner below, verbatim.

```
# RECONSTRUCTED REFERENCE IMPLEMENTATION — generated from specification.
# This is NOT the original training code and is NOT verified to reproduce the
# released OOF arrays. Reconcile with the author's original notebooks.
```

---

## VERIFIED

| File | What it does |
|---|---|
| `src/disaster/labels.py` | Label constants + submission-column handling |
| `src/disaster/seed.py` | Global seed + deterministic flags |
| `src/disaster/config.py` | YAML loader |
| `src/disaster/data/folds.py` | Load / rebuild the canonical 5-fold split |
| `src/disaster/ensemble/blend.py` | Dirichlet-restart Nelder–Mead blend (notebook-faithful RNG) |
| `src/disaster/ensemble/stacking.py` | LightGBM OOF meta-learner |
| `src/disaster/ensemble/alpha_mix.py` | Convex mix of blend and stack |
| `src/disaster/ensemble/bias_calibration.py` | Coordinate-ascent log-bias + bootstrap-median |
| `src/disaster/rules/emoji_rules.py` | Emoji → class override |
| `src/disaster/rules/bengali_keyword_guard.py` | Confidence-gated keyword guard + aftermath regex |
| `src/disaster/infer/predict.py` | Full test-set pipeline → submission.csv |
| `src/disaster/infer/make_submission.py` | Submission writer (auto-detects label column) |
| `src/disaster/eda/analysis.py`, `plots.py` | EDA + all paper figures |
| `src/disaster/train/specialist.py` → `apply_spec`, `search_beta` | Power-fusion (array math) |
| `src/disaster/train/pseudo_label.py` → `select_pseudo_labels` | Pseudo-label selection (array logic) |
| `scripts/reproduce_ensemble.py` | End-to-end reproduction; prints real F1 |
| `tests/` | Fold integrity, artifact shapes/F1, and full meta-layer reproduction of **0.9956** |

**Reproduced values** (from `reproduce_ensemble.py`, matching the notebook exactly):
blend 0.99495 · α-mix (α=0.855) 0.99495 · bias single-shot 0.99590 ·
bias bootstrap-stable **0.99557** · final report macro-F1 **0.9956**.

---

## TRANSCRIBED (from `notebooks/Final_notebook.ipynb`)

| File | Original notebook element |
|---|---|
| `src/disaster/models/fusion.py` → `JointMM` | the cross-attention fusion class, verbatim |
| `src/disaster/data/transforms.py` | `pad_resize` + the Albumentations train/val pipelines |
| `src/disaster/data/dataset.py` → `MMDataset`, `resolve_img` | the multimodal dataset + image resolver |
| `src/disaster/data/tokenization.py` | the tokenizer call (max_len 256) |
| `src/disaster/losses.py` → `soft_ce`, `cosine_warmup` | the label-smoothed soft-CE and LR schedule |

---

## RECONSTRUCTED (from spec — banner-marked)

| File | Reconstructs |
|---|---|
| `src/disaster/losses.py` → `FocalLabelSmoothingLoss` | focal + label smoothing (paper text; notebook shows `soft_ce`) |
| `src/disaster/models/text_encoder.py` | HF encoder + mean pool + head |
| `src/disaster/models/vision_encoder.py` | timm backbone + head |
| `src/disaster/models/fusion.py` → `LateFusion` | late-fusion ablation baseline |
| `src/disaster/train/_common.py` | shared 5-fold OOF training driver |
| `src/disaster/train/train_text.py` | text branch training (multi-seed) |
| `src/disaster/train/train_vision.py` | vision branch training |
| `src/disaster/train/train_fusion.py` | fusion (JointMM) training |
| `src/disaster/train/specialist.py` (training half) | 3-class specialist training |
| `src/disaster/train/pseudo_label.py` (re-finetune half) | PL-MuRIL re-finetune |

---

## Notes & honest caveats

- **Submission column.** The build prompt (`CLAUDE.md`) states the contest column
  is the misspelled `categry`. The notebook instead auto-detects it from the
  contest's `sample_submission.csv` and resolves it to the correctly-spelled
  `category`. The two sources disagree, so `make_submission` **auto-detects** from
  `sample_submission.csv` when given one (the safe, notebook-faithful behaviour),
  falling back to `labels.SUBMISSION_LABEL_COLUMN` otherwise.
- **Backbone for the final fusion run.** The notebook's `JM_CFG` uses
  **ConvNeXt V2-base @384 × MuRIL** ("faster than EVA"); the released
  `oof_fusion_eva_muril` artifact is the **EVA-02 × MuRIL** variant. Both are
  covered by `JointMM` via config (`configs/fusion_*_muril.yaml`).
- **`soft_ce` vs focal loss.** The notebook's shown objective is label-smoothed
  soft cross-entropy. The paper describes focal + label smoothing. Both are
  provided; `losses.py` documents which is which.
- **Blend RNG matters.** `blend.py` uses the notebook's legacy
  `np.random.seed(trial)` seeding. A different RNG reaches the same 0.99495 blend
  but a different weight vector, shifting the final bias result by ~2e-4. Do not
  change it without re-checking `REPRODUCE.md`.

## Artifact provenance

Frozen `.npy` arrays in `artifacts/` were produced on the contest's **custom
host-assembled** Bengali disaster dataset (proprietary to the organizers; not
redistributed here — see `data/README.md`). The three BanglaBERT-base OOF files
in the raw zip are byte-identical (multi-seed averaging applied before saving).
The partial EVA-02 file (F1=0.687, 3 folds) is excluded from the canonical set.

The fold file (`data/folds/folds_canonical.csv`) embeds the host's captions and
labels, so it is **git-ignored** and regenerated locally via `make folds` rather
than committed.
