# Detect the Disaster — Intra CUET ML Contest 2.0

[![CI](https://github.com/rafilovestosuffer/Cuet_ML_contest_2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/rafilovestosuffer/Cuet_ML_contest_2.0/actions/workflows/ci.yml)
[![Public LB macro-F1](https://img.shields.io/badge/public%20LB%20macro--F1-0.99826-success)](https://www.kaggle.com/competitions/intra-cuet-ml-contest-2)
[![OOF macro-F1](https://img.shields.io/badge/OOF%20macro--F1-0.9956-blue)](REPRODUCE.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Text-dominant **multimodal ensemble** for 8-class Bengali disaster
classification (Drought, Earthquake, Flood, Human Damage, Landslides,
Non Disaster, Tropical Storm, Wildfire). Each instance pairs a Bengali caption
(`context`) with an image; the metric is **macro-F1**.

> **OOF result (5-fold, primary metric): macro-F1 = 0.9956**
> **Public leaderboard: macro-F1 = 0.99826** (submission `submission_final.csv` —
> the bootstrap-stable bias submission with the Bengali keyword post-processing).

The **OOF** numbers in this README are computed by `scripts/reproduce_ensemble.py`
from frozen OOF arrays; nothing is hardcoded. The **public-leaderboard** number is
the Kaggle grader's score on the hidden test labels (it cannot be recomputed
locally) — it is reported here as measured on the contest, not derived in-repo.
See `PROVENANCE.md` for the verified-vs-reconstructed boundary.

---

## OOF results table

| Stage | OOF macro-F1 |
|---|---|
| BanglaBERT-base (text, multi-seed avg) | 0.96587 |
| BanglaBERT multilingual (text)         | 0.96656 |
| MuRIL-Large (text, best unimodal)      | 0.97556 |
| EVA-02-Large @448 (image)              | 0.96714 |
| EVA-02 × MuRIL cross-attention fusion  | 0.99212 |
| PL-MuRIL (pseudo-labeled re-finetune)  | 0.97525 |
| LightGBM stack (s3e)                   | 0.99351 |
| Dirichlet blend (50 NM restarts)       | 0.99495 |
| + α-mix with stack (α=0.855)           | 0.99495 |
| + bias single-shot (coord. ascent)     | 0.99590 |
| **+ bias bootstrap-stable (submitted)**| **0.99557** |

Per-class F1 (bootstrap-stable bias, reproduced from frozen OOF):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Drought        | 0.9987 | 0.9950 | 0.9969 |
| Earthquake     | 1.0000 | 0.9888 | 0.9943 |
| Flood          | 0.9962 | 0.9925 | 0.9944 |
| Human Damage   | 0.9962 | 0.9962 | 0.9962 |
| Landslides     | 0.9853 | 0.9988 | 0.9920 |
| Non Disaster   | 0.9987 | 0.9975 | 0.9981 |
| Tropical Storm | 0.9963 | 0.9988 | 0.9975 |
| Wildfire       | 0.9931 | 0.9972 | 0.9951 |
| **macro avg**  | **0.9956** | **0.9956** | **0.9956** |

---

## Method at a glance

1. **Frozen 5-fold split** (`data/folds/folds_canonical.csv`) shared by every
   branch so all OOF probabilities are stackable without leakage.
2. **Six base models** — text: BanglaBERT-base (seeds 123+456), BanglaBERT
   multilingual, MuRIL-Large; image: EVA-02-Large @448; multimodal: EVA-02×MuRIL
   cross-attention; semi-supervised: PL-MuRIL.
3. **Meta-learning** — LightGBM stacking → Dirichlet-restart convex blend
   (macro-F1 objective, 50 Nelder–Mead restarts) → α-mix (α=0.855).
4. **Per-class log-bias calibration** — coordinate ascent + 15-resample
   bootstrap median (the submitted, stable config).
5. **Deterministic Bengali rules** — emoji map + a confidence-gated keyword
   override with an *aftermath guard* (`[hazard]+por` = aftermath ≠ live event).

Blend weights from Dirichlet search:
`banglabert_base=0.069, banglabert_multi=0.004, muril_large=0.180,
eva02_large=0.443, fusion_eva_muril=0.233, pl_muril=0.071`

---

## Quickstart (reproduce in 5 commands)

```bash
git clone https://github.com/rafilovestosuffer/Cuet_ML_contest_2.0.git
cd Cuet_ML_contest_2.0
pip install -r requirements.txt          # or: conda env create -f environment.yml

# 1) download data & artifacts (see data/README.md and artifacts/README.md)
# 2) reproduce the ensemble result from frozen OOF arrays
PYTHONPATH=src python scripts/reproduce_ensemble.py --figures

# 3) run the test suite
PYTHONPATH=src pytest tests/ -v
```

Or via Makefile:
```bash
make reproduce    # runs reproduce_ensemble.py
make test         # runs pytest
make figures      # generates all paper figures into results/figures/
```

---

## Repository layout

```
notebooks/      Final_notebook.ipynb — the actual competition work (training + eval)
src/disaster/   the evaluation pipeline + model, refactored from the notebook
  models/       fusion.py — JointMM cross-attention model        [TRANSCRIBED]
  losses.py     soft_ce + cosine_warmup (training objective)     [TRANSCRIBED]
  ensemble/     blend, stacking, alpha_mix, bias_calibration     [VERIFIED]
  rules/        emoji_rules, bengali_keyword_guard               [VERIFIED]
  infer/        predict (full pipeline), make_submission         [VERIFIED]
  eda/          analysis, plots (evaluation study + figures)     [VERIFIED]
  data/         folds.py — rebuild the canonical split           [VERIFIED]
  labels.py     label space + submission-column handling         [VERIFIED]
scripts/        reproduce_ensemble.py
artifacts/      OOF/test .npy arrays (Git LFS / Release; not committed)
data/           download instructions (host dataset not committed)
results/        generated figures + tables (git-ignored)
tests/          fold integrity + F1 reproduction
```

This repo keeps **only code that was actually used**: the notebook, the `JointMM`
model, and the evaluation/ensemble logic. See `PROVENANCE.md` for the
verified-vs-transcribed boundary.

---

## Dataset & reproducibility

- **Dataset**: a custom multimodal Bengali disaster dataset assembled by the
  contest host for *Intra CUET ML Contest 2.0*. It is proprietary to the
  organizers and is **not redistributed here** (not even the fold file, which
  embeds the host's captions/labels). Obtain it from the official contest page;
  see `data/README.md`.
- **Reproducibility**: OOF CV result is the primary metric. The test arrays in
  `artifacts/` were produced by the same trained models. Post-processing rules
  are logged to `results/tables/applied_corrections.csv`. Regenerate the
  canonical folds locally with `make folds`.
- **Honesty**: text post-processing (emoji map + keyword guard) is labeled as
  post-processing in `PROVENANCE.md` — it is not part of the model pipeline.

## Final submission

The leaderboard submission (`submission_final.csv`, public macro-F1 **0.99826**)
is the bootstrap-stable bias prediction with the Bengali keyword post-processing
applied. Reproduce it from the frozen test arrays with:

```bash
make submission   # PYTHONPATH=src python -m disaster.infer.predict --test-csv ...
```

The submission's label column is **auto-detected** from the contest's
`sample_submission.csv` (it resolves to `category`; see the note in `PROVENANCE.md`
about the `category` vs `categry` discrepancy between the build prompt and the
actual contest file).

## Compliance with contest rules

Encoder-only CNN/transformer backbones; open-source pretrained weights only; no
vision–language or generative model; no external dataset; inference fits free
Kaggle/Colab limits.

## Citation

See `CITATION.cff`. Code is MIT-licensed; the dataset retains its own license.

## Contest link

[Intra CUET ML Contest 2.0](https://www.kaggle.com/competitions/intra-cuet-ml-contest-2)
