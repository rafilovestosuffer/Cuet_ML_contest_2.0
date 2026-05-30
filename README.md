# Detect the Disaster — Intra CUET ML Contest 2.0

Text-dominant **multimodal ensemble** for 8-class Bengali disaster
classification (Drought, Earthquake, Flood, Human Damage, Landslides,
Non Disaster, Tropical Storm, Wildfire). Each instance pairs a Bengali caption
(`context`) with an image; the metric is **macro-F1**.

> **Result:** 5-fold out-of-fold (OOF) **macro-F1 ≈ 0.9956**.
> Public leaderboard: _TODO (insert your verified LB score & rank)_.

## Method at a glance
1. **Frozen 5-fold split** (`data/folds/folds_canonical.csv`) shared by every
   branch so all OOF probabilities are stackable without leakage.
2. **Six base models** — text: BanglaBERT-base, BanglaBERT multilingual,
   MuRIL-Large; image: EVA-02-Large; multimodal: EVA-02×MuRIL cross-attention;
   semi-supervised: PL-MuRIL.
3. **Meta-learning** — LightGBM stacking → Dirichlet-restart convex blend
   (macro-F1 objective) → α-mix.
4. **Per-class log-bias calibration** — coordinate ascent + 15-resample
   bootstrap median (the submitted, stable config).
5. **Deterministic Bengali rules** — emoji map + a confidence-gated keyword
   override with an *aftermath guard* (`[hazard]+por` = after the event ≠ event).

| Stage | OOF macro-F1 |
|---|---|
| MuRIL-Large (best unimodal text) | 0.97556 |
| EVA-02-Large (best image)        | 0.96714 |
| EVA-02 × MuRIL fusion            | 0.99212 |
| LightGBM stack                   | 0.99351 |
| Dirichlet blend                  | 0.99495 |
| + per-class bias (submitted)     | **≈ 0.9956** |

(Numbers above are produced by `scripts/reproduce_ensemble.py`, not hardcoded.)

## Quickstart
```bash
git clone https://github.com/rafilovestosuffer/Cuet_ML_contest_2.0.git
cd Cuet_ML_contest_2.0
pip install -r requirements.txt          # or: conda env create -f environment.yml

# 1) get the data (see data/README.md) and build folds
make folds

# 2) reproduce the ensemble result from frozen OOF arrays
make reproduce

# 3) run the tests
make test
```

## Repository layout
```
configs/      one YAML per experiment (hyperparameters)
data/         download instructions + frozen folds (no raw data committed)
src/disaster/ package: data, models, train, ensemble, rules, infer, eda
scripts/      reproduce_ensemble.py and CLI wrappers
notebooks/    original Kaggle notebooks (provenance)
artifacts/    OOF/test probability arrays (Git LFS / Release; see its README)
results/      generated figures and tables
paper/        IEEE manuscript + figures
tests/        fold integrity + artifact reproduction
```

## Reproducibility & honesty
- Every reported number is regenerated from artifacts; nothing is hardcoded.
- Dataset provenance is documented in `data/README.md` (BanglaCalamityMMD; the
  contest test set = the dataset's test+validation splits). The OOF CV result is
  the primary metric. Post-processing rules are logged to
  `results/tables/applied_corrections.csv`.

## Compliance with contest rules
Encoder-only CNN/transformer backbones; open-source pretrained weights only; no
vision–language or generative model; no external dataset; inference fits free
Kaggle/Colab limits; submission preserves the `categry` column spelling.

## Citation
See `CITATION.cff`. Code is MIT-licensed; the dataset retains its own license.
