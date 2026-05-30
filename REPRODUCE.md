# Reproducing the paper

All commands assume `PYTHONPATH=src` and that OOF arrays are present in
`artifacts/` (see `artifacts/README.md`).

## One command: the ensemble result
```
PYTHONPATH=src python scripts/reproduce_ensemble.py
```
Prints individual OOF F1, the Dirichlet blend, the alpha-mix, and the
single-shot / bootstrap-stable per-class bias results (paper Tables I–III).

## Paper table / figure -> command
| Artifact | How to regenerate |
|---|---|
| Table I (individual OOF) | `scripts/reproduce_ensemble.py` (top block) |
| Table II (progression)   | `scripts/reproduce_ensemble.py` |
| Table III (per-class)    | `scripts/reproduce_ensemble.py` (classification report) |
| Fig. class distribution  | `disaster.eda.plots.fig_class_dist('data/raw/Disaster_train.csv')` |
| Fig. confusion matrix    | `disaster.eda.plots.fig_confusion('artifacts/oof_mix.npy', 'data/folds/folds_canonical.csv')` |
| Fig. fusion comparison   | `disaster.eda.plots.fig_fusion_compare({...})` |
| Submission CSV           | `disaster.infer.make_submission(...)` |

## Retrain a branch from scratch
```
PYTHONPATH=src python -m disaster.train.train_text   --config configs/text_muril_large.yaml
PYTHONPATH=src python -m disaster.train.train_vision --config configs/vision_eva02_large.yaml
PYTHONPATH=src python -m disaster.train.train_fusion --config configs/fusion_eva02_muril.yaml
```
(Training entry points are stubs to be ported from `notebooks/`.)
