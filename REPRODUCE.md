# Reproduce — Paper Tables & Figures

Every number and figure in the paper can be regenerated from the frozen OOF
arrays in `artifacts/` and the canonical fold split in `data/folds/`.
No raw data is required to reproduce the ensemble metrics; raw data is only
needed to retrain the base models.

All commands assume the repo root as the working directory.

---

## Prerequisites

```bash
git clone https://github.com/rafilovestosuffer/Cuet_ML_contest_2.0.git
cd Cuet_ML_contest_2.0
pip install -r requirements.txt

# Download OOF/test arrays (not committed; see artifacts/README.md):
#   Option A: GitHub Release → place .npy files in artifacts/
#   Option B: Git LFS pull
```

---

## Paper table / figure → command

| Paper item | Command | Output |
|---|---|---|
| Table: individual OOF F1 | `PYTHONPATH=src python scripts/reproduce_ensemble.py` | printed to stdout |
| Table: blend / α-mix / bias F1 | same | printed to stdout |
| Table: per-class F1 report | same | printed to stdout |
| Fig: class distribution | `PYTHONPATH=src python scripts/reproduce_ensemble.py --figures` | `results/figures/fig_class_dist.pdf` |
| Fig: OOF waterfall | same | `results/figures/fig_oof_waterfall.pdf` |
| Fig: per-class F1 | same | `results/figures/fig_per_class_f1.pdf` |
| Fig: confusion matrix | same | `results/figures/fig_confusion.pdf` |
| Fig: fusion strategy comparison | same | `results/figures/fig_fusion_compare.pdf` |
| Fig: per-class log-bias vector | same | `results/figures/fig_bias_vector.pdf` |
| Fig: modality error overlap | same | `results/figures/fig_error_overlap.pdf` |
| Submission CSV (from test arrays) | `PYTHONPATH=src python -m disaster.infer.predict --test-csv data/Test/test.csv` | `results/tables/submission.csv` |
| Rule corrections log | same (side-effect) | `results/tables/applied_corrections.csv` |

Or using the Makefile:

```bash
make reproduce   # ensemble metrics
make figures     # all paper figures
make test        # fold integrity + F1 reproduction
make submission  # submission.csv (requires data/Test/test.csv)
```

---

## Expected output of `reproduce_ensemble.py`

```
Individual OOF macro-F1:
  oof_banglabert_base         0.96587
  oof_banglabert_multi        0.96656
  oof_muril_large             0.97556
  oof_eva02_large             0.96714
  oof_fusion_eva_muril        0.99212
  oof_pl_muril                0.97525
  oof_stack                   0.99351

Dirichlet blend              : 0.99495
  weights: {'banglabert_base': 0.069, 'banglabert_multi': 0.004,
            'muril_large': 0.180, 'eva02_large': 0.443,
            'fusion_eva_muril': 0.233, 'pl_muril': 0.071}
alpha-mix (alpha=0.855)       : 0.99495
+ bias single-shot           : 0.99590
+ bias bootstrap-stable      : 0.99557  <-- submitted

Per-class F1 (bootstrap-stable bias):
                precision    recall  f1-score   support
       Drought     0.9987    0.9950    0.9969       800
    Earthquake     1.0000    0.9888    0.9943       800
         Flood     0.9962    0.9925    0.9944       800
  Human Damage     0.9962    0.9962    0.9962       800
    Landslides     0.9853    0.9988    0.9920       803
  Non Disaster     0.9987    0.9975    0.9981       800
Tropical Storm     0.9963    0.9988    0.9975       800
      Wildfire     0.9931    0.9972    0.9951       720
     macro avg     0.9956    0.9956    0.9956      6323
```

---

## Training

Training was done in **`notebooks/Final_notebook.ipynb`** on Kaggle (Tesla T4) —
that notebook is the source of truth for how the base models and the `JointMM`
fusion model were trained. The model definition (`src/disaster/models/fusion.py`)
and the training objective (`src/disaster/losses.py`) are extracted from it as
reference `.py` files; the full training loop lives in the notebook.

This repository reproduces the **evaluation** half (the part that turns the
frozen OOF/test arrays into the final score and submission) — see the table above.

```bash
make folds       # rebuild the canonical 5-fold split from the host train CSV
make stack       # rebuild the LightGBM stack OOF from the base OOF arrays
make reproduce   # reproduce the ensemble macro-F1 from frozen OOF arrays
```
