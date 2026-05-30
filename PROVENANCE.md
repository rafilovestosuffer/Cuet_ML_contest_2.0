# Provenance — Truth Boundary

This repository contains **only code that was actually used in the competition**:
the original notebook, the model it trained, and the evaluation/ensemble logic
that produced the submission. Earlier versions of this repo also shipped a
generic training/data framework reconstructed from a spec — that has been
**removed** because it was never part of the real solution.

Two tiers remain:

1. **VERIFIED** — runs in this repo on the frozen `.npy` arrays and reproduces the
   reported numbers. Every metric in `README.md` / `REPRODUCE.md` comes from
   running these. Covered by the test suite.
2. **TRANSCRIBED** — copied faithfully from `notebooks/Final_notebook.ipynb` (the
   author's original code). Not re-executed in CI (training needs raw images +
   GPU), but it is the real code, not a reconstruction.

---

## VERIFIED

| File | What it does |
|---|---|
| `src/disaster/labels.py` | Label constants + submission-column handling |
| `src/disaster/data/folds.py` | Rebuild the canonical 5-fold split |
| `src/disaster/ensemble/blend.py` | Dirichlet-restart Nelder–Mead blend (notebook-faithful RNG) |
| `src/disaster/ensemble/stacking.py` | LightGBM OOF meta-learner |
| `src/disaster/ensemble/alpha_mix.py` | Convex mix of blend and stack |
| `src/disaster/ensemble/bias_calibration.py` | Coordinate-ascent log-bias + bootstrap-median |
| `src/disaster/rules/emoji_rules.py` | Emoji → class override |
| `src/disaster/rules/bengali_keyword_guard.py` | Confidence-gated keyword guard + aftermath regex |
| `src/disaster/infer/predict.py` | Full test-set pipeline → submission.csv |
| `src/disaster/infer/make_submission.py` | Submission writer (auto-detects label column) |
| `src/disaster/eda/analysis.py`, `plots.py` | Evaluation study + figures |
| `scripts/reproduce_ensemble.py` | End-to-end reproduction; prints real F1 |
| `tests/` | Fold integrity, artifact shapes/F1, full meta-layer reproduction of **0.9956** |

**Reproduced values** (from `reproduce_ensemble.py`, matching the notebook exactly):
blend 0.99495 · α-mix (α=0.855) 0.99495 · bias single-shot 0.99590 ·
bias bootstrap-stable **0.99557** · final OOF macro-F1 **0.9956**.
Public leaderboard (Kaggle, hidden labels): **0.99826**.

---

## TRANSCRIBED (from `notebooks/Final_notebook.ipynb`)

| File | Original notebook element |
|---|---|
| `src/disaster/models/fusion.py` → `JointMM` | the cross-attention fusion model, verbatim |
| `src/disaster/losses.py` → `soft_ce`, `cosine_warmup` | the label-smoothed objective + LR schedule |

The model definitions, training recipe, and the full pipeline are also visible in
context in `notebooks/Final_notebook.ipynb`.

---

## Notes & honest caveats

- **The real work is the notebook.** Training was done in
  `notebooks/Final_notebook.ipynb` on Kaggle (Tesla T4). This repo refactors the
  *evaluation* half (cells 6–13: load OOF/test arrays → blend → α-mix → per-class
  bias → text rules → submission) into a tested, reproducible Python package, and
  keeps the model definition (`JointMM`) and training objective as `.py` for
  reference.
- **Submission column.** The build prompt stated the contest column was the
  misspelled `categry`; the notebook auto-detects it from `sample_submission.csv`
  and it resolves to `category`. `make_submission` auto-detects (notebook
  behaviour), falling back to `labels.SUBMISSION_LABEL_COLUMN` otherwise.
- **Blend RNG matters.** `blend.py` uses the notebook's legacy
  `np.random.seed(trial)` seeding. A different RNG reaches the same 0.99495 blend
  but a different weight vector, shifting the final bias result by ~2e-4. Do not
  change it without re-checking `REPRODUCE.md`.

## Dataset & artifact provenance

The dataset is a **custom dataset assembled by the contest host** (proprietary;
not redistributed — see `data/README.md`). Frozen `.npy` arrays in `artifacts/`
are the OOF/test probabilities of the trained branches; they are distributed via
Git LFS / Release, not committed. The fold file embeds the host's captions/labels
and is git-ignored — regenerate it locally with `make folds`.
