"""LightGBM stacking meta-learner over concatenated OOF probability vectors.

CV-safe: trained on OOF predictions only. At inference the same trained models
are applied to the base models' test predictions.
"""
import numpy as np

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


def train_stacker(oof_features: np.ndarray, y: np.ndarray, folds: np.ndarray,
                  params: dict | None = None):
    """Out-of-fold stacking. Returns (oof_pred_proba, list_of_models)."""
    if not _HAS_LGB:
        raise ImportError("lightgbm is required for the stacker (pip install lightgbm).")
    params = params or dict(
        objective="multiclass", num_class=8, learning_rate=0.03,
        num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, min_data_in_leaf=40, verbose=-1, seed=42,
    )
    n, k = len(y), 8
    oof = np.zeros((n, k))
    models = []
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        dtr = lgb.Dataset(oof_features[tr], label=y[tr])
        m = lgb.train(params, dtr, num_boost_round=400)
        oof[va] = m.predict(oof_features[va])
        models.append(m)
    return oof, models
