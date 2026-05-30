import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from disaster.ensemble.alpha_mix import apply_alpha, search_alpha
from disaster.ensemble.bias_calibration import apply_bias, fit_bias_bootstrap, fit_bias_single
from disaster.ensemble.blend import apply_blend, search_blend_weights

pytestmark = [pytest.mark.needs_artifacts, pytest.mark.slow]

A = "artifacts/"
BASE = ["oof_banglabert_base", "oof_banglabert_multi", "oof_muril_large",
        "oof_eva02_large", "oof_fusion_eva_muril", "oof_pl_muril"]

PAPER_BLEND = 0.99495
PAPER_SINGLE = 0.99590
PAPER_STABLE = 0.99557
TOL = 1e-3


@pytest.fixture(scope="module")
def setup():
    y = pd.read_csv("data/folds/folds_canonical.csv",
                    encoding="utf-8-sig")["label"].values
    oofs = [np.load(A + n + ".npy") for n in BASE]
    stack = np.load(A + "oof_stack.npy")
    w, blend_f1 = search_blend_weights(oofs, y, n_restarts=50)
    blend = apply_blend(oofs, w)
    alpha, _ = search_alpha(blend, stack, y)
    mix = apply_alpha(blend, stack, alpha)
    return dict(y=y, w=w, blend=blend, blend_f1=blend_f1, alpha=alpha, mix=mix)


def test_blend_reproduces_paper(setup):
    assert abs(setup["blend_f1"] - PAPER_BLEND) < TOL, \
        f"blend {setup['blend_f1']:.5f} != {PAPER_BLEND}"


def test_alpha_in_expected_range(setup):
    assert 0.80 <= setup["alpha"] <= 0.90, f"alpha={setup['alpha']}"


def test_single_shot_bias_reproduces_paper(setup):
    _, f = fit_bias_single(setup["mix"], setup["y"])
    assert abs(f - PAPER_SINGLE) < TOL, f"single-shot {f:.5f} != {PAPER_SINGLE}"


def test_bootstrap_bias_reproduces_paper(setup):
    _, f = fit_bias_bootstrap(setup["mix"], setup["y"], n_boot=15)
    assert abs(f - PAPER_STABLE) < TOL, f"bootstrap {f:.5f} != {PAPER_STABLE}"


def test_final_macro_f1_is_at_least_995(setup):
    b, _ = fit_bias_bootstrap(setup["mix"], setup["y"], n_boot=15)
    pred = apply_bias(setup["mix"], b).argmax(1)
    final = f1_score(setup["y"], pred, average="macro")
    assert final >= 0.995, f"final macro-F1 {final:.5f} below 0.995"
