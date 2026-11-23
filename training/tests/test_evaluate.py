"""T5: metrics, including the thesis' DET-area "AUC"."""
import numpy as np
import pytest
from sklearn.metrics import auc, det_curve, roc_auc_score

from lasdaicvi_kws import evaluate


def toy(n=300, k=4, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, k, n)
    logits = r.normal(size=(n, k))
    logits[np.arange(n), y] += 1.5
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return y, np.eye(k)[y], p


# Golden DET areas for `toy()`: computed with the thesis' `plot_roc_curve_fn_fp`
# (`det_curve` + `auc(fpr, fnr)`, per class and micro-averaged).
GOLDEN_DET_PER_CLASS = [0.10370788631658197, 0.05578093306288032, 0.07937378167641326,
                        0.06793206793206794]
GOLDEN_DET_MICRO = 0.07769999999999999


def test_det_area_matches_golden_values():
    _, onehot, p = toy()
    assert evaluate.det_area_micro(onehot, p) == pytest.approx(GOLDEN_DET_MICRO)
    assert evaluate.det_area_per_class(onehot, p) == pytest.approx(GOLDEN_DET_PER_CLASS)


def test_det_area_equals_sklearn_composition():
    _, onehot, p = toy()
    fpr, fnr, _ = det_curve(onehot.ravel(), p.ravel())
    assert evaluate.det_area_micro(onehot, p) == pytest.approx(auc(fpr, fnr))


def test_det_area_handles_degenerate_curves():
    onehot = np.eye(3)[[0, 0, 1, 1]]  # class 2 never occurs: its DET curve is undefined
    p = np.full((4, 3), 1 / 3)
    areas = evaluate.det_area_per_class(onehot, p)
    assert not np.isnan(areas[:2]).any()
    assert np.isnan(areas[2])


def test_roc_micro_and_macro_are_labelled_correctly():
    y, onehot, p = toy()
    out = evaluate.roc_auc(onehot, p)
    assert out["micro"] == pytest.approx(roc_auc_score(onehot, p, average="micro"), abs=1e-6)
    assert out["macro"] == pytest.approx(roc_auc_score(onehot, p, average="macro"), abs=2e-3)
    assert out["micro"] != out["macro"]


def test_evaluate_predictions_bundle():
    y, _, p = toy()
    res = evaluate.evaluate_predictions(y, p, ["a", "b", "c", "d"])
    assert res["accuracy"] == pytest.approx((p.argmax(1) == y).mean())
    assert np.asarray(res["confusion_matrix"]).sum() == len(y)
    assert set(res["report"]) >= {"a", "accuracy", "macro avg"}
    assert {"roc_auc_micro", "roc_auc_macro", "det_area_micro", "det_area_per_class"} <= set(res)
