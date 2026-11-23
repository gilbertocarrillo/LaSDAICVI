"""Metrics: accuracy, report, confusion matrix, ROC AUCs and the thesis' DET-area "AUC".

The thesis' "AUC" is `auc(fpr, fnr)` of `sklearn.metrics.det_curve` (the area under the
detection-error-tradeoff curve), micro-averaged over the flattened one-hot labels, as printed
by the original `plot_roc_curve_fn_fp`. It is NOT a ROC AUC.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (auc, classification_report, confusion_matrix, det_curve,
                             roc_curve)


def accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Top-1 accuracy of softmax outputs `probs` against integer labels `y_true`."""
    return float((np.asarray(probs).argmax(axis=1) == np.asarray(y_true)).mean())


def _det_area(y: np.ndarray, scores: np.ndarray) -> float:
    """`auc(fpr, fnr)` of the DET curve."""
    fpr, fnr, _ = det_curve(y, scores)
    if fnr.shape[0] == 1 or fpr.shape[0] == 1:  # same guard as the thesis code: a one-point curve has no area
        fnr, fpr = np.append(fnr, fnr), np.append(fpr, fpr)
    return float(auc(fpr, fnr))


def det_area_micro(onehot: np.ndarray, scores: np.ndarray) -> float:
    """Area under the DET curve of the flattened one-hot labels (the thesis' reported AUC)."""
    return _det_area(np.ravel(onehot), np.ravel(scores))


def det_area_per_class(onehot: np.ndarray, scores: np.ndarray) -> list[float]:
    """DET area per class; NaN for a class absent from the labels (the curve is undefined)."""
    areas = []
    for i in range(onehot.shape[1]):
        try:
            areas.append(_det_area(onehot[:, i], scores[:, i]))
        except ValueError:
            areas.append(float("nan"))
    return areas


def roc_auc(onehot: np.ndarray, scores: np.ndarray) -> dict:
    """Micro and macro ROC AUC; macro interpolates the per-class curves as the original did.

    Classes with a single outcome in the labels have no ROC curve and are left out of the macro.
    """
    fpr, tpr, _ = roc_curve(np.ravel(onehot), np.ravel(scores))
    curves = []
    for i in range(onehot.shape[1]):
        column = onehot[:, i]
        if 0 < column.sum() < len(column):
            f, t, _ = roc_curve(column, scores[:, i])
            curves.append((f, t))
    all_fpr = np.unique(np.concatenate([f for f, _ in curves]))
    mean_tpr = sum(np.interp(all_fpr, f, t) for f, t in curves) / len(curves)
    return {"micro": float(auc(fpr, tpr)), "macro": float(auc(all_fpr, mean_tpr))}


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    """All metrics for integer labels `y_true` and softmax outputs `probs`."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    k = len(class_names)
    onehot = np.eye(k)[y_true]
    pred = probs.argmax(axis=1)
    roc = roc_auc(onehot, probs)
    return {
        "accuracy": accuracy(y_true, probs),
        "roc_auc_micro": roc["micro"],
        "roc_auc_macro": roc["macro"],
        "det_area_micro": det_area_micro(onehot, probs),
        "det_area_per_class": det_area_per_class(onehot, probs),
        "report": classification_report(y_true, pred, labels=list(range(k)),
                                        target_names=class_names, output_dict=True,
                                        zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(k))).tolist(),
    }


def save_plots(result: dict, class_names: list[str], out_dir, prefix: str) -> None:
    """Write `confusion_<prefix>.png` (row-normalised heatmap with raw counts) into `out_dir`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(result["confusion_matrix"], dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(shown, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=60, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if shown[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{prefix} (accuracy {result['accuracy']:.4f})")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{prefix}.png", dpi=120)
    plt.close(fig)
