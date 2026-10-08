"""Evaluation metrics and threshold analysis for credit scoring."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov separation between the score distributions of the
    two classes - the classic credit-scoring discrimination metric."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = np.sort(y_score[y_true == 1])
    neg = np.sort(y_score[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # evaluate ECDF difference at all observed scores
    grid = np.unique(np.concatenate([pos, neg]))
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def discrimination_metrics(y_true, y_score) -> dict[str, float]:
    """ROC-AUC, PR-AUC, Brier score and KS statistic."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
    }


def confusion_at_threshold(y_true, y_score, threshold: float) -> dict[str, int]:
    """Confusion matrix entries at a decision threshold on the score."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def expected_cost(entries: dict[str, int], fn_cost: float, fp_cost: float) -> float:
    """Business cost of a confusion matrix under asymmetric error costs.

    A missed defaulter (FN) costs ``fn_cost``; a wrongly declined good client
    (FP) costs ``fp_cost``. This is the quantity threshold selection
    minimizes - NOT accuracy, which treats both errors as equal.
    """
    return float(entries["fn"] * fn_cost + entries["fp"] * fp_cost)


def threshold_sweep(
    y_true,
    y_score,
    fn_cost: float,
    fp_cost: float,
    n_thresholds: int = 81,
) -> list[dict]:
    """Evaluate confusion/cost/metrics over a threshold grid."""
    thresholds = np.linspace(0.05, 0.85, n_thresholds)
    rows = []
    for t in thresholds:
        e = confusion_at_threshold(y_true, y_score, t)
        total = e["tn"] + e["fp"] + e["fn"] + e["tp"]
        rows.append(
            {
                "threshold": float(t),
                **e,
                "accuracy": (e["tn"] + e["tp"]) / total,
                "expected_cost": expected_cost(e, fn_cost, fp_cost),
            }
        )
    return rows


def optimal_threshold(rows: list[dict]) -> float:
    """The threshold minimizing expected cost (ties -> lowest threshold)."""
    best = min(rows, key=lambda r: (r["expected_cost"], r["threshold"]))
    return float(best["threshold"])
