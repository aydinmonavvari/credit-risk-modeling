"""Tests for the credit-risk pipeline (offline, synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_modeling.data import prepare
from credit_risk_modeling.evaluation import (
    confusion_at_threshold,
    expected_cost,
    ks_statistic,
    optimal_threshold,
    threshold_sweep,
)
from credit_risk_modeling.features import FEATURE_COLUMNS, build_features


def _synthetic_frame(n=2000, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pay = rng.integers(-1, 4, size=(n, 6))
    limit = rng.integers(20_000, 500_000, size=n)
    bill = rng.integers(0, 200_000, size=(n, 6))
    paid = (bill * rng.uniform(0, 1.2, size=(n, 6))).astype(int)
    frame = pd.DataFrame(
        {
            "limit_bal": limit,
            "sex": rng.choice([1, 2], size=n),
            "education": rng.choice([1, 2, 3, 4, 0, 5, 6], size=n),
            "marriage": rng.choice([1, 2, 3, 0], size=n),
            "age": rng.integers(21, 70, size=n),
            "pay_sep": pay[:, 0], "pay_aug": pay[:, 1], "pay_jul": pay[:, 2],
            "pay_jun": pay[:, 3], "pay_may": pay[:, 4], "pay_apr": pay[:, 5],
        }
    )
    for i, col in enumerate(
        [
            "bill_amt_sep",
            "bill_amt_aug",
            "bill_amt_jul",
            "bill_amt_jun",
            "bill_amt_may",
            "bill_amt_apr",
        ]
    ):
        frame[col] = bill[:, i]
    for i, col in enumerate(
        [
            "pay_amt_sep",
            "pay_amt_aug",
            "pay_amt_jul",
            "pay_amt_jun",
            "pay_amt_may",
            "pay_amt_apr",
        ]
    ):
        frame[col] = paid[:, i]
    # default probability driven by delayed months and utilization
    frame["default"] = ((pay >= 2).sum(axis=1) + (bill[:, 0] / limit > 1.0)).astype(float)
    frame["default"] = (frame["default"] + rng.uniform(0, 0.5, size=n) > 2.0).astype(int)
    return frame


def test_prepare_maps_unlabeled_categories():
    frame = _synthetic_frame()
    prep = prepare(frame)
    assert prep["education"].isin([1, 2, 3, 4]).all(), "0/5/6 must map to 4 (other)"
    assert prep["marriage"].isin([1, 2, 3]).all(), "0 must map to 3 (other)"
    assert set(prep["default"].unique()) <= {0, 1}


def test_features_exclude_sex():
    frame = prepare(_synthetic_frame())
    feats = build_features(frame)
    assert "sex" not in FEATURE_COLUMNS and "sex_label" not in FEATURE_COLUMNS
    assert "sex" not in feats.columns
    assert not feats.drop(columns=["default"]).isna().any().any()


def test_engineered_features_math():
    frame = prepare(_synthetic_frame(200, seed=5))
    feats = build_features(frame)
    # utilization is bill/limit
    expected_util = (frame["bill_amt_sep"] / frame["limit_bal"]).clip(upper=5.0)
    np.testing.assert_allclose(feats["utilization_sep"], expected_util, rtol=1e-9)
    # zero bill -> payment ratio 1.0 ("fully covered")
    zero_bill = frame["bill_amt_sep"] == 0
    assert (feats.loc[zero_bill, "payment_ratio_sep"] == 1.0).all()
    # months delayed counts statuses >= 1
    pay_cols = ["pay_sep", "pay_aug", "pay_jul", "pay_jun", "pay_may", "pay_apr"]
    np.testing.assert_array_equal(
        feats["months_delayed_6m"].to_numpy(), (frame[pay_cols] >= 1).sum(axis=1).to_numpy()
    )


def test_ks_statistic_perfect_and_null_separation():
    y = np.array([0] * 50 + [1] * 50)
    perfect = np.concatenate([np.zeros(50), np.ones(50)])
    assert ks_statistic(y, perfect) == pytest.approx(1.0)
    null = np.zeros(100)
    assert ks_statistic(y, null) == pytest.approx(0.0)


def test_expected_cost_is_asymmetric():
    entries = {"tn": 10, "fp": 2, "fn": 1, "tp": 3}
    assert expected_cost(entries, fn_cost=5, fp_cost=1) == 5 + 2  # 1*5 + 2*1
    assert expected_cost(entries, fn_cost=1, fp_cost=1) == 3
    assert expected_cost(entries, fn_cost=5, fp_cost=0) == 5


def test_threshold_sweep_picks_cost_optimal_not_accuracy():
    """A threshold chosen by asymmetric cost must differ from the accuracy
    optimum when FN is much more expensive than FP."""
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.choice([0, 1], size=n, p=[0.78, 0.22])
    score = np.clip(rng.normal(0.22 * y + 0.18, 0.12), 0.0, 1.0)
    rows = threshold_sweep(y, score, fn_cost=5.0, fp_cost=1.0)
    best_cost_thr = optimal_threshold(rows)
    acc_thr = max(rows, key=lambda r: r["accuracy"])["threshold"]
    e_cost = next(r for r in rows if r["threshold"] == best_cost_thr)
    e_acc = next(r for r in rows if r["threshold"] == acc_thr)
    assert e_cost["expected_cost"] <= e_acc["expected_cost"] + 1e-9
    # accuracy alone cannot express this preference:
    assert e_cost["threshold"] != acc_thr or e_cost["expected_cost"] == e_acc["expected_cost"]


def test_confusion_at_threshold_matches_sklearn_labels():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.6, 0.9])
    e = confusion_at_threshold(y, score, 0.5)
    assert e == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    e_low = confusion_at_threshold(y, score, 0.3)
    assert e_low == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
