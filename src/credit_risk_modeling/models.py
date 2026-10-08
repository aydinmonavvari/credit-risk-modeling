"""Models, calibration and fairness analysis for the credit-risk study."""

from __future__ import annotations

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS

MODEL_ORDER = ("logistic_regression", "random_forest", "gradient_boosting")


def make_preprocessor() -> ColumnTransformer:
    """Scale numeric features; one-hot the small categorical sets.

    Scalers and encoders are fit on training data only (inside the pipeline),
    so no statistic can leak from validation/test rows.
    """
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )


def make_estimators(seed: int) -> dict[str, object]:
    """The three model families, all with seeds pinned."""
    return {
        "logistic_regression": Pipeline(
            [
                ("pre", make_preprocessor()),
                ("clf", LogisticRegression(max_iter=3000, random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("pre", make_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=20,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("pre", make_preprocessor()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        learning_rate=0.06,
                        max_depth=3,
                        n_estimators=250,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def calibrate(model, method: str, X_cal, y_cal):
    """Wrap a fitted model in Platt (sigmoid) or isotonic calibration.

    Calibration uses a dedicated split that never overlaps the test set, so
    the test set remains a fully untouched estimate of calibrated behavior.
    """
    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def permutation_importance_table(
    model, X, y, n_repeats: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Permutation importance on the (untouched) test set."""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1
    )
    return (
        pd.DataFrame({"feature": X.columns, "importance_mean": result.importances_mean,
                      "importance_std": result.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def fairness_by_sex(
    y_true: pd.Series,
    y_pred: pd.Series,
    sex: pd.Series,
) -> dict[str, float]:
    """Group fairness summary by sex (analysis only; SEX is not a feature).

    Reports selection rates and the difference in true-positive rates
    ("equal opportunity" difference). These are descriptive diagnostics to
    *inform the discussion*, not a certification of fairness.
    """
    rows = {}
    for group in sex.unique():
        mask = sex == group
        yt, yp = y_true[mask], y_pred[mask]
        rows[str(group)] = {
            "share": float(mask.mean()),
            "selection_rate": float(yp.mean()),
            "default_rate": float(yt.mean()),
            "tpr": float(yp[yt == 1].mean()) if (yt == 1).any() else float("nan"),
        }
    groups = sorted(rows)
    if len(groups) == 2:
        g0, g1 = groups
        rows["equal_opportunity_diff"] = abs(rows[g0]["tpr"] - rows[g1]["tpr"])
        rows["selection_rate_diff"] = abs(
            rows[g0]["selection_rate"] - rows[g1]["selection_rate"]
        )
    return rows
