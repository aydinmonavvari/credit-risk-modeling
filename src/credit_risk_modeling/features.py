"""Feature engineering for the credit-risk study."""

from __future__ import annotations

import pandas as pd

# SEX is deliberately absent: the fairness module analyses it separately.
CATEGORICAL_FEATURES = ("education_label", "marriage_label")
NUMERIC_FEATURES = (
    "limit_bal",
    "age",
    "pay_sep",
    "pay_aug",
    "pay_jul",
    "pay_jun",
    "pay_may",
    "pay_apr",
    "bill_amt_sep",
    "bill_amt_aug",
    "bill_amt_jul",
    "bill_amt_jun",
    "bill_amt_may",
    "bill_amt_apr",
    "pay_amt_sep",
    "pay_amt_aug",
    "pay_amt_jul",
    "pay_amt_jun",
    "pay_amt_may",
    "pay_amt_apr",
    "utilization_sep",
    "payment_ratio_sep",
    "months_delayed_6m",
    "log_limit_bal",
)

TARGET = "default"


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the model-ready feature matrix (plus the target column).

    Engineered features (documented, deterministic, no leakage — they use
    only the six most recent billing cycles available at scoring time):
    - ``utilization_sep``: most recent bill amount / credit limit (capped at
      5 to bound outliers);
    - ``payment_ratio_sep``: most recent payment / most recent bill (capped
      at 5; undefined when the bill is zero -> set to 1, "fully covered");
    - ``months_delayed_6m``: count of months with a delayed payment in the
      repayment-status history (status >= 1);
    - ``log_limit_bal``: log of the credit limit (heavy right tail).
    """
    out = pd.DataFrame(index=frame.index)
    for col in NUMERIC_FEATURES:
        if col in frame.columns:
            out[col] = frame[col]
    for col in CATEGORICAL_FEATURES:
        out[col] = frame[col].astype("category")

    limit = frame["limit_bal"].clip(lower=1.0)
    out["utilization_sep"] = (frame["bill_amt_sep"] / limit).clip(upper=5.0)
    ratio = frame["pay_amt_sep"] / frame["bill_amt_sep"].replace(0, np_nan())
    out["payment_ratio_sep"] = ratio.fillna(1.0).clip(upper=5.0)
    pay_cols = ["pay_sep", "pay_aug", "pay_jul", "pay_jun", "pay_may", "pay_apr"]
    out["months_delayed_6m"] = (frame[pay_cols] >= 1).sum(axis=1)
    out["log_limit_bal"] = pd.Series(
        np_log(frame["limit_bal"].to_numpy()), index=frame.index
    )
    out[TARGET] = frame[TARGET].astype(int)
    if out.drop(columns=[TARGET]).isna().any().any():
        raise ValueError("feature matrix must not contain missing values")
    return out


FEATURE_COLUMNS = tuple(c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def np_nan() -> float:
    import numpy as np

    return float(np.nan)


def np_log(values):
    import numpy as np

    return np.log(values)
