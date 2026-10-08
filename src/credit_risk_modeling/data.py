"""Data acquisition and preparation for the UCI credit-card default dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Semantic renaming of the UCI X1..X23 coding (Yeh & Lien 2009 documentation).
RENAME = {
    "X1": "limit_bal",
    "X2": "sex",
    "X3": "education",
    "X4": "marriage",
    "X5": "age",
    "X6": "pay_sep",
    "X7": "pay_aug",
    "X8": "pay_jul",
    "X9": "pay_jun",
    "X10": "pay_may",
    "X11": "pay_apr",
    "X12": "bill_amt_sep",
    "X13": "bill_amt_aug",
    "X14": "bill_amt_jul",
    "X15": "bill_amt_jun",
    "X16": "bill_amt_may",
    "X17": "bill_amt_apr",
    "X18": "pay_amt_sep",
    "X19": "pay_amt_aug",
    "X20": "pay_amt_jul",
    "X21": "pay_amt_jun",
    "X22": "pay_amt_may",
    "X23": "pay_amt_apr",
    "Y": "default",
}
CATEGORICAL = ("education", "marriage")
SEX_VALUES = {1: "male", 2: "female"}
EDUCATION_KNOWN = {1: "graduate_school", 2: "university", 3: "high_school", 4: "other"}
MARRIAGE_KNOWN = {1: "married", 2: "single", 3: "other"}


def load_raw(config) -> pd.DataFrame:
    """Fetch the UCI dataset (id=350) with a local CSV cache.

    Priority: local cache -> ucimlrepo fetch (the official UCI distribution
    channel). The cache lives in ``data/raw/`` which is git-ignored; no
    dataset is committed to the repository.
    """
    cache = Path(config.raw_dir) / "uci_credit_default.csv"
    if cache.exists():
        frame = pd.read_csv(cache)
        logger.info("loaded %d cached rows from %s", len(frame), cache)
        return frame

    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=350)
    frame = pd.concat([ds.data.features, ds.data.targets], axis=1)
    frame = frame.rename(columns=RENAME)
    frame.to_csv(cache, index=False)
    logger.info("fetched %d rows from UCI and cached to %s", len(frame), cache)
    return frame


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean categories, coerce types and validate the prepared frame.

    - ``education``: values {0, 5, 6} are unlabeled in the source; map to 4
      ("other") exactly as the codebook instructs.
    - ``marriage``: value 0 is unlabeled; map to 3 ("other").
    - ``default``: integer 0/1 target.
    - ``sex``: kept in the frame ONLY for the fairness analysis; the feature
      builder must not consume it (see features.py).
    """
    out = frame.copy()
    out["education"] = out["education"].where(out["education"].isin(EDUCATION_KNOWN), 4)
    out["marriage"] = out["marriage"].where(out["marriage"].isin(MARRIAGE_KNOWN), 3)
    out["education_label"] = out["education"].map(EDUCATION_KNOWN)
    out["marriage_label"] = out["marriage"].map(MARRIAGE_KNOWN)
    out["sex_label"] = out["sex"].map(SEX_VALUES)
    if out["sex_label"].isna().any():
        raise ValueError("unexpected values in sex column")
    out["default"] = out["default"].astype(int)
    if not set(out["default"].unique()) <= {0, 1}:
        raise ValueError("target must be binary 0/1")
    if (out[["limit_bal", "age"]] <= 0).any().any():
        raise ValueError("limit_bal and age must be strictly positive")
    if out.isna().any().any():
        raise ValueError("prepared frame must not contain missing values")
    return out
