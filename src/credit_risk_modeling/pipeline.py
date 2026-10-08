"""End-to-end orchestration for the credit-risk study."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from .config import DEFAULT_CONFIG, PROJECT_ROOT
from .data import load_raw, prepare
from .evaluation import (
    discrimination_metrics,
    optimal_threshold,
    threshold_sweep,
)
from .features import FEATURE_COLUMNS, TARGET, build_features
from .models import (
    MODEL_ORDER,
    calibrate,
    fairness_by_sex,
    make_estimators,
    permutation_importance_table,
)

logger = logging.getLogger(__name__)


def run_pipeline(config=None, save_outputs: bool = True) -> dict:
    config = config or DEFAULT_CONFIG
    config.ensure_dirs()
    rng = config.random_state

    frame = prepare(load_raw(config))
    feats = build_features(frame)
    X = feats[list(FEATURE_COLUMNS)]
    y = feats[TARGET]
    sex = frame["sex_label"]

    # ---- stratified splits: train -> (train, calibration); test untouched --
    X_trainval, X_test, y_trainval, y_test, sex_trainval, sex_test = train_test_split(
        X, y, sex, test_size=config.test_size, stratify=y, random_state=rng
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval,
        y_trainval,
        test_size=config.calibration_size,
        stratify=y_trainval,
        random_state=rng,
    )
    logger.info(
        "train=%d cal=%d test=%d | default rates %.4f/%.4f/%.4f",
        len(X_train),
        len(X_cal),
        len(X_test),
        y_train.mean(),
        y_cal.mean(),
        y_test.mean(),
    )

    estimators = make_estimators(rng)
    rows, threshold_rows, calib_curves = [], [], []
    best_thresholds: dict[str, float] = {}
    importance_tables: dict[str, pd.DataFrame] = {}
    uncalibrated_scores: dict[str, np.ndarray] = {}
    calibrated_scores: dict[str, np.ndarray] = {}

    for name in MODEL_ORDER:
        model = estimators[name]
        model.fit(X_train, y_train)
        # select threshold + calibration mapping on the CALIBRATION split only
        raw_cal = model.predict_proba(X_cal)[:, 1]
        raw_test = model.predict_proba(X_test)[:, 1]
        uncalibrated_scores[name] = raw_test
        thr_rows = threshold_sweep(
            y_cal, raw_cal, config.fn_cost, config.fp_cost, config.threshold_grid
        )
        thr = optimal_threshold(thr_rows)
        best_thresholds[name] = thr

        method = "sigmoid" if name == "logistic_regression" else "isotonic"
        calib = calibrate(model, method, X_cal, y_cal)
        cal_test = calib.predict_proba(X_test)[:, 1]
        calibrated_scores[name] = cal_test

        disc_raw = discrimination_metrics(y_test, raw_test)
        disc_cal = discrimination_metrics(y_test, cal_test)
        final_rows = threshold_sweep(
            y_test, cal_test, config.fn_cost, config.fp_cost, config.threshold_grid
        )
        chosen = min(
            final_rows,
            key=lambda r: abs(r["threshold"] - thr),
        )
        rows.append({"model": name, **disc_raw, **{f"cal_{k}": v for k, v in disc_cal.items()},
                     "threshold_from_cal": thr})
        threshold_rows.append({"model": name, **chosen})
        frac_pos, mean_pred = calibration_curve(
            y_test, cal_test, n_bins=10, strategy="quantile"
        )
        calib_curves.append(
            {"model": name, "fraction_positive": frac_pos, "mean_predicted": mean_pred}
        )
        importance_tables[name] = permutation_importance_table(
            calib, X_test, y_test, seed=rng
        )

    metrics = pd.DataFrame(rows)
    threshold_tbl = pd.DataFrame(threshold_rows)
    importance = pd.concat(
        [df.assign(model=n) for n, df in importance_tables.items()]
    )
    fairness = fairness_by_sex(
        y_test.reset_index(drop=True),
        pd.Series(
            (
                threshold_tbl.iloc[0]["threshold"]
                <= calibrated_scores["logistic_regression"]
            ).astype(int),
            index=y_test.reset_index(drop=True).index,
        ),
        sex_test.reset_index(drop=True),
    )

    if save_outputs:
        _save(config, metrics, threshold_tbl, importance, fairness, calib_curves,
              uncalibrated_scores, calibrated_scores, y_test, best_thresholds)

    return {
        "metrics": metrics,
        "threshold_tbl": threshold_tbl,
        "importance": importance,
        "fairness": fairness,
    }


def _save(config, metrics, threshold_tbl, importance, fairness, calib_curves,
          raw_scores, cal_scores, y_test, best_thresholds) -> None:
    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "project": "credit-risk-modeling",
        "description": "Credit default probability modeling on the UCI Taiwan credit "
        "dataset with calibration, asymmetric-cost threshold selection and fairness "
        "diagnostics (educational research project; not a production credit system).",
        "config": {
            k: (str(v).replace(str(PROJECT_ROOT) + "/", "") if isinstance(v, Path) else v)
            for k, v in asdict(config).items()
        },
        "metrics": metrics.round(6).to_dict("records"),
        "threshold_analysis": threshold_tbl.round(6).to_dict("records"),
        "fairness": fairness,
        "selected_thresholds": best_thresholds,
    }
    (config.reports_dir / "credit_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    threshold_tbl.round(6).to_csv(config.reports_dir / "threshold_analysis.csv", index=False)
    metrics.round(6).to_csv(config.reports_dir / "model_metrics.csv", index=False)
    importance.round(6).to_csv(config.reports_dir / "permutation_importance.csv", index=False)

    # ---- figures -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in raw_scores:
        fpr, tpr, _ = roc_curve(y_test, cal_scores[name])
        ax[0].plot(fpr, tpr, lw=1.6, label=name)
        disc = metrics.set_index("model").loc[name]
        logger.info("%s ROC-AUC %.4f -> cal %.4f", name, disc["roc_auc"], disc["cal_roc_auc"])
    ax[0].plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
    ax[0].set_title("ROC curves (calibrated scores, test set)")
    ax[0].set_xlabel("false positive rate")
    ax[0].set_ylabel("true positive rate")
    ax[0].legend()
    for curve in calib_curves:
        ax[1].plot(curve["mean_predicted"], curve["fraction_positive"], marker="o", lw=1.4,
                   label=curve["model"])
    ax[1].plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--", label="perfect calibration")
    ax[1].set_title("Reliability diagrams (test set)")
    ax[1].set_xlabel("mean predicted probability")
    ax[1].set_ylabel("observed default rate")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(config.figures_dir / "roc_calibration.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    top = (
        importance.groupby("feature")["importance_mean"].mean().sort_values().tail(12)
    )
    ax.barh(top.index, top.values, color="tab:blue")
    ax.set_title("Permutation importance (mean across models, test set)")
    ax.set_xlabel("mean decrease in score")
    fig.tight_layout()
    fig.savefig(config.figures_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sweep = pd.DataFrame(
        threshold_sweep(y_test, cal_scores["gradient_boosting"], 5.0, 1.0)
    )
    ax.plot(sweep["threshold"], sweep["expected_cost"], lw=2, color="tab:red")
    ax.axvline(best_thresholds["gradient_boosting"], color="grey", ls="--", lw=1,
               label="cost-optimal threshold (from calibration split)")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("expected cost (FN=5, FP=1)")
    ax.set_title("Expected cost vs decision threshold - gradient boosting")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.figures_dir / "threshold_cost.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_pipeline(DEFAULT_CONFIG)
    print("done")
