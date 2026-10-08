# credit-risk-modeling

![CI](https://github.com/aydinmonavvari/credit-risk-modeling/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 1 · Short description

Probability-of-default modeling on the UCI *Default of Credit Card Clients* dataset
(Yeh & Lien 2009; 30,000 accounts) with probability calibration, threshold selection
under asymmetric business costs, ROC/PR/KS evaluation, permutation importance, and
explicit fairness diagnostics — engineered as a research study, not a product.

*Educational research project. Not a production credit system and not credit advice.*

## 2 · Research question

> How should a probability-of-default model be evaluated and deployed when the two
> error types have different business costs — and what do accuracy, calibration,
> and group-level diagnostics each reveal that the others miss?

## 3 · Motivation

Credit scoring is the canonical "accuracy is the wrong metric" problem: defaults are a
minority class, the cost of missing one (a charge-off) differs wildly from the cost of
declining a good client (lost revenue), and the decision threshold is a *business*
choice that belongs outside the model. Probability quality matters as much as ranking:
a model whose 0.8 is really a 0.6 cannot support pricing or provisioning. This project
builds the full evaluation stack — discrimination, calibration, cost-based thresholds,
interpretability and fairness — on a public dataset with documented provenance.

## 4 · Why this matters

- **For finance:** expected-loss estimation (PD × LGD × EAD) is the foundation of
  credit risk management and Basel-style regulation; PD calibration is what makes the
  "PD" an actual probability.
- **For machine learning:** class imbalance, monotone metrics, calibration and
  threshold economics generalize directly to fraud, churn, and medical triage.
- **For responsible ML:** the dataset includes a demographic attribute (sex), which
  this project deliberately excludes from features while still measuring group
  outcomes — the discussable, honest way to handle it.

## 5 · Methodology

1. **Data** — UCI dataset (id 350) fetched through the official `ucimlrepo` channel,
   cached locally (never committed), X1–X23 renamed to semantic names.
2. **Preparation** — unlabeled categories mapped per the codebook (`education` 0/5/6 →
   "other"; `marriage` 0 → "other"); engineered features: latest-month utilization,
   payment ratio, count of delayed months in 6 months, log credit limit; type and
   missing-value validation.
3. **Splits** — stratified: 56.25% train / 18.75% calibration / 25% test. The test set
   is touched exactly once, after threshold selection.
4. **Models** — logistic regression (scaled, one-hot pipeline), random forest
   (300 trees), gradient boosting; all seeds pinned; all preprocessing inside sklearn
   pipelines (no statistic leakage).
5. **Calibration** — Platt scaling (LR) and isotonic regression (tree models) fitted
   on the calibration split; reliability diagrams and Brier scores on test.
6. **Threshold selection** — the decision threshold minimizing *expected cost*
   (FN = 5 × FP in the base case) is chosen **on the calibration split** and applied to
   test — thresholds are not tuned on test.
7. **Interpretability** — permutation importance on test (mean across models).
8. **Fairness** — SEX excluded from features; group selection rates, true-positive
   rates and their differences reported as diagnostics.

## 6 · Dataset

| Property | Value |
| --- | --- |
| Source | UCI ML Repository — *Default of Credit Card Clients* (Yeh & Lien 2009) |
| Units | 30,000 credit-card accounts, Taiwan, April–September 2005 |
| Features | 23: credit limit, demographics, 6 months of repayment status, bills and payments |
| Target | `default` next month (1 = default) — **22.12% positive rate** |
| Splits (actual) | 16,875 train / 5,625 calibration / 7,500 test (stratified) |

## 7 · Data sources

UCI Machine Learning Repository (official `ucimlrepo` package). The dataset is
publicly available for research; the paper to cite is Yeh & Lien (2009). Raw CSV
cached under `data/raw/` (git-ignored) — the repository never redistributes the data.

## 8 · Architecture

```
UCI (ucimlrepo) ──► cache ──► prepare (codebook mapping, validation)
                                   │
                     feature build (utilization, payment ratio, delays)
                     SEX excluded from features (fairness module only)
                                   │
             stratified split: train / calibration / test (untouched)
                                   │
        logistic regression · random forest · gradient boosting
                        │                     │
        threshold from expected cost      Platt / isotonic
              (calibration split)        (calibration split)
                        └─────────┬─────────┘
                       single evaluation on test
                                   │
        ROC/PR/KS · Brier · reliability · cost curve · importance · fairness
```

## 9 · Experimental design

**Why accuracy is the wrong metric here** (demonstrated, not asserted): with a 22%
default rate, "never default" scores 78% accuracy while having zero business value.
The study's decision layer minimizes expected cost `5·FN + 1·FP` instead, and the
tests assert that the cost-optimal threshold differs from the accuracy-optimal one on
synthetic data.

**Leakage controls:** scaling/encoding inside pipelines; calibration and threshold
selection on the calibration split only; test set evaluated once.

**Class imbalance handling:** the imbalance is moderate (1:3.5). The study deliberately
does *not* resample; class weights/resampling are discussed as alternatives, with the
known pitfall that resampling before splitting distorts calibration.

## 10 · Models

| Model | Configuration |
| --- | --- |
| Logistic regression | `StandardScaler` + one-hot, `max_iter=3000` |
| Random forest | 300 trees, `min_samples_leaf=20`, `n_jobs=-1` |
| Gradient boosting | `learning_rate=0.06`, depth 3, 250 trees |
| Calibration | Platt (LR), isotonic (RF/GBM), `cv="prefit"` on the calibration split |

All `random_state = 42`.

## 11 · Evaluation metrics

- **Discrimination:** ROC-AUC, PR-AUC (primary under imbalance), KS statistic
  (credit-scoring standard).
- **Calibration:** Brier score, reliability diagrams.
- **Decision layer:** confusion matrix and expected cost at the cost-optimal
  threshold; threshold–cost curve.
- **Interpretability:** permutation importance (test set).
- **Fairness:** selection-rate and equal-opportunity differences by sex.

## 12 · Results

Actual outputs of the committed run (`reports/credit_results.json`).

**Discrimination and calibration (test set, n = 7,500):**

| Model | ROC-AUC | PR-AUC | KS | Brier | Brier (calibrated) |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.746 | 0.513 | 0.399 | 0.1417 | 0.1417 |
| Random forest | 0.775 | 0.554 | 0.421 | 0.1357 | 0.1362 |
| Gradient boosting | **0.778** | **0.558** | 0.416 | **0.1351** | 0.1357 |

**Decision layer (thresholds selected on the calibration split, applied to test):**
cost-optimal thresholds 0.16 (LR), 0.20 (RF), 0.19 (GBM) under FN:FP = 5:1 — far below
the naive 0.5, because missing a defaulter is 5× worse than declining a good client.

**Fairness diagnostics (LR decisions at the cost-optimal threshold):** true-positive
rate 71.2% (male) vs 70.3% (female) — equal-opportunity difference 0.009; selection
rate difference 0.046. Sex was not a feature; the remaining differences come through
correlated features (limit, utilization), illustrating why "we removed the attribute"
is not a fairness guarantee.

Figures: [`roc_calibration.png`](figures/roc_calibration.png) (ROC + reliability
diagrams), [`threshold_cost.png`](figures/threshold_cost.png) (expected cost vs
threshold with the selected point),
[`feature_importance.png`](figures/feature_importance.png) (top features across
models — dominated by `pay_sep` (latest repayment status), `months_delayed_6m` and
`limit_bal`).

## 13 · Interpretation

1. **Gradient boosting wins on ranking, narrowly.** ROC-AUC 0.778 vs 0.746 for LR —
   consistent with the published range for this dataset (≈0.72–0.78). The gain
   concentrates where it matters operationally: higher PR-AUC at the same operating
   region.
2. **Calibration moved Brier only slightly — and that is informative.** Isotonic
   recalibration of a well-trained GBM on 5,625 rows yields a modest Brier change
   (0.1351 → 0.1357; RF similar). ROC-AUC/PR-AUC/KS are unchanged *by construction*
   (monotone transforms preserve ranking). The lesson: calibration is about absolute
   probability quality, not ranking, and it needs data — a model already close to
   well-calibrated has little left to gain.
3. **The threshold is a business decision with model-shaped consequences.** Moving the
   threshold from 0.5 to ≈0.19 trades a flood of false positives for far fewer missed
   defaults — the expected-cost curve (`figures/threshold_cost.png`) makes the optimum
   visible and shows how steeply cost deteriorates away from it.
4. **Feature importance matches credit intuition.** Latest repayment status
   (`pay_sep`), the number of delayed months and the credit limit dominate; engineered
   utilization features rank above raw bill amounts.
5. **Fairness is not achieved by deletion.** Excluding sex leaves a 4.6 pp selection-
   rate gap driven by correlated inputs. The honest statement is measurement +
   discussion, not a claim of fairness certification.

## 14 · Limitations

- **Static 2005 Taiwanese sample.** No time-based split is possible (no scoring-date
  column); real credit portfolios need through-time validation (population stability,
  vintage curves).
- **No LGD/EAD.** The FN cost (5×FP) is an illustrative constant; real expected-loss
  models need loss-given-default and exposure at default.
- **Regulation is out of scope.** Adverse-action explanations, ECOA/GDPR-style
  requirements and adverse-effect thresholds require legal context this project does
  not model.
- **Fairness metrics are descriptive.** No debiasing is attempted; demographic
  parity/equal opportunity are reported, not optimized.
- **One split, one seed.** Confidence intervals on AUC (DeLong) are future work;
  differences of 0.01 AUC between models should not be over-read.

## 15 · Reproducibility

```bash
# 1) environment (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# 2) data (network, then cached; dataset NOT committed to the repo)
python scripts/download_data.py

# 3) full study (~2 min; deterministic, seed = 42)
python scripts/run_study.py

# 4) verify: lint + 7 offline unit tests
ruff check .
pytest -q
```

## 16 · Installation

```bash
git clone https://github.com/aydinmonavvari/credit-risk-modeling.git
cd credit-risk-modeling
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

## 17 · Usage

```bash
python scripts/download_data.py   # fetch + cache UCI dataset (id=350)
python scripts/run_study.py       # train, calibrate, threshold, evaluate
pytest -q                         # offline test suite
```

As a library:

```python
from credit_risk_modeling.config import DEFAULT_CONFIG
from credit_risk_modeling.data import load_raw, prepare
from credit_risk_modeling.features import build_features
from credit_risk_modeling.evaluation import discrimination_metrics

frame = prepare(load_raw(DEFAULT_CONFIG))
feats = build_features(frame)
print(feats["default"].mean())  # 0.2212 - the default rate
```

## 18 · Example

Actual model behavior at the selected operating point (gradient boosting,
threshold 0.19 chosen on the calibration split, applied to the untouched test set):

- the model flags ~26% of test accounts as high-risk;
- at that point it captures the majority of true defaulters while declining a
  controlled share of good clients;
- under the FN:FP = 5:1 cost model this operating point beats every
  accuracy-optimal threshold — the concrete demonstration that **threshold choice is
  where business value enters the pipeline**.

## 19 · Project structure

```
credit-risk-modeling/
├── README.md
├── LICENSE · CITATION.cff · pyproject.toml · requirements.txt
├── .python-version · .gitignore
├── src/credit_risk_modeling/
│   ├── config.py        # splits, costs, seed, paths
│   ├── data.py          # ucimlrepo fetch, cache, codebook mapping, validation
│   ├── features.py      # feature matrix + engineered features (SEX excluded)
│   ├── models.py        # pipelines, calibration, importance, fairness
│   ├── evaluation.py    # ROC/PR/KS/Brier, confusion, expected cost, sweep
│   └── pipeline.py      # orchestration + reports + figures
├── tests/               # 7 offline tests (preprocessing, KS math, cost asymmetry)
├── notebooks/           # EDA of the real dataset
├── scripts/             # download_data.py, run_study.py
├── data/raw · data/processed   # git-ignored caches (.gitkeep tracked)
├── reports/             # JSON/CSV results (generated, committed)
├── figures/             # 3 generated figures (committed)
├── docs/research_report.md
└── .github/workflows/ci.yml
```

## 20 · Future work

- Through-time validation and population-stability monitoring.
- LGD/EAD modeling toward a full expected-loss stack.
- DeLong confidence intervals and bootstrap model comparison.
- Counterfactual explanations for adverse-action letters.
- Fairness-aware training comparisons (with legal-context caveats).

## 21 · Citation

If you use this work, please cite (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{monavvari2026creditriskmodeling,
  author  = {Monavvari, Aydin},
  title   = {credit-risk-modeling: default probability modeling with calibration, cost-based thresholds and fairness diagnostics},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/aydinmonavvari/credit-risk-modeling}
}
```

Dataset citation: Yeh, I.-C., & Lien, C.-h. (2009). The comparisons of data mining
techniques for the predictive accuracy of probability of default of credit card
clients. *Expert Systems with Applications*, 36(2), 2473–2480.

## 22 · License

MIT — see [`LICENSE`](LICENSE).

## 23 · Acknowledgments

Dataset from the UCI Machine Learning Repository. Built with pandas, scikit-learn and
matplotlib.
