# Research Report — Credit Risk Modeling

**Probability-of-default modeling with calibration, cost-based threshold selection and fairness diagnostics**

*Author: Aydin Monavvari — educational research project (not a production credit system).*
*Run date: 2026-10-08. All numbers are actual outputs of the committed pipeline.*

---

## Abstract

This study builds and evaluates probability-of-default (PD) models on the UCI *Default of Credit Card Clients* dataset (30,000 Taiwanese credit accounts, 22.12% default rate; Yeh & Lien 2009), with an evaluation stack designed around the realities of credit risk: class imbalance, asymmetric error costs, and the need for calibrated probabilities. Three model families — logistic regression, random forest and gradient boosting — are trained with all preprocessing inside sklearn pipelines, calibrated (Platt/isotonic) on a dedicated split, and evaluated exactly once on an untouched test set (n = 7,500). Gradient boosting achieves the best discrimination (ROC-AUC 0.778, PR-AUC 0.558, KS 0.416), consistent with published benchmarks for this dataset. Decision thresholds selected on the calibration split under an expected-cost model (missed default = 5× wrongly declined client) land at 0.16–0.20, far below the naive 0.5, demonstrating that the operating point is a business decision. Calibration barely changes Brier scores — an instructive null result showing that well-trained tree models on this sample are nearly calibrated already, and that monotone recalibration cannot alter ranking metrics. Sex is excluded from features on ethical grounds while group diagnostics are still reported (equal-opportunity difference 0.009; selection-rate difference 0.046), illustrating why attribute removal alone does not guarantee fairness.

## Introduction

Credit scoring is the domain where machine-learning evaluation is most consequential and most frequently oversimplified. Accuracy is meaningless under imbalance; ROC-AUC ranks but does not price; thresholds embed economics that models do not know; and regulatory and ethical constraints shape what may be used as input. This project implements the complete evaluation stack on a public, documented dataset, with the explicit goal of demonstrating each layer — discrimination, calibration, decision economics, interpretability, fairness — as a separate, testable concern.

## Research Question

> How should a probability-of-default model be evaluated and deployed when the two error types carry different business costs — and what do discrimination, calibration and group-level diagnostics each reveal that the others miss?

## Related Work

- **Yeh, I.-C., & Lien, C.-h. (2009).** "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients." *Expert Systems with Applications*, 36(2), 2473–2480. Source dataset and baseline study.
- **Hand, D. J., & Henley, W. E. (1997).** "Statistical classification methods in consumer credit scoring: a review." *Journal of the Royal Statistical Society: Series A*, 160(3), 523–541. Foundations of the scoring discipline.
- **Platt, J. (1999).** "Probabilistic outputs for support vector machines…" in *Advances in Large Margin Classifiers*; **Zadrozny, B., & Elkan, C. (2002).** "Transforming classifier scores into accurate multiclass probability estimates." *KDD '02*. Calibration methods (Platt; isotonic).
- **Saito, T., & Rehmsmeier, M. (2015).** "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets." *PLOS ONE*, 10(3). Basis for treating PR-AUC as primary under imbalance.
- **Hardt, M., Price, E., & Srebro, N. (2016).** "Equality of Opportunity in Supervised Learning." *NeurIPS 29*. The equal-opportunity framing used in the fairness diagnostics.

## Data

| Property | Value |
| --- | --- |
| Source | UCI ML Repository id 350, fetched via the official `ucimlrepo` package |
| Sample | 30,000 accounts (16,875 train / 5,625 calibration / 7,500 test, stratified) |
| Features | credit limit, age, 6 months of repayment status / bill amounts / payment amounts |
| Target | default next month: 22.12% positive |
| Cleaning | education {0,5,6}→other; marriage {0}→other (per codebook); validated |
| Engineered | utilization (bill/limit), payment ratio, months delayed (6m), log limit |

The dataset is cached under `data/raw/` (git-ignored) and never redistributed by this repository.

## Methodology

**Pipelines.** All preprocessing (scaling, one-hot encoding of the small categorical sets) lives inside sklearn `ColumnTransformer` pipelines fitted on training data only.

**Calibration.** Platt scaling for logistic regression; isotonic regression for tree ensembles; both fitted on the dedicated calibration split (5,625 rows) via `CalibratedClassifierCV(cv="prefit")`.

**Threshold economics.** Expected cost `C = 5·FN + 1·FP` is minimized over an 81-point threshold grid **on the calibration split**; the chosen threshold is applied unchanged to the test set. The sweep, cost curve and confusion matrices are reported.

**Fairness.** Sex is excluded from the feature matrix (documented design decision). Group-level selection rates and true-positive rates are computed on test decisions as descriptive diagnostics.

## Experimental Design

The test set is evaluated exactly once, after both calibration and threshold selection are frozen on the calibration split — a three-way split that mirrors how a scoring system would be rolled out (fit → calibrate → set cutoff → launch). Unit tests cover category mapping, feature correctness (utilization math, zero-bill convention, delay counting), KS-statistic properties (perfect and null separation), cost asymmetry, and the assertion that cost-optimal thresholds differ from accuracy-optimal ones.

## Results

Actual outputs (`reports/credit_results.json`).

| Model | ROC-AUC | PR-AUC | KS | Brier | Brier (calibrated) |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.746 | 0.513 | 0.399 | 0.1417 | 0.1417 |
| Random forest | 0.775 | 0.554 | 0.421 | 0.1357 | 0.1362 |
| Gradient boosting | **0.778** | **0.558** | 0.416 | **0.1351** | 0.1357 |

Cost-optimal thresholds (from calibration split): 0.16 (LR), 0.20 (RF), 0.19 (GBM) under FN:FP = 5:1. Fairness diagnostics (LR at its threshold): TPR 71.2% vs 70.3% (difference 0.009); selection rate 46.9% vs 42.3% (difference 0.046) for male vs female groups.

## Discussion

1. **Ranking is nearly saturated; differences are small.** GBM leads LR by 0.032 ROC-AUC — within the range where sample noise matters, and consistent with the dataset's published literature. PR-AUC shows the same ordering with more practical relevance under imbalance.
2. **Calibration is a separate axis from ranking.** Monotone recalibration leaves ROC-AUC, PR-AUC and KS untouched by construction; only Brier and reliability change — and here they change very little, because the tree models' probability outputs are close to calibrated on this sample. The lesson generalizes: models that "need calibration" are not necessarily bad rankers, and good rankers are not automatically calibrated.
3. **The threshold is where economics enters.** At FN:FP = 5:1 the optimum sits at ≈0.19, not 0.5 — accepting many more false positives to catch more defaulters. The threshold–cost curve quantifies the steepness of the penalty for deviating from the optimum and is the artifact a business committee should argue about.
4. **Interpretability agrees with domain priors.** Latest-month repayment status, the count of delayed months and the credit limit dominate permutation importance; engineered utilization ranks above raw bill amounts — a small but genuine feature-engineering win.
5. **Fairness: removal ≠ neutrality.** With sex excluded, a 4.6 pp selection-rate gap persists through correlated features (limits, utilization). The honest response is measurement and discussion, not a claim of fairness — and the equal-opportunity difference (0.009) shows that different metrics can tell different stories about the same decisions.

## Limitations

- **Single static snapshot (2005).** No through-time validation, population-stability monitoring or vintage analysis is possible with this dataset.
- **Illustrative cost ratio.** The 5:1 FN:FP ratio is an assumption; real portfolios estimate it from LGD/EAD and approval economics.
- **No regulatory layer.** Adverse-action explanations and legal fairness tests are out of scope.
- **Single split and seed.** Bootstrap confidence intervals (DeLong) are future work; small AUC differences should not be over-interpreted.
- **Descriptive fairness only.** No debiasing interventions are attempted or claimed.

## Conclusion

The study delivers a complete, leakage-audited PD evaluation stack on a public dataset: gradient boosting ranks best (ROC-AUC 0.778, KS 0.416), calibration adds little because the models are already nearly calibrated, cost-based thresholds move the operating point far below naive defaults, and group diagnostics quantify what feature removal does and does not change. Every layer is separated, testable and documented — the structure a production credit model would inherit.

## Future Research

- Through-time validation with population-stability indices on dated portfolios.
- Full expected-loss stack (PD × LGD × EAD) with stress scenarios.
- DeLong/bootstrap intervals and cost-sensitive model comparison.
- Counterfactual explanations and adverse-action reason codes.
- Fairness-aware training with explicit legal framing.

## References

- Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: a review. *Journal of the Royal Statistical Society: Series A*, 160(3), 523–541.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems*, 29.
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers* (pp. 61–74). MIT Press.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.
- Yeh, I.-C., & Lien, C.-h. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480.
- Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *Proceedings of the 8th ACM SIGKDD*, 694–699.
