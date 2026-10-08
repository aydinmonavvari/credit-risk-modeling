"""Configuration for the credit-risk modeling study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CreditConfig:
    """Static configuration for one run of the credit-risk study.

    Attributes
    ----------
    test_size, calibration_size:
        Fractions of the sample held out (stratified by the label) for the
        untouched test set and for the calibration/validation set.
    random_state:
        Global seed; every model pins its own random_state to this value.
    fn_cost, fp_cost:
        Business costs of a false negative (missed defaulter) and a false
        positive (wrongly declined good client), in arbitrary but comparable
        units. Threshold selection minimizes expected cost under this ratio.
    include_sex_feature:
        Always False in the main pipeline: SEX is excluded from the feature
        matrix on ethical/legal grounds and retained only for the fairness
        analysis. This is a design decision, not an oversight.
    """

    test_size: float = 0.25
    calibration_size: float = 0.25  # of the remaining train+calibration part
    random_state: int = 42
    fn_cost: float = 5.0
    fp_cost: float = 1.0
    threshold_grid: int = 81  # thresholds from 0.05 to 0.85
    include_sex_feature: bool = False

    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    figures_dir: Path = PROJECT_ROOT / "figures"
    reports_dir: Path = PROJECT_ROOT / "reports"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.figures_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = CreditConfig()
