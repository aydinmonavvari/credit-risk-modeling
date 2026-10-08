"""CLI: fetch and cache the UCI credit-card default dataset (id=350)."""

from credit_risk_modeling.config import DEFAULT_CONFIG
from credit_risk_modeling.data import load_raw

if __name__ == "__main__":
    frame = load_raw(DEFAULT_CONFIG)
    print(f"cached {len(frame)} rows")
