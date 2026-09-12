"""
Check whether the leverage/asymmetry parameter in GJR-GARCH and EGARCH is
statistically significant for AAPL -- this directly tests the explanation
for why the asymmetric models didn't beat plain GARCH or historical vol in
the walk-forward backtest.

Run: python check_leverage.py
"""

from src.data.historical_loader import load_ticker
from src.preprocessing.cleaning import clean_series
from src.preprocessing.returns import log_returns, train_test_split_series
from src.models.volatility.egarch_gjr import (
    fit_gjr_garch,
    fit_egarch,
    leverage_effect_significant,
)

TICKER = "AAPL"
START, END = "2022-01-01", "2024-01-01"

print(f"Loading {TICKER}...")
df = load_ticker(TICKER, START, END, data_dir="data")
cleaned_df, _ = clean_series(df, column="Close")
returns = log_returns(cleaned_df["Close"])
train, _ = train_test_split_series(returns, test_size=0.15)

print(f"Fitting on {len(train)} observations (full training set, not a single walk-forward window)...\n")

print("=" * 60)
print("GJR-GARCH")
print("=" * 60)
gjr_fit = fit_gjr_garch(train)
print(gjr_fit.summary())
print("\nLeverage/asymmetry parameter significance:")
print(leverage_effect_significant(gjr_fit))

print("\n" + "=" * 60)
print("EGARCH")
print("=" * 60)
egarch_fit = fit_egarch(train)
print(egarch_fit.summary())
print("\nLeverage/asymmetry parameter significance:")
print(leverage_effect_significant(egarch_fit))