"""
Run the walk-forward backtest on AAPL and print a summary.
Run: python run_backtest.py
"""

from src.data.historical_loader import load_ticker
from src.preprocessing.cleaning import clean_series
from src.preprocessing.returns import log_returns
from src.evaluation.backtest import backtest_price, backtest_volatility, summarize_backtest

TICKER = "AAPL"
START, END = "2022-01-01", "2024-01-01"

# Walk-forward settings:
#   initial_train_size=300 -> first window trains on ~14 months of data
#   horizon=5              -> forecast 5 trading days ahead each window
#   step=20                -> slide forward ~1 month between windows
INITIAL_TRAIN_SIZE = 300
HORIZON = 5
STEP = 20

print(f"Loading {TICKER}...")
df = load_ticker(TICKER, START, END, data_dir="data")
cleaned_df, _ = clean_series(df, column="Close")
price_series = cleaned_df["Close"]
returns_series = log_returns(price_series)

print(f"\nRunning price backtest ({len(price_series)} rows, "
      f"initial_train={INITIAL_TRAIN_SIZE}, horizon={HORIZON}, step={STEP})...")
price_results = backtest_price(
    price_series,
    initial_train_size=INITIAL_TRAIN_SIZE,
    horizon=HORIZON,
    step=STEP,
    arima_order=(1, 1, 1),
)
print(f"-> {len(price_results)} windows completed")
print(price_results)

print("\nRunning volatility backtest...")
vol_results = backtest_volatility(
    returns_series,
    initial_train_size=INITIAL_TRAIN_SIZE,
    horizon=HORIZON,
    step=STEP,
    include_asymmetric=True,
)
print(f"-> {len(vol_results)} windows completed")
print(vol_results)

print("\n" + "=" * 60)
print("SUMMARY: Price track (ARIMA vs. naive)")
print("=" * 60)
summary = summarize_backtest(price_results, "arima_rmse", "naive_rmse")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("SUMMARY: Price track (ARIMA vs. rolling mean)")
print("=" * 60)
summary = summarize_backtest(price_results, "arima_rmse", "rolling_mean_rmse")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("SUMMARY: Volatility track (GARCH vs. historical vol) -- QLIKE")
print("=" * 60)
# QLIKE: lower (more negative) is better, same convention as RMSE/MAE here
summary = summarize_backtest(vol_results, "garch_qlike", "hist_vol_qlike")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\nDone. Save price_results / vol_results to CSV if you want to plot"
      " the per-window comparison for the README.")

print("\n" + "=" * 60)
print("SUMMARY: Volatility track (GJR-GARCH vs. historical vol) -- QLIKE")
print("=" * 60)
summary = summarize_backtest(vol_results, "gjr_qlike", "hist_vol_qlike")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("SUMMARY: Volatility track (EGARCH vs. historical vol) -- QLIKE")
print("=" * 60)
summary = summarize_backtest(vol_results, "egarch_qlike", "hist_vol_qlike")
for k, v in summary.items():
    print(f"  {k}: {v}")