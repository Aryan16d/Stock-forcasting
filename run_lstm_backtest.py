"""
Walk-forward backtest for LSTM volatility, kept separate from
run_backtest.py since this is much slower (retrains an LSTM at every
window). Prints progress per window so you can see it's working, not stuck.

Run: python run_lstm_backtest.py
"""

from src.data.historical_loader import load_ticker
from src.preprocessing.cleaning import clean_series
from src.preprocessing.returns import log_returns
from src.evaluation.backtest import backtest_lstm_volatility, summarize_backtest

TICKER = "AAPL"
START, END = "2022-01-01", "2024-01-01"

INITIAL_TRAIN_SIZE = 300
HORIZON = 5
STEP = 20

print(f"Loading {TICKER}...")
df = load_ticker(TICKER, START, END, data_dir="data")
cleaned_df, _ = clean_series(df, column="Close")
returns_series = log_returns(cleaned_df["Close"])

print(f"\nRunning LSTM volatility backtest (this will take several minutes -- "
      f"training a fresh LSTM at each window)...\n")
lstm_results = backtest_lstm_volatility(
    returns_series,
    initial_train_size=INITIAL_TRAIN_SIZE,
    horizon=HORIZON,
    step=STEP,
)

print(f"\n-> {len(lstm_results)} windows completed")
print(lstm_results)

print("\n" + "=" * 60)
print("SUMMARY: Volatility track (LSTM vs. historical vol) -- QLIKE")
print("=" * 60)
summary = summarize_backtest(lstm_results, "lstm_qlike", "hist_vol_qlike")
for k, v in summary.items():
    print(f"  {k}: {v}")

lstm_results.to_csv("data/lstm_backtest_results.csv", index=False)
print("\nSaved results to data/lstm_backtest_results.csv")