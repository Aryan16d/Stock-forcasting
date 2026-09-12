"""
Quick single-window test of the LSTM volatility model before wiring it into
the full walk-forward backtest (LSTM training is much slower than GARCH per
window, so worth confirming this works correctly once first).

Run: python test_lstm_volatility.py
"""

from src.data.historical_loader import load_ticker
from src.preprocessing.cleaning import clean_series
from src.preprocessing.returns import log_returns, train_test_split_series
from src.models.volatility.lstm_volatility import fit_lstm_volatility, forecast_lstm_volatility
from src.models.baselines import historical_volatility_forecast
from src.evaluation.metrics import volatility_metrics

TICKER = "AAPL"
START, END = "2022-01-01", "2024-01-01"
HORIZON = 5

print(f"Loading {TICKER}...")
df = load_ticker(TICKER, START, END, data_dir="data")
cleaned_df, _ = clean_series(df, column="Close")
returns = log_returns(cleaned_df["Close"])
train_returns, test_returns = train_test_split_series(returns, test_size=0.15)

print(f"Training LSTM on {len(train_returns)} observations (this will take a bit)...")
model, lookback = fit_lstm_volatility(train_returns, lookback=20, epochs=50, verbose=1)

print(f"\nForecasting {HORIZON} days ahead...")
lstm_vol_fc = forecast_lstm_volatility(model, train_returns, lookback, horizon=HORIZON)
print(f"LSTM volatility forecast:\n{lstm_vol_fc}")

hist_vol_fc = historical_volatility_forecast(train_returns, steps=HORIZON)

actual_returns = test_returns.iloc[:HORIZON].values
lstm_metrics = volatility_metrics(actual_returns, lstm_vol_fc.values)
hist_metrics = volatility_metrics(actual_returns, hist_vol_fc.values)

print("\nComparison (vs realized vol from next 5 days' returns):")
print(f"  LSTM:           RMSE={lstm_metrics['rmse_vol']:.6f}  MAE={lstm_metrics['mae_vol']:.6f}  QLIKE={lstm_metrics['qlike']:.4f}")
print(f"  historical_vol: RMSE={hist_metrics['rmse_vol']:.6f}  MAE={hist_metrics['mae_vol']:.6f}  QLIKE={hist_metrics['qlike']:.4f}")