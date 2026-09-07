"""
Quick sanity check that the ported modules still work end to end.
Run: python smoke_test.py
"""

from src.data.historical_loader import load_ticker
from src.preprocessing.cleaning import clean_series
from src.preprocessing.returns import log_returns, train_test_split_series
from src.preprocessing.stationarity import adf_test, difference_until_stationary
from src.models.price.arima_sarima import grid_search_arima, forecast
from src.models.volatility.garch import fit_garch, forecast_volatility

TICKER = "AAPL"
START, END = "2022-01-01", "2024-01-01"

print(f"[1/5] Loading {TICKER}...")
df = load_ticker(TICKER, START, END, data_dir="data")
print(f"      -> {len(df)} rows")

print("[2/5] Cleaning outliers...")
cleaned_df, report = clean_series(df, column="Close")
print(f"      -> {report['n_outliers']} outliers flagged ({report['pct_outliers']}%)")

print("[3/5] Computing log returns + stationarity check...")
returns = log_returns(cleaned_df["Close"])
adf = adf_test(returns)
print(f"      -> ADF p-value={adf['p_value']:.4f}, stationary={adf['is_stationary']}")

train, test = train_test_split_series(cleaned_df["Close"], test_size=0.15)

print("[4/5] Fitting ARIMA (small grid, keep it fast for the smoke test)...")
order, model = grid_search_arima(train, p_range=[0, 1], d_range=[1], q_range=[0, 1])
mean_fc, conf_int = forecast(model, steps=5)
# Note: forecast index is integer step (0,1,2,...), not a date -- see the
# _drop_index_freq() comment in arima_sarima.py for why.
print(f"      -> best order={order}, 5-day forecast head:\n{mean_fc.head()}")

print("[5/5] Fitting GARCH(1,1) on returns...")
train_returns, test_returns = train_test_split_series(returns, test_size=0.15)
garch_fit = fit_garch(train_returns)
vol_fc = forecast_volatility(garch_fit, horizon=5)
print(f"      -> 5-day volatility forecast:\n{vol_fc}")

print("\n[bonus] Comparing against baselines...")
from src.models.baselines import compare_price_baselines, historical_volatility_forecast
from src.evaluation.metrics import rmse, mae, volatility_metrics

# --- Price: ARIMA vs naive vs rolling mean, evaluated on the first 5 held-out days
actual_price = test.iloc[:5].reset_index(drop=True)
price_baselines = compare_price_baselines(train, steps=5)

print("\nPrice forecast comparison (RMSE / MAE vs actual next 5 days):")
print(f"  ARIMA{order}:    RMSE={rmse(actual_price, mean_fc.values):.4f}  MAE={mae(actual_price, mean_fc.values):.4f}")
for name, fc in price_baselines.items():
    print(f"  {name:12s}: RMSE={rmse(actual_price, fc.values):.4f}  MAE={mae(actual_price, fc.values):.4f}")

# --- Volatility: GARCH vs rolling historical vol, evaluated with QLIKE
actual_returns = test_returns.iloc[:5].reset_index(drop=True)
hist_vol_fc = historical_volatility_forecast(train_returns, steps=5)

garch_metrics = volatility_metrics(actual_returns, vol_fc.values)
hist_metrics = volatility_metrics(actual_returns, hist_vol_fc.values)

print("\nVolatility forecast comparison (vs realized vol from next 5 days' returns):")
print(f"  GARCH(1,1):       RMSE={garch_metrics['rmse_vol']:.6f}  MAE={garch_metrics['mae_vol']:.6f}  QLIKE={garch_metrics['qlike']:.4f}")
print(f"  historical_vol:   RMSE={hist_metrics['rmse_vol']:.6f}  MAE={hist_metrics['mae_vol']:.6f}  QLIKE={hist_metrics['qlike']:.4f}")

print("\nAll modules ran without errors.")