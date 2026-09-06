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
train_returns, _ = train_test_split_series(returns, test_size=0.15)
garch_fit = fit_garch(train_returns)
vol_fc = forecast_volatility(garch_fit, horizon=5)
print(f"      -> 5-day volatility forecast:\n{vol_fc}")
 
print("\nAll modules ran without errors.")