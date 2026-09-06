"""
Data loading.

Tries to pull real OHLCV data from Yahoo Finance via `yfinance`. If that
fails (no network, ticker delisted, rate limit, etc.) it falls back to a
synthetic-but-realistic daily price series generated with geometric
Brownian motion, calibrated with drift/volatility parameters roughly in
line with each ticker's historical behavior. The synthetic path exists so
the rest of the pipeline (cleaning, ARIMA/SARIMA, GARCH, LSTM/GRU) is fully
runnable and testable offline -- swap in real data by installing
`yfinance` and having network access.
"""

import os
import numpy as np
import pandas as pd

# Rough daily drift/volatility calibration per ticker, used only for the
# synthetic fallback. These are illustrative, not a market forecast.
_SYNTHETIC_PARAMS = {
    "NVDA": {"start_price": 45.0, "annual_drift": 0.55, "annual_vol": 0.55},
    "AAPL": {"start_price": 65.0, "annual_drift": 0.20, "annual_vol": 0.30},
    "TSLA": {"start_price": 70.0, "annual_drift": 0.35, "annual_vol": 0.60},
}

TRADING_DAYS_PER_YEAR = 252


def _generate_synthetic_ohlcv(ticker, start, end, seed=42):
    """Generate a synthetic daily OHLCV DataFrame via GBM for one ticker."""
    params = _SYNTHETIC_PARAMS.get(
        ticker, {"start_price": 50.0, "annual_drift": 0.25, "annual_vol": 0.40}
    )
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    rng = np.random.default_rng(seed + abs(hash(ticker)) % 1000)
    dt = 1.0 / TRADING_DAYS_PER_YEAR
    mu, sigma = params["annual_drift"], params["annual_vol"]

    # occasional volatility regime shocks so GARCH has something to find
    regime = np.ones(n)
    n_shocks = max(1, n // 250)
    shock_starts = rng.choice(n, size=n_shocks, replace=False)
    for s in shock_starts:
        span = min(n - s, rng.integers(10, 40))
        regime[s : s + span] *= rng.uniform(1.8, 3.0)

    shocks = rng.normal(0, 1, n)
    daily_returns = (mu - 0.5 * sigma**2) * dt + sigma * regime * np.sqrt(dt) * shocks

    log_prices = np.log(params["start_price"]) + np.cumsum(daily_returns)
    close = np.exp(log_prices)

    # Inject a handful of sharp one-day outliers to exercise the z-score
    # cleaning step later in the pipeline.
    n_outliers = max(1, n // 300)
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    close_with_outliers = close.copy()
    close_with_outliers[outlier_idx] *= rng.choice([0.7, 1.35], size=n_outliers)

    daily_range = np.abs(rng.normal(0, sigma * 0.3, n)) * close
    open_ = close_with_outliers * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close_with_outliers) + daily_range * 0.5
    low = np.minimum(open_, close_with_outliers) - daily_range * 0.5
    volume = rng.integers(5_000_000, 60_000_000, n)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close_with_outliers,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def load_ticker(ticker, start, end, data_dir="data", force_synthetic=False):
    """
    Return a DataFrame of daily OHLCV data for `ticker`.

    Order of preference: cached CSV in `data_dir` -> yfinance download ->
    synthetic fallback. The result is cached to `data_dir` so re-runs are
    fast and reproducible.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, f"{ticker}.csv")

    if os.path.exists(cache_path) and not force_synthetic:
        return pd.read_csv(cache_path, index_col="Date", parse_dates=True)

    df = None
    if not force_synthetic:
        try:
            import yfinance as yf

            raw = yf.download(ticker, start=start, end=end, progress=False)
            if raw is not None and len(raw) > 0:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index.name = "Date"
        except Exception as exc:  # noqa: BLE001 -- log and fall back
            print(f"[data_loader] yfinance failed for {ticker} ({exc}); "
                  f"using synthetic data instead.")

    if df is None or df.empty:
        print(f"[data_loader] Using synthetic data for {ticker}.")
        df = _generate_synthetic_ohlcv(ticker, start, end)

    df.to_csv(cache_path)
    return df


def load_all(tickers, start, end, data_dir="data", force_synthetic=False):
    """Return {ticker: DataFrame} for every ticker in `tickers`."""
    return {
        t: load_ticker(t, start, end, data_dir=data_dir, force_synthetic=force_synthetic)
        for t in tickers
    }
