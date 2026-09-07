
"""
Naive baselines for both forecasting tracks.

These exist so every "real" model (ARIMA, GARCH, LSTM, ...) has an honest,
trivial-to-compute reference point to beat. If a model can't outperform
these, that's a genuine finding worth reporting, not a bug to hide.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Price baselines
# ---------------------------------------------------------------------------

def naive_forecast(train_series, steps):
    """
    Random-walk baseline: tomorrow = today. Repeats the last observed value
    for every step of the forecast horizon.

    This is the standard null hypothesis for price forecasting -- financial
    prices are close to a random walk, so this is a genuinely hard baseline
    to beat, not a strawman.
    """
    last_value = train_series.iloc[-1]
    return pd.Series([last_value] * steps, name="naive_forecast")


def rolling_mean_forecast(train_series, steps, window=20):
    """
    Rolling-mean baseline: forecast = average of the last `window`
    observations, repeated for every step of the horizon.
    """
    mean_value = train_series.iloc[-window:].mean()
    return pd.Series([mean_value] * steps, name="rolling_mean_forecast")


def compare_price_baselines(train_series, steps, window=20):
    """Convenience: return both baselines together for side-by-side comparison."""
    return {
        "naive": naive_forecast(train_series, steps),
        "rolling_mean": rolling_mean_forecast(train_series, steps, window=window),
    }


# ---------------------------------------------------------------------------
# Volatility baselines
# ---------------------------------------------------------------------------

def rolling_historical_volatility(returns_series, window=20):
    """
    In-sample rolling historical volatility: the trailing standard
    deviation of returns at each point in time. This is the standard naive
    baseline for volatility forecasting (as opposed to GARCH/LSTM), since
    it captures no forward-looking structure -- just "recent realized vol,
    carried forward."
    """
    return returns_series.rolling(window=window).std()


def historical_volatility_forecast(train_returns, steps, window=20):
    """
    Forecast: the most recent rolling historical volatility value, repeated
    for every step of the horizon (mirrors how the naive price forecast
    repeats the last value forward).
    """
    hist_vol = rolling_historical_volatility(train_returns, window=window)
    last_vol = hist_vol.dropna().iloc[-1]
    return pd.Series([last_vol] * steps, name="historical_vol_forecast")