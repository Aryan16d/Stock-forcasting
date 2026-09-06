"""
Return calculation and chronological train/test splitting.

This is the shared connector between the price-forecasting track and the
volatility-forecasting track: both consume log returns computed here, so
there's one canonical definition instead of two slightly different ones
living in separate model modules.
"""

import numpy as np


def log_returns(price_series):
    """Log returns, first row dropped (NaN)."""
    return np.log(price_series / price_series.shift(1)).dropna()


def train_test_split_series(series, test_size=0.15):
    """Chronological split -- never shuffle time series data."""
    n_test = int(len(series) * test_size)
    n_train = len(series) - n_test
    return series.iloc[:n_train], series.iloc[n_train:]
