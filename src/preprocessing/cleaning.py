"""
z-score based outlier detection and cleaning for price series.

The resume bullet this backs: "Engineered a z-score based outlier-detection
and data-cleaning pipeline... to ensure reliable model inputs." Concretely:
we flag points whose *rolling* z-score exceeds a threshold (rolling, not
global, so we catch local spikes rather than only extreme values relative to
the whole multi-year history) and repair them with linear interpolation.
"""

import numpy as np
import pandas as pd


def rolling_zscore(series, window=20, min_periods=5):
    """Rolling z-score of `series` using a trailing window."""
    roll_mean = series.rolling(window=window, min_periods=min_periods).mean()
    roll_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    # avoid divide-by-zero on flat windows
    roll_std = roll_std.replace(0, np.nan)
    z = (series - roll_mean) / roll_std
    return z


def detect_outliers(series, window=20, threshold=3.0):
    """Boolean mask, True where |rolling z-score| exceeds `threshold`."""
    z = rolling_zscore(series, window=window)
    mask = z.abs() > threshold
    return mask.fillna(False)


def clean_series(df, column="Close", window=20, threshold=3.0):
    """
    Return (cleaned_df, report) where flagged outliers in `column` are
    replaced via linear interpolation over the flagged points.
    """
    df = df.copy()
    mask = detect_outliers(df[column], window=window, threshold=threshold)

    cleaned = df[column].copy()
    cleaned[mask] = np.nan
    cleaned = cleaned.interpolate(method="linear", limit_direction="both")

    df[f"{column}_raw"] = df[column]
    df[column] = cleaned

    report = {
        "n_points": len(df),
        "n_outliers": int(mask.sum()),
        "pct_outliers": round(100 * mask.sum() / len(df), 3),
        "outlier_dates": df.index[mask].tolist(),
    }
    return df, report
