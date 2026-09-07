"""
Evaluation metrics, split by track.

Price forecasts are evaluated on point-value error (RMSE/MAE). Volatility
forecasts additionally use QLIKE, the standard loss function in the
volatility-forecasting literature -- plain RMSE against squared returns is
noisy and can misleadingly favor models that just predict low, flat
volatility, since squared returns are an extremely noisy proxy for the
true (unobserved) variance.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared / price metrics
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Volatility-specific metrics
# ---------------------------------------------------------------------------

def qlike(realized_variance, predicted_variance, eps=1e-12):
    """
    QLIKE loss: mean of [ log(pred_var) + realized_var / pred_var ].

    Lower is better. Unlike RMSE on raw variance, QLIKE penalizes
    under-prediction of variance more heavily than over-prediction, which
    matches how volatility forecasts are actually used in practice (e.g.
    underestimating risk is more costly than overestimating it in most
    risk-management contexts). Requires variance, not standard deviation --
    square your volatility forecasts before calling this.
    """
    realized_variance = np.asarray(realized_variance, dtype=float)
    predicted_variance = np.asarray(predicted_variance, dtype=float)
    predicted_variance = np.clip(predicted_variance, eps, None)
    loss = np.log(predicted_variance) + realized_variance / predicted_variance
    return float(np.mean(loss))


def volatility_metrics(realized_returns, predicted_volatility):
    """
    Convenience wrapper: given realized returns over the forecast horizon
    and a predicted daily volatility (std dev, NOT variance) series of the
    same length, compute RMSE, MAE (both on volatility/std-dev scale) and
    QLIKE (on variance scale, as the metric requires).

    `realized_returns` should be the actual returns observed during the
    forecast horizon; squared returns serve as the (noisy) realized-variance
    proxy, which is standard practice when higher-frequency (e.g. intraday)
    data isn't available to compute a cleaner realized-variance estimate.
    """
    realized_returns = np.asarray(realized_returns, dtype=float)
    predicted_volatility = np.asarray(predicted_volatility, dtype=float)

    realized_variance = realized_returns ** 2
    realized_volatility = np.sqrt(realized_variance)
    predicted_variance = predicted_volatility ** 2

    return {
        "rmse_vol": rmse(realized_volatility, predicted_volatility),
        "mae_vol": mae(realized_volatility, predicted_volatility),
        "qlike": qlike(realized_variance, predicted_variance),
    }