"""
GARCH-family volatility modeling on log returns via the `arch` package.
Returns are scaled by 100 before fitting (standard practice -- keeps the
optimizer's numbers in a well-conditioned range) and the forecasted
variance is scaled back down before being reported.
"""

from arch import arch_model


def fit_garch(returns, p=1, q=1, dist="normal", vol="Garch"):
    """
    Fit a GARCH(p, q) model on `returns` (a pandas Series of log returns,
    NOT scaled). Returns the fitted result object.
    """
    scaled = returns * 100
    model = arch_model(scaled, vol=vol, p=p, q=q, dist=dist, mean="Constant")
    fitted = model.fit(disp="off")
    return fitted


def forecast_volatility(fitted_result, horizon=10):
    """
    Return a pandas Series of forecasted daily volatility (annualized-free,
    i.e. daily std dev of returns in original, unscaled units) for the next
    `horizon` days.
    """
    fc = fitted_result.forecast(horizon=horizon, reindex=False)
    variance_scaled = fc.variance.iloc[-1]  # last origin's horizon forecasts
    daily_vol = (variance_scaled ** 0.5) / 100  # undo the *100 scaling
    return daily_vol


def conditional_volatility(fitted_result):
    """In-sample fitted conditional volatility series, unscaled."""
    return fitted_result.conditional_volatility / 100
