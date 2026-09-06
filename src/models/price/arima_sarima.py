"""
ARIMA and SARIMA fitting with a small manual grid search over (p, d, q),
selected by AIC. Kept dependency-light (statsmodels only, no pmdarima) so
the environment stays simple.
"""
 
import warnings
 
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
 
 
def _drop_index_freq(series):
    """
    Reset to a plain integer index before fitting.
 
    statsmodels needs to know the step size between observations to
    generate out-of-sample forecast dates. A date index round-tripped
    through a CSV (or pulled from yfinance) usually has no `freq` set, and
    with market data (weekends/holidays skipped) statsmodels often can't
    infer one either -- `get_forecast()` then fails with "No supported
    index is available." Fitting on a plain 0..n-1 integer index sidesteps
    this: the model only needs evenly-spaced observations to do the math,
    not real calendar dates. The forecast this returns is indexed by
    integer step (0, 1, 2, ...), not by date -- map it back to real dates
    yourself if you need them for plotting/reporting.
    """
    return series.reset_index(drop=True)
 
 
def grid_search_arima(series, p_range, d_range, q_range, verbose=False):
    """
    Fit ARIMA(p, d, q) for every combination in the given ranges, return the
    (order, fitted_model) with the lowest AIC. Silently skips combinations
    that fail to converge.
    """
    series = _drop_index_freq(series)
    best_aic = np.inf
    best_order = None
    best_model = None
 
    for p in p_range:
        for d in d_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                    if verbose:
                        print(f"ARIMA({p},{d},{q}) AIC={fitted.aic:.2f}")
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                except Exception:  # noqa: BLE001 -- skip non-converging orders
                    continue
 
    if best_model is None:
        raise RuntimeError("No ARIMA order in the given grid converged.")
    return best_order, best_model
 
 
def fit_sarima(series, order, seasonal_order):
    """Fit a SARIMAX model with the given (p,d,q) and seasonal order."""
    series = _drop_index_freq(series)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
    return fitted
 
 
def forecast(fitted_model, steps):
    """Return (point_forecast, conf_int_df) for `steps` ahead."""
    fc = fitted_model.get_forecast(steps=steps)
    mean = fc.predicted_mean
    conf_int = fc.conf_int()
    return mean, conf_int