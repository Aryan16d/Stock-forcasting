"""
Stationarity diagnostics (Augmented Dickey-Fuller test) used to decide the
differencing order `d` before fitting ARIMA/SARIMA, and to confirm log
returns are stationary before feeding them to the GARCH model.
"""

from statsmodels.tsa.stattools import adfuller


def adf_test(series, alpha=0.05):
    """
    Run the Augmented Dickey-Fuller test.

    Returns a dict with the test statistic, p-value, critical values, and a
    boolean `is_stationary` (True when we reject the unit-root null at
    `alpha`).
    """
    series = series.dropna()
    stat, pvalue, usedlag, nobs, crit_values, icbest = adfuller(
        series, autolag="AIC"
    )
    return {
        "statistic": stat,
        "p_value": pvalue,
        "used_lag": usedlag,
        "n_obs": nobs,
        "critical_values": crit_values,
        "is_stationary": pvalue < alpha,
    }


def difference_until_stationary(series, max_diff=2, alpha=0.05):
    """
    Repeatedly first-difference `series` until the ADF test says it's
    stationary (or `max_diff` is reached). Returns (d, differenced_series,
    adf_result_at_each_step).
    """
    current = series.dropna()
    history = []
    for d in range(max_diff + 1):
        result = adf_test(current, alpha=alpha)
        history.append(result)
        if result["is_stationary"]:
            return d, current, history
        current = current.diff().dropna()
    return max_diff, current, history
