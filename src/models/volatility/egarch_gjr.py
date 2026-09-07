"""
Asymmetric GARCH-family models: EGARCH and GJR-GARCH.

Plain GARCH(1,1) treats a positive and negative return of the same size as
having an identical effect on future volatility. In real equity markets
that's usually false -- there's a well-documented "leverage effect" where
a negative shock (bad news) tends to increase volatility MORE than a
positive shock of the same magnitude. GJR-GARCH and EGARCH both add a term
that captures this asymmetry; the difference is mainly in how that
asymmetry term enters the variance equation (additive for GJR, inside a
log for EGARCH, which also guarantees variance stays positive without
needing to constrain parameters).

Same interface as garch.py (fit_*, forecast_volatility, conditional_volatility)
so these drop straight into backtest.py in place of fit_garch.
"""

from arch import arch_model


def fit_gjr_garch(returns, p=1, o=1, q=1, dist="normal"):
    """
    Fit a GJR-GARCH(p, o, q) model. The `o` term is the asymmetry order --
    o=1 is standard and adds one extra parameter that activates only on
    negative return days, letting the model estimate directly how much
    "extra" volatility follows bad news vs. good news.
    """
    scaled = returns * 100
    model = arch_model(scaled, vol="Garch", p=p, o=o, q=q, dist=dist, mean="Constant")
    fitted = model.fit(disp="off")
    return fitted


def fit_egarch(returns, p=1, o=1, q=1, dist="normal"):
    """
    Fit an EGARCH(p, o, q) model. Models log-variance rather than variance
    directly, which has two practical advantages over GJR-GARCH: variance
    is guaranteed positive with no parameter constraints needed, and the
    asymmetry term is naturally multiplicative rather than additive.
    """
    scaled = returns * 100
    model = arch_model(scaled, vol="EGARCH", p=p, o=o, q=q, dist=dist, mean="Constant")
    fitted = model.fit(disp="off")
    return fitted


def forecast_volatility(fitted_result, horizon=10, method="analytic", simulations=1000):
    """
    Same as garch.py's version, with one addition: EGARCH has no closed-form
    multi-step-ahead formula (its log-variance formulation doesn't support
    the analytic recursion GARCH/GJR-GARCH use), so the `arch` package
    raises "Analytic forecasts not available for horizon > 1" for EGARCH
    whenever horizon > 1. The fix is to simulate forward instead: draw many
    random future paths consistent with the fitted model and average their
    variance at each step. Pass method="simulation" for EGARCH forecasts
    with horizon > 1; GARCH/GJR-GARCH can stay on the faster "analytic"
    method.
    """
    fc = fitted_result.forecast(
        horizon=horizon, reindex=False, method=method, simulations=simulations
    )
    variance_scaled = fc.variance.iloc[-1]
    daily_vol = (variance_scaled ** 0.5) / 100
    return daily_vol


def conditional_volatility(fitted_result):
    """In-sample fitted conditional volatility series, unscaled."""
    return fitted_result.conditional_volatility / 100


def leverage_effect_significant(fitted_result, alpha=0.05):
    """
    Convenience check: is the asymmetry ('gamma'/'o') parameter
    statistically significant? If not, the leverage effect isn't detectably
    present for this ticker/period, and plain GARCH(1,1) may be just as good
    -- worth checking and reporting rather than assuming asymmetry always
    helps.
    """
    pvalues = fitted_result.pvalues
    gamma_params = [p for p in pvalues.index if p.startswith("gamma")]
    if not gamma_params:
        return None  # no asymmetry term in this fit (shouldn't happen if o=1 was used)
    return {p: (pvalues[p], pvalues[p] < alpha) for p in gamma_params}