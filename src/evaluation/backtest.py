"""
Walk-forward validation harness.

A single train/test split (what smoke_test.py does) tells you almost
nothing reliable -- a 5-day window can favor either model just by chance,
depending on whether that particular week happened to be calm or volatile.
Walk-forward backtesting repeats the same train -> forecast -> evaluate
cycle across many rolling windows through the full history, so you get a
distribution of results instead of one anecdote.

Two models are compared per track:
  price:      ARIMA vs. naive vs. rolling_mean
  volatility: GARCH(1,1) vs. historical_vol
"""

import numpy as np
import pandas as pd

from src.models.price.arima_sarima import grid_search_arima, forecast as arima_forecast
from src.models.volatility.garch import fit_garch, forecast_volatility
from src.models.volatility.egarch_gjr import (
    fit_gjr_garch,
    fit_egarch,
    forecast_volatility as forecast_volatility_egarch,
)
from src.models.baselines import (
    naive_forecast,
    rolling_mean_forecast,
    historical_volatility_forecast,
)
from src.evaluation.metrics import rmse, mae, volatility_metrics


def walk_forward_windows(n_obs, initial_train_size, horizon, step):
    """
    Yield (train_start, train_end, test_end) integer index bounds for each
    walk-forward window. Uses an EXPANDING window (train always starts at 0
    and grows) rather than a fixed-size rolling window, since more history
    generally helps ARIMA/GARCH and there's no strong reason to discard old
    data here. Stops once there isn't enough remaining data for a full
    `horizon`-length test window.
    """
    train_end = initial_train_size
    while train_end + horizon <= n_obs:
        yield 0, train_end, train_end + horizon
        train_end += step


def backtest_price(
    price_series,
    initial_train_size,
    horizon=5,
    step=20,
    arima_order=(1, 1, 1),
):
    """
    Run the walk-forward comparison for the price track: ARIMA (fixed
    order, for speed -- re-running a full grid search at every window is
    expensive and the order rarely changes much) vs. naive vs. rolling
    mean.

    Returns a DataFrame, one row per window, with RMSE/MAE for each model.
    A fixed order is used deliberately: refitting ARIMA's grid search at
    every one of potentially dozens of windows would be slow and, in
    practice, the selected (p,d,q) is usually stable across windows for a
    single ticker -- pass arima_order=None to instead re-run a small grid
    search each window if you want to verify that assumption.
    """
    price_series = price_series.reset_index(drop=True)
    rows = []

    for train_start, train_end, test_end in walk_forward_windows(
        len(price_series), initial_train_size, horizon, step
    ):
        train = price_series.iloc[train_start:train_end]
        actual = price_series.iloc[train_end:test_end].values

        try:
            if arima_order is None:
                order, model = grid_search_arima(
                    train, p_range=[0, 1], d_range=[1], q_range=[0, 1]
                )
            else:
                from statsmodels.tsa.arima.model import ARIMA
                order = arima_order
                model = ARIMA(train.reset_index(drop=True), order=order).fit()
            mean_fc, _ = arima_forecast(model, steps=horizon)
            arima_pred = mean_fc.values
        except Exception as exc:  # noqa: BLE001 -- skip unstable windows, don't crash the whole backtest
            print(f"[backtest_price] window ending at {train_end} failed: {exc}")
            continue

        naive_pred = naive_forecast(train, steps=horizon).values
        rolling_pred = rolling_mean_forecast(train, steps=horizon).values

        rows.append({
            "window_end": train_end,
            "arima_rmse": rmse(actual, arima_pred),
            "arima_mae": mae(actual, arima_pred),
            "naive_rmse": rmse(actual, naive_pred),
            "naive_mae": mae(actual, naive_pred),
            "rolling_mean_rmse": rmse(actual, rolling_pred),
            "rolling_mean_mae": mae(actual, rolling_pred),
        })

    return pd.DataFrame(rows)


def backtest_volatility(
    returns_series,
    initial_train_size,
    horizon=5,
    step=20,
    include_asymmetric=False,
):
    """
    Same idea as backtest_price, but for the volatility track: GARCH(1,1)
    vs. historical_vol, evaluated with RMSE/MAE (on vol scale) and QLIKE
    (on variance scale) via volatility_metrics().

    Pass include_asymmetric=True to also fit GJR-GARCH and EGARCH at every
    window and include them in the comparison -- off by default since it
    roughly triples the fitting time per window.
    """
    returns_series = returns_series.reset_index(drop=True)
    rows = []

    for train_start, train_end, test_end in walk_forward_windows(
        len(returns_series), initial_train_size, horizon, step
    ):
        train = returns_series.iloc[train_start:train_end]
        actual_returns = returns_series.iloc[train_end:test_end].values

        try:
            garch_fit = fit_garch(train)
            garch_vol = forecast_volatility(garch_fit, horizon=horizon).values
        except Exception as exc:  # noqa: BLE001
            print(f"[backtest_volatility] window ending at {train_end} failed (GARCH): {exc}")
            continue

        hist_vol = historical_volatility_forecast(train, steps=horizon).values

        garch_m = volatility_metrics(actual_returns, garch_vol)
        hist_m = volatility_metrics(actual_returns, hist_vol)

        row = {
            "window_end": train_end,
            "garch_rmse": garch_m["rmse_vol"],
            "garch_mae": garch_m["mae_vol"],
            "garch_qlike": garch_m["qlike"],
            "hist_vol_rmse": hist_m["rmse_vol"],
            "hist_vol_mae": hist_m["mae_vol"],
            "hist_vol_qlike": hist_m["qlike"],
        }

        if include_asymmetric:
            try:
                gjr_fit = fit_gjr_garch(train)
                gjr_vol = forecast_volatility(gjr_fit, horizon=horizon).values
                gjr_m = volatility_metrics(actual_returns, gjr_vol)
                row["gjr_rmse"] = gjr_m["rmse_vol"]
                row["gjr_mae"] = gjr_m["mae_vol"]
                row["gjr_qlike"] = gjr_m["qlike"]
            except Exception as exc:  # noqa: BLE001
                print(f"[backtest_volatility] window ending at {train_end} failed (GJR-GARCH): {exc}")

            try:
                egarch_fit = fit_egarch(train)
                # EGARCH has no analytic multi-step forecast -- must simulate
                egarch_vol = forecast_volatility_egarch(
                    egarch_fit, horizon=horizon, method="simulation"
                ).values
                egarch_m = volatility_metrics(actual_returns, egarch_vol)
                row["egarch_rmse"] = egarch_m["rmse_vol"]
                row["egarch_mae"] = egarch_m["mae_vol"]
                row["egarch_qlike"] = egarch_m["qlike"]
            except Exception as exc:  # noqa: BLE001
                print(f"[backtest_volatility] window ending at {train_end} failed (EGARCH): {exc}")

        rows.append(row)

    return pd.DataFrame(rows)

def backtest_lstm_volatility(
    returns_series,
    initial_train_size,
    horizon=5,
    step=20,
    lookback=20,
    epochs=50,
):
    """
    Walk-forward backtest for the LSTM volatility model, kept as a
    separate function from backtest_volatility rather than folded in as
    another `include_*` flag -- LSTM training is dramatically slower than
    GARCH-family fitting (tens of seconds vs. a fraction of a second per
    window), so it's kept independent to run on its own schedule instead
    of always re-training alongside the fast models.
    """
    from src.models.volatility.lstm_volatility import (
        fit_lstm_volatility,
        forecast_lstm_volatility,
    )

    returns_series = returns_series.reset_index(drop=True)
    rows = []

    for train_start, train_end, test_end in walk_forward_windows(
        len(returns_series), initial_train_size, horizon, step
    ):
        train = returns_series.iloc[train_start:train_end]
        actual_returns = returns_series.iloc[train_end:test_end].values

        try:
            model, lb = fit_lstm_volatility(
                train, lookback=lookback, epochs=epochs, verbose=0
            )
            lstm_vol = forecast_lstm_volatility(
                model, train, lb, horizon=horizon
            ).values
        except Exception as exc:  # noqa: BLE001
            print(f"[backtest_lstm_volatility] window ending at {train_end} failed: {exc}")
            continue

        hist_vol = historical_volatility_forecast(train, steps=horizon).values

        lstm_m = volatility_metrics(actual_returns, lstm_vol)
        hist_m = volatility_metrics(actual_returns, hist_vol)

        rows.append({
            "window_end": train_end,
            "lstm_rmse": lstm_m["rmse_vol"],
            "lstm_mae": lstm_m["mae_vol"],
            "lstm_qlike": lstm_m["qlike"],
            "hist_vol_rmse": hist_m["rmse_vol"],
            "hist_vol_mae": hist_m["mae_vol"],
            "hist_vol_qlike": hist_m["qlike"],
        })
        print(f"  window ending at {train_end} done "
              f"(lstm_qlike={lstm_m['qlike']:.4f}, hist_vol_qlike={hist_m['qlike']:.4f})")

    return pd.DataFrame(rows)


def summarize_backtest(results_df, model_a_col, model_b_col, lower_is_better=True):
    """
    Given a backtest results DataFrame and two metric column names,
    report: mean of each, and the win rate (fraction of windows where
    model_a beat model_b). This win-rate number is the actual defensible
    claim -- e.g. "ARIMA beat naive on 63% of windows" -- rather than a
    single aggregate RMSE, which can be dominated by one or two extreme
    windows.
    """
    if lower_is_better:
        wins = (results_df[model_a_col] < results_df[model_b_col]).mean()
    else:
        wins = (results_df[model_a_col] > results_df[model_b_col]).mean()

    return {
        "n_windows": len(results_df),
        f"{model_a_col}_mean": results_df[model_a_col].mean(),
        f"{model_b_col}_mean": results_df[model_b_col].mean(),
        "model_a_win_rate": round(wins, 3),
    }