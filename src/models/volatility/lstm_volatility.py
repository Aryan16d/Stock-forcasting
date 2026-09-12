"""
LSTM-based volatility forecasting.

Unlike GARCH-family models, an LSTM has no built-in parametric assumption
about how variance evolves -- it can in principle learn any pattern in the
data, at the cost of needing more data and being harder to interpret. This
is the model most likely to beat the historical-vol baseline where GARCH,
GJR-GARCH, and EGARCH all failed to, precisely because it isn't constrained
to their specific functional forms.

Target/feature choice: since true (latent) volatility is never directly
observed, squared returns are used as the standard realized-variance proxy
-- the same proxy `evaluation/metrics.py` uses for QLIKE. The model learns
to map a rolling window of past squared returns to the next day's squared
return.
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

# Squared daily returns are tiny (~1e-4 to 1e-3), which can make gradient
# descent converge slowly/unstably. Scaling by a fixed constant before
# training keeps inputs in a better-conditioned range -- same motivation as
# the *100 scaling in garch.py, just a different constant chosen for the
# squared (not raw) return scale.
SCALE = 1e4


def _make_sequences(squared_returns_scaled, lookback):
    """
    Turn a 1D array of scaled squared returns into supervised-learning
    sequences: X[i] = the `lookback` days before day i, y[i] = day i's
    value. Shape of X is (n_samples, lookback, 1) -- the trailing 1 is the
    single feature dimension Keras' LSTM layer expects.
    """
    X, y = [], []
    for i in range(lookback, len(squared_returns_scaled)):
        X.append(squared_returns_scaled[i - lookback:i])
        y.append(squared_returns_scaled[i])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y


def _build_model(lookback, units=32, dropout=0.1):
    """
    Small, deliberately simple LSTM: one recurrent layer + a dense output.
    A single layer with modest width (32 units) is a reasonable starting
    point for a ~400-observation training set -- a deeper/wider network
    would likely overfit before it out-forecasts historical volatility,
    given how little data is available compared to typical deep-learning
    datasets.
    """
    model = keras.Sequential([
        keras.Input(shape=(lookback, 1)),
        layers.LSTM(units, activation="tanh"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def fit_lstm_volatility(train_returns, lookback=20, units=32, epochs=50,
                         batch_size=16, verbose=0):
    """
    Fit an LSTM volatility model on a training returns series. Returns
    (model, lookback) -- lookback is returned alongside since
    forecast_lstm_volatility needs to know how many past values to feed in,
    and it's easy to accidentally mismatch this if it's tracked separately.
    """
    squared = (train_returns.values ** 2) * SCALE
    X, y = _make_sequences(squared, lookback)

    model = _build_model(lookback, units=units)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model, lookback


def forecast_lstm_volatility(model, train_returns, lookback, horizon=5):
    """
    Recursive multi-step forecast: predict one day ahead, append that
    prediction to the history, slide the window forward, predict the next
    day -- repeat `horizon` times.

    This is necessary because the LSTM only knows how to predict one step
    ahead; there's no closed-form multi-step formula the way there sort-of
    is for GARCH/GJR-GARCH. It's the same fundamental idea as EGARCH's
    simulation-based forecasting, just deterministic (feeding back the
    point prediction) rather than stochastic (averaging many simulated
    paths).

    Caveat worth knowing: errors can compound across steps since each
    prediction feeds off the previous one's (possibly imperfect) output --
    this is a real limitation of recursive forecasting, not specific to
    this implementation.
    """
    history = list((train_returns.values[-lookback:] ** 2) * SCALE)
    preds_scaled = []

    for _ in range(horizon):
        x_input = np.array(history[-lookback:]).reshape(1, lookback, 1)
        pred = model.predict(x_input, verbose=0)[0, 0]
        pred = max(pred, 1e-8)  # guard against a negative/zero prediction under the sqrt below
        preds_scaled.append(pred)
        history.append(pred)

    preds_variance = np.array(preds_scaled) / SCALE
    daily_vol = np.sqrt(preds_variance)
    return pd.Series(daily_vol, name="lstm_vol_forecast")