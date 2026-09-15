# Stock Forecasting v2: Price & Volatility Forecasting on AAPL

An end-to-end, modular (not notebook-based) pipeline comparing multiple approaches to
two distinct forecasting problems: **price/return direction** and **volatility**, evaluated
with rigorous walk-forward backtesting rather than a single train/test split.

## Why two separate tracks

Price forecasting and volatility forecasting are genuinely different problems:

| | Price forecasting | Volatility forecasting |
|---|---|---|
| What's predicted | Level/direction of price or return | Magnitude of price swings |
| Models used here | ARIMA(1,1,1), naive, rolling mean | GARCH(1,1), GJR-GARCH, EGARCH, LSTM, rolling historical vol |
| Underlying reality | Prices are close to a random walk -- direction is genuinely hard to forecast | Volatility clusters over time (calm periods stay calm, turbulent periods stay turbulent) -- more tractable |

This project treats them as separate tracks with separate baselines and separate metrics,
rather than one blended pipeline.

## Architecture

Data (yfinance, cached; synthetic fallback)
|
Cleaning (rolling z-score outlier detection + interpolation)
|
Returns + stationarity check (ADF test)
|
|
| |
Price track Volatility track
(ARIMA) (GARCH / GJR-GARCH / EGARCH / LSTM)
| |
|__ Baselines (naive, rolling mean) __ Baseline (rolling historical vol)
|
Walk-forward backtest (expanding window, RMSE/MAE/QLIKE)
|
[Planned] Live data (Finnhub) -> Streamlit dashboard


## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

**Windows note:** if `pip install tensorflow` fails with an OSError about a missing nested
file path, this is Windows' 260-character path length limit, not a real installation
problem. Either enable long paths (`New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem"
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`, run as Administrator, then
restart) or keep the project at a short path (e.g. `C:\dev\stock_forecasting_v2` rather than
nested several folders deep).

Copy `.env.example` to `.env` and add a free Finnhub API key (finnhub.io) if/when using the
live-data pieces:

FINNHUB_API_KEY=your_key_here


## Running it

```bash
python smoke_test.py          # end-to-end sanity check on one train/test split
python run_backtest.py        # walk-forward backtest: ARIMA + GARCH/GJR/EGARCH vs. baselines
python run_lstm_backtest.py   # walk-forward backtest: LSTM volatility (slow -- trains per window)
python check_leverage.py      # statistical significance of the leverage/asymmetry parameter
```

## Data

Ticker: **AAPL**, 2022-01-01 to 2024-01-01 (501 trading days). Pulled via `yfinance`, cached
to `data/AAPL.csv` after the first run. Falls back to a calibrated synthetic (GBM-based)
series if the network/API is unavailable, so the pipeline always runs.

## Key findings

### Price track: ARIMA has a small, real edge over naive

Walk-forward backtest, 10 expanding-window iterations, 5-day forecast horizon:

| Model | Mean RMSE | Win rate vs. ARIMA |
|---|---|---|
| **ARIMA(1,1,1)** | **4.308** | -- |
| Naive (random walk) | 4.322 | ARIMA wins 60% of windows |
| Rolling mean (20-day) | 5.623 | ARIMA wins 70% of windows |

This is the expected, credible pattern for financial price forecasting: since prices are
close to a random walk, ARIMA's edge over naive is small but real, and both comfortably beat
the (poorly-suited) rolling mean baseline. A model that dramatically beat naive on this data
would be a red flag for a lucky/overfit result, not a sign of a better model.

### Volatility track: every model tried underperformed a simple baseline

Four fundamentally different modeling approaches were backtested against a 20-day rolling
historical volatility baseline, using QLIKE (the standard volatility-forecasting loss
function; lower/more negative is better):

| Model | Mean QLIKE | Win rate vs. baseline |
|---|---|---|
| GJR-GARCH | -7.872 | 20% |
| EGARCH | -7.773 | 30% |
| GARCH(1,1) | -7.865 | 30% |
| LSTM | -7.885 | 40% |
| **Historical vol (baseline)** | **-8.025** | -- |

**All four models -- a linear parametric model, two asymmetric extensions of it, and a
nonlinear deep learning model -- lost to a parameter-free rolling standard deviation.** LSTM
came closest but still didn't win outright.

### Leverage effect: found by EGARCH, not by GJR-GARCH, and didn't help forecasting anyway

Fitting both asymmetric models on the full training set (425 observations) to check whether
a leverage effect (negative shocks increasing volatility more than positive ones) is actually
present:

- **GJR-GARCH:** asymmetry parameter gamma = 0.102, p = 0.128 -- **not statistically
  significant.**
- **EGARCH:** asymmetry parameter gamma = -0.139, p < 0.001 -- **highly significant**, and in
  the expected direction for a leverage effect.

The two model specifications disagree on whether an asymmetric effect is even present,
because they parameterize the asymmetry term differently (additive vs. multiplicative in log
variance). More importantly: **even though EGARCH found a highly significant leverage
effect, it still underperformed the historical-vol baseline in walk-forward testing** --
demonstrating that a statistically significant in-sample effect doesn't guarantee improved
out-of-sample forecast accuracy.

### What this suggests

The consistent failure of every volatility model -- regardless of modeling paradigm -- to
beat a simple baseline points toward a structural explanation rather than a weakness in any
one approach: walk-forward training windows here are relatively short (roughly 300-480
observations), which is likely insufficient for GARCH-family MLE or an LSTM to reliably
out-forecast a parameter-free rolling statistic. A 20-day rolling historical volatility
baseline already captures the core phenomenon (volatility clustering) that these more complex
models are designed to exploit, making it a genuinely difficult baseline to beat, not a
strawman.

## Limitations & future work

- **Single ticker (AAPL), single time period.** Findings may not generalize to other
  tickers, asset classes, or market regimes -- worth testing on a small basket of tickers
  before treating any of the above as a general claim.
- **Sample size.** All volatility models were trained on 300-480 observations per window;
  it's plausible (and the leading hypothesis here) that more training data would change the
  GARCH-family and LSTM results.
- **Short forecast horizon (5 days).** More complex models might show a clearer advantage at
  longer horizons, where naive/rolling approaches decay faster.
- **Live data integration (planned, not yet built):** a Finnhub-based live feed
  (`src/data/live_feed.py`) and rolling buffer (`src/data/streaming_store.py`) to move beyond
  historical backfill.
- **Serving layer (planned, not yet built):** a Streamlit dashboard (`app/dashboard.py`)
  showing live price + forecast bands, refreshed as new data arrives.
- **LSTM architecture was deliberately kept small** (single 32-unit layer) given the limited
  training data; a wider hyperparameter search was not performed and might change the result.

## Project structure
stock_forecasting_v2
├── src/
│ ├── data/ # historical + (planned) live data loading
│ ├── preprocessing/ # cleaning, returns, stationarity
│ ├── models/
│ │ ├── price/ # ARIMA/SARIMA
│ │ ├── volatility/ # GARCH, GJR-GARCH, EGARCH, LSTM
│ │ └── baselines.py # naive, rolling mean, historical vol
│ ├── evaluation/ # metrics (RMSE/MAE/QLIKE), walk-forward backtest harness
│ └── serving/ # (planned) live re-forecast trigger
├── app/ # (planned) Streamlit dashboard
├── smoke_test.py # end-to-end sanity check
├── run_backtest.py # ARIMA + GARCH-family walk-forward backtest
├── run_lstm_backtest.py # LSTM walk-forward backtest
└── check_leverage.py # leverage-effect significance check