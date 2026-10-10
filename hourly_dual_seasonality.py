"""Feasibility test: dual-seasonality SARIMAX on HOURLY data.

Supervisor's request: a SARIMAX carrying both the daily cycle (period 24) and
the annual cycle (period 365). That cannot be done on the daily aggregates the
dissertation currently uses -- at daily resolution there is no 0-24 cycle left
to model, it has been averaged away. It requires going back to the hourly
series, with two sets of Fourier terms:

    period 24        K = 4   diurnal
    period 8766      K = 3   annual  (365.25 x 24 hours)

This script tests whether that is tractable and what it costs, on one region,
before committing the project to it.

The known obstacle: 46% of hourly rows are exact zeros (night). SARIMAX assumes
Gaussian errors, and a series censored at zero for half its observations
violates that badly. Two variants are therefore tested:

    ALL      every hour, zeros included
    DAYLIGHT daylight hours only (is_daylight == 1)

Run:  python hourly_dual_seasonality.py
"""
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.deterministic import Fourier, DeterministicProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

PES = 19                      # Crawley
ORDER = (2, 0, 1)
K_DAY, K_YEAR = 4, 3
P_DAY, P_YEAR = 24, 365.25 * 24
TRAIN_HOURS = 24 * 730        # two years, matching the daily protocol


def hourly_series(df, pes_id, daylight_only):
    g = df[df.pes_id == pes_id].sort_values("timestamp").set_index("timestamp")
    if daylight_only:
        g = g[g.is_daylight == 1]
    y = (g.generation_mw / g.installed_mwp).clip(0, 1).rename("cf").dropna()
    ts = y.index.tz_localize(None)
    # Elapsed CLOCK hours since the first observation. Using row position would
    # be wrong for the daylight-only variant, where skipped night hours make
    # position and elapsed time diverge and the annual cycle drift out of phase.
    hours = (ts - ts[0]).total_seconds().values / 3600.0
    # statsmodels needs a supported index to forecast from, so carry a plain
    # integer one and keep elapsed time separately.
    return y.reset_index(drop=True), hours


def fourier_exog(t):
    """Two Fourier bases evaluated at elapsed clock hours `t`."""
    cols = {}
    for k in range(1, K_DAY + 1):
        cols[f"day_sin{k}"] = np.sin(2 * np.pi * k * t / P_DAY)
        cols[f"day_cos{k}"] = np.cos(2 * np.pi * k * t / P_DAY)
    for k in range(1, K_YEAR + 1):
        cols[f"yr_sin{k}"] = np.sin(2 * np.pi * k * t / P_YEAR)
        cols[f"yr_cos{k}"] = np.cos(2 * np.pi * k * t / P_YEAR)
    return pd.DataFrame(cols, index=pd.RangeIndex(len(t)))


def run(df, daylight_only):
    label = "DAYLIGHT only" if daylight_only else "ALL hours"
    y, hours = hourly_series(df, PES, daylight_only)
    X = fourier_exog(hours)
    n_train = min(TRAIN_HOURS, int(len(y) * 0.67))
    ytr, Xtr = y.iloc[:n_train], X.iloc[:n_train]
    yte, Xte = y.iloc[n_train:], X.iloc[n_train:]

    print(f"\n--- {label} ---")
    print(f"  observations   : {len(y):,}  (train {len(ytr):,} / test {len(yte):,})")
    print(f"  exact zeros    : {100 * (y == 0).mean():.1f}%")

    t0 = time.time()
    # trend="c" is essential. Fourier terms are zero-mean by construction, so
    # with no constant the only source of level is the AR component, which
    # decays to zero over a long forecast -- the first run of this script
    # returned -99.8% annual error for exactly that reason.
    res = SARIMAX(ytr, exog=Xtr, order=ORDER, trend="c",
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False, maxiter=60)
    fit_s = time.time() - t0

    fc = res.forecast(steps=len(yte), exog=Xte)
    err = fc.values - yte.values
    mae = np.abs(err).mean()
    ann_err = 100 * (fc.mean() - yte.mean()) / yte.mean()

    print(f"  fit time       : {fit_s:.1f}s   AIC {res.aic:,.0f}")
    print(f"  hourly MAE     : {mae:.5f}")
    print(f"  annual error   : {ann_err:+.2f}%")
    print(f"  residual skew  : {pd.Series(res.resid).skew():.2f}  "
          f"(0 = symmetric; large values mean the Gaussian assumption fails)")
    return dict(variant=label, n=len(y), fit_s=fit_s, aic=res.aic,
                hourly_mae=mae, annual_err_pct=ann_err)


if __name__ == "__main__":
    df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"],
                     usecols=["timestamp", "pes_id", "generation_mw",
                              "installed_mwp", "is_daylight"])
    print("=" * 74)
    print("DUAL-SEASONALITY HOURLY SARIMAX — feasibility, Crawley")
    print(f"ARMA{ORDER} + Fourier(24, K={K_DAY}) + Fourier(8766, K={K_YEAR})")
    print("=" * 74)
    out = [run(df, False), run(df, True)]
    pd.DataFrame(out).to_csv("hourly_dual_seasonality.csv", index=False)
    print("\nSaved hourly_dual_seasonality.csv")
