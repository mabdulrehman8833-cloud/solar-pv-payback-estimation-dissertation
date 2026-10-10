"""
Walk-forward (expanding window) backtest for the weather-blind
dynamic harmonic regression baseline:  ARMA(2,0,1) + Fourier(365.25, K=3)

Design
------
  Initial train size : 730 days (2 full years)
  Test period        : remaining ~366 days (final year)
  Scheme             : expanding window
  Refit cadence      : re-estimate parameters every 30 days;
                       between refits the model state is updated with new
                       observations via .append(refit=False)  (cheap, standard)
  Horizons           : h = 1, 7, 30 days ahead
  Benchmark          : seasonal naive (value on the same day last year)
  Metrics            : MAE, RMSE, R2, MASE (scaled by in-sample seasonal-naive MAE)

Environment
-----------
Results shift slightly across library versions (SARIMAX optimiser internals
changed between statsmodels 0.14 and 0.15). Pin the environment so figures
reported in the dissertation stay reproducible. In a Colab/Jupyter CELL, run:

    !pip install statsmodels==0.14.6 pandas==2.2.3 pymannkendall

then restart the runtime before executing this script. The `!` magic only
works inside a notebook cell, not in a plain .py file -- so the equivalent
install is done programmatically below when PIN_ENV is True.
"""
import subprocess, sys

# Set True to force the pinned versions at import time (Colab-safe).
# Leave False if the environment is already pinned, e.g. via requirements.txt.
PIN_ENV  = False
PINS     = ["statsmodels==0.14.6", "pandas==2.2.3", "pymannkendall"]

if PIN_ENV:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS], check=False)

import pandas as pd, numpy as np, time, json
import statsmodels, scipy
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import Fourier
import warnings; warnings.filterwarnings("ignore")

# Stamp the environment into every run's output, so any saved log records
# exactly which versions produced the numbers in it.
print(f"env: python {sys.version.split()[0]} | pandas {pd.__version__} | "
      f"numpy {np.__version__} | scipy {scipy.__version__} | "
      f"statsmodels {statsmodels.__version__}")

df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])

REGION_NAMES = (df[["pes_id", "region_name"]].drop_duplicates()
                  .set_index("pes_id")["region_name"].to_dict())

ORDER          = (2, 0, 1)
FOURIER_K      = 3
PERIOD         = 365.25
INITIAL_TRAIN  = 730
REFIT_EVERY    = 30
HORIZONS       = [1, 7, 30]


def daily_capacity_factor(pes_id):
    reg = df[df["pes_id"] == pes_id].sort_values("timestamp").set_index("timestamp")
    d = pd.DataFrame({
        "generation_mwh": reg["generation_mw"].resample("D").sum(),
        "installed_mwp":  reg["installed_mwp"].resample("D").mean(),
    }).dropna()
    y = (d["generation_mwh"] / (d["installed_mwp"] * 24)).clip(0, 1)
    y.index = y.index.tz_localize(None)
    y.index.freq = "D"
    return y.rename("cf")


def backtest_region(y):
    fourier = Fourier(period=PERIOD, order=FOURIER_K)
    X_all = fourier.in_sample(y.index)

    n = len(y)
    max_h = max(HORIZONS)
    records = []
    res = None
    n_refits = 0
    t0 = time.time()

    for i, origin in enumerate(range(INITIAL_TRAIN, n - 1)):
        # ---- expanding window: everything strictly before `origin` is training
        if i % REFIT_EVERY == 0:
            mod = SARIMAX(y.iloc[:origin], exog=X_all.iloc[:origin], order=ORDER,
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            n_refits += 1
        else:
            # update state with the single newest observation, keep parameters fixed
            res = res.append(y.iloc[origin-1:origin], exog=X_all.iloc[origin-1:origin],
                             refit=False)

        steps = min(max_h, n - origin)
        if steps < 1:
            break
        fc = res.forecast(steps=steps, exog=X_all.iloc[origin:origin + steps])

        for h in HORIZONS:
            if h <= steps:
                records.append({
                    "origin":   y.index[origin - 1],
                    "target":   y.index[origin + h - 1],
                    "h":        h,
                    "y_true":   y.iloc[origin + h - 1],
                    "y_pred":   fc.iloc[h - 1],
                })

    elapsed = time.time() - t0
    return pd.DataFrame(records), n_refits, elapsed


def metrics(sub, y_full, scale_mae):
    err = sub["y_pred"] - sub["y_true"]
    mae  = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    ss_res = (err ** 2).sum()
    ss_tot = ((sub["y_true"] - sub["y_true"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    # seasonal naive benchmark on the same targets
    snaive = y_full.reindex(sub["target"] - pd.Timedelta(days=365)).values
    mask = ~np.isnan(snaive)
    sn_mae = np.abs(snaive[mask] - sub["y_true"].values[mask]).mean()
    return dict(MAE=mae, RMSE=rmse, R2=r2, MASE=mae / scale_mae, SNaive_MAE=sn_mae)


summary = []
for pes_id, name in REGION_NAMES.items():
    y = daily_capacity_factor(pes_id)
    preds, n_refits, elapsed = backtest_region(y)

    # MASE scale: in-sample seasonal-naive MAE over the initial training window
    train = y.iloc[:INITIAL_TRAIN]
    scale_mae = (train - train.shift(365)).abs().mean()

    for h in HORIZONS:
        sub = preds[preds["h"] == h]
        m = metrics(sub, y, scale_mae)
        summary.append(dict(pes_id=pes_id, region=name, h=h, n=len(sub), **m))

    preds.to_csv(f"backtest_preds_pes{pes_id}.csv", index=False)
    print(f"{name:32s} refits={n_refits:3d}  origins={preds['origin'].nunique():4d}  {elapsed:5.1f}s")

s = pd.DataFrame(summary)
s.to_csv("backtest_summary.csv", index=False)

pd.set_option("display.width", 200)
print()
print("="*110)
print("WALK-FORWARD BACKTEST — weather-blind ARMA(2,0,1) + Fourier(K=3), expanding window, refit every 30d")
print("="*110)
print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
print("MEAN ACROSS REGIONS BY HORIZON")
print(s.groupby("h")[["MAE","RMSE","R2","MASE","SNaive_MAE"]].mean().to_string(float_format=lambda v: f"{v:.4f}"))
