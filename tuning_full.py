"""Hyperparameter tuning for both tree models, both information sets.

Supervisor's instruction: tune regardless of whether it changes anything, and
if it changes nothing, prove it.

Protocol is identical to ml_backtest.py -- same walk-forward origins, same
target dates, same features. Only the hyperparameters vary. Reported on the
h = 1 complete-year window, which is what the method ladder uses.

Run:  python tuning_full.py
Out:  tuning_full.csv   (every configuration)
      tuning_summary.csv (best and default per model x information set)
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

from ml_backtest import load_daily, build_design, INITIAL_TRAIN, REFIT_EVERY

SEED, H = 42, 1

# The settings actually used in the dissertation, for reference rows.
DEFAULTS = {
    "RF":  dict(n_estimators=200, min_samples_leaf=2),
    "XGB": dict(n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8),
}

RF_GRID = [dict(n_estimators=n, min_samples_leaf=l, max_features=f)
           for n, l, f in itertools.product([300], [1, 2, 5, 10], [1.0, 0.5, 0.3])]
XGB_GRID = [dict(n_estimators=n, learning_rate=lr, max_depth=d,
                 subsample=0.8, colsample_bytree=0.8)
            for n, lr, d in itertools.product([500], [0.05, 0.02], [2, 3, 5, 8])]


def build(kind, params):
    if kind == "RF":
        return RandomForestRegressor(n_jobs=-1, random_state=SEED, **params)
    return XGBRegressor(n_jobs=-1, random_state=SEED, verbosity=0, **params)


def annual_error(daily, kind, params, weather_aware, h=H):
    """Annual generation error (%) over the walk-forward test window."""
    dates, n = daily.index, len(daily)
    X, y, _ = build_design(daily, h, weather_aware=weather_aware)
    origins = list(range(INITIAL_TRAIN, n - 1))
    tgts, preds = [], []
    for blk in range(0, len(origins), REFIT_EVERY):
        block = origins[blk:blk + REFIT_EVERY]
        train = X.index < dates[block[0]]
        if train.sum() < 100:
            continue
        m = build(kind, params).fit(X[train], y[train])
        td = [dates[o + h - 1] for o in block
              if o + h - 1 < n and dates[o + h - 1] in X.index]
        if not td:
            continue
        tgts += td
        preds += list(m.predict(X.loc[td]))
    if not tgts:
        return np.nan
    truth = y.loc[tgts]
    return 100 * (np.mean(preds) - truth.mean()) / truth.mean()


if __name__ == "__main__":
    df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])
    regions = df[["pes_id", "region_name"]].drop_duplicates().values
    panels = {int(p): load_daily(df, p) for p, _ in regions}

    rows = []
    for wx in (False, True):
        info = "weather-aware" if wx else "weather-blind"
        for kind, grid in (("RF", RF_GRID), ("XGB", XGB_GRID)):
            for params in [DEFAULTS[kind]] + grid:
                tag = "default" if params is DEFAULTS[kind] else "grid"
                errs = [annual_error(panels[p], kind, params, wx)
                        for p, _ in regions]
                e = np.array(errs, dtype=float)
                rows.append({"model": kind, "info_set": info, "source": tag,
                             "params": str(params),
                             "MAE": np.abs(e).mean(), "SD": e.std(ddof=1),
                             "mean_error": e.mean()})
                print(f"  {info:14s} {kind:4s} {tag:7s} "
                      f"MAE {rows[-1]['MAE']:6.2f}%  SD {rows[-1]['SD']:5.2f}"
                      f"  {str(params)[:62]}", flush=True)

    T = pd.DataFrame(rows)
    T.to_csv("tuning_full.csv", index=False)

    out = []
    for (kind, info), g in T.groupby(["model", "info_set"], sort=False):
        d = g[g.source == "default"].iloc[0]
        b = g[g.source == "grid"].sort_values("MAE").iloc[0]
        out.append({"model": kind, "info_set": info,
                    "default_MAE": d.MAE, "best_tuned_MAE": b.MAE,
                    "improvement_pp": d.MAE - b.MAE,
                    "best_params": b.params})
    S = pd.DataFrame(out)
    S.to_csv("tuning_summary.csv", index=False)
    print("\n" + "=" * 96)
    print(S.round(3).to_string(index=False))
    print("\n  statistical baseline (SARIMA/Fourier), weather-blind: MAE 2.91%")
