"""Does hyperparameter tuning change the weather-blind conclusion?

The headline claim is that, given only calendar and lagged generation, the
dynamic harmonic regression baseline (2.91% mean absolute annual error) beats
both tree ensembles (7.68% and 9.18%). The obvious objection is that the trees
were run at fixed default-ish settings and might close the gap if tuned.

This tests that directly. Same walk-forward protocol, same origins, same
targets, weather-blind information set, h = 1 -- only the hyperparameters vary.
If the best configuration in the grid still loses to 2.91%, the conclusion is
about the information set, not about tuning effort.

Run:  python tuning_check.py
Out:  tuning_check.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

from ml_backtest import load_daily, build_design, INITIAL_TRAIN, REFIT_EVERY

SEED = 42
H = 1

RF_GRID = [
    dict(n_estimators=n, min_samples_leaf=l, max_features=f)
    for n, l, f in itertools.product([300], [1, 2, 5], [1.0, 0.5])
]
XGB_GRID = [
    dict(n_estimators=n, learning_rate=lr, max_depth=d, subsample=0.8,
         colsample_bytree=0.8)
    for n, lr, d in itertools.product([500], [0.05, 0.02], [3, 5, 8])
]


def walk_forward(daily, make, h=H):
    """Identical protocol to ml_backtest.walk_forward, weather-blind, one horizon."""
    dates, n = daily.index, len(daily)
    X, y, _ = build_design(daily, h, weather_aware=False)
    origins = list(range(INITIAL_TRAIN, n - 1))
    rows = []
    for blk in range(0, len(origins), REFIT_EVERY):
        block = origins[blk:blk + REFIT_EVERY]
        train = X.index < dates[block[0]]
        if train.sum() < 100:
            continue
        model = make()
        model.fit(X[train], y[train])
        tgt = [dates[o + h - 1] for o in block
               if o + h - 1 < n and dates[o + h - 1] in X.index]
        if not tgt:
            continue
        for d, p in zip(tgt, model.predict(X.loc[tgt])):
            rows.append((d, float(y.loc[d]), float(p)))
    P = pd.DataFrame(rows, columns=["target", "y_true", "y_pred"])
    # annual generation error: what the dissertation actually reports
    return 100 * (P.y_pred.mean() - P.y_true.mean()) / P.y_true.mean()


if __name__ == "__main__":
    df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])
    regions = df[["pes_id", "region_name"]].drop_duplicates().values

    out = []
    for kind, grid in [("RF", RF_GRID), ("XGB", XGB_GRID)]:
        for params in grid:
            errs = []
            for pes_id, name in regions:
                daily = load_daily(df, pes_id)
                make = (lambda p=params: RandomForestRegressor(
                            n_jobs=-1, random_state=SEED, **p)) if kind == "RF" else \
                       (lambda p=params: XGBRegressor(
                            n_jobs=-1, random_state=SEED, verbosity=0, **p))
                errs.append(walk_forward(daily, make))
            mae = float(np.mean(np.abs(errs)))
            out.append({"model": kind, **params, "mean_abs_annual_err_pct": mae})
            print(f"  {kind:4s} {str(params):72s} {mae:6.2f}%")

    T = pd.DataFrame(out).sort_values("mean_abs_annual_err_pct")
    T.to_csv("tuning_check.csv", index=False)

    print("\n" + "=" * 78)
    print("BEST CONFIGURATION IN EACH FAMILY, weather-blind, h = 1")
    print("=" * 78)
    for kind in ("RF", "XGB"):
        b = T[T.model == kind].iloc[0]
        print(f"  {kind:4s} best {b.mean_abs_annual_err_pct:5.2f}%   "
              f"(reported in dissertation: "
              f"{'7.68' if kind == 'RF' else '9.18'}%)")
    print("\n  dynamic harmonic regression baseline: 2.91%")
    print("  weather-aware trees:                  1.35% / 1.42%")
