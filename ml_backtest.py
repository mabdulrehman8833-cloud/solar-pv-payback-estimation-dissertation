"""
================================================================================
Walk-forward backtest for the tree-based models — identical protocol to the
SARIMA baseline, so every model is scored on the SAME forecast origins and the
SAME target dates.

Three information sets are compared (this is the point of the design):

  A. SARIMA / Fourier      weather-blind, statistical      [already run]
  B. RF & XGB weather-blind calendar + lagged generation only
  C. RF & XGB weather-aware calendar + lags + weather on the target date

B answers "is a tree model better than SARIMA given the SAME information?"
C answers "how much does knowing the weather add?" — the thesis question.

C uses recorded weather for the target date, i.e. perfect weather foresight.
That is an UPPER BOUND on operational accuracy, not an operational forecast,
and must be reported as such. It is the correct basis for comparison against a
vendor formula (which likewise assumes typical/known conditions rather than
forecasting the weather), and for the 25-year payback calculation, where
climatological weather — not a weather forecast — is the relevant input.

Leakage control: for a target date j at horizon h, the most recent observable
generation value is at j-h. All lag features respect that, and the training set
for an origin contains only targets that fall strictly before that origin.
================================================================================
"""
import sys, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

import sklearn, xgboost
print(f"env: pandas {pd.__version__} | numpy {np.__version__} | "
      f"sklearn {sklearn.__version__} | xgboost {xgboost.__version__}")

# ------------------------------------------------------------------ config ---
DATA_FILE     = "solar_weather_dataset.csv"
INITIAL_TRAIN = 730          # identical to the SARIMA backtest
REFIT_EVERY   = 30           # identical
HORIZONS      = [1, 7, 30]   # identical
SEED          = 42

WEATHER_COLS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
                "temperature_2m", "relative_humidity_2m", "cloud_cover",
                "wind_speed_10m", "precipitation"]


def load_daily(df, pes_id):
    """Hourly regional data -> daily capacity factor + daily-mean weather."""
    reg = df[df["pes_id"] == pes_id].sort_values("timestamp").set_index("timestamp")
    daily = pd.DataFrame({
        "generation_mwh": reg["generation_mw"].resample("D").sum(),
        "installed_mwp":  reg["installed_mwp"].resample("D").mean(),
    })
    for c in WEATHER_COLS:
        daily[c] = reg[c].resample("D").mean()
    daily["cf"] = (daily["generation_mwh"] / (daily["installed_mwp"] * 24)).clip(0, 1)
    daily = daily.dropna()
    daily.index = daily.index.tz_localize(None)
    return daily


def build_design(daily, h, weather_aware):
    """Feature matrix indexed by TARGET date, for a fixed horizon h.

    Every feature must be knowable at the forecast origin (= target - h),
    except the target-date weather in the weather-aware set, which is the
    deliberate perfect-foresight assumption documented above.
    """
    cf  = daily["cf"]
    doy = daily.index.dayofyear.values
    X = pd.DataFrame(index=daily.index)

    # deterministic calendar (known arbitrarily far ahead)
    X["day_sin"]   = np.sin(2 * np.pi * doy / 365.25)
    X["day_cos"]   = np.cos(2 * np.pi * doy / 365.25)
    X["month_sin"] = np.sin(2 * np.pi * daily.index.month.values / 12)
    X["month_cos"] = np.cos(2 * np.pi * daily.index.month.values / 12)

    # lagged generation — shifted by h so nothing after the origin is used
    for k in range(0, 7):
        X[f"cf_lag{h + k}"] = cf.shift(h + k)
    X["cf_roll7"]   = cf.shift(h).rolling(7).mean()
    X["cf_roll30"]  = cf.shift(h).rolling(30).mean()
    X["cf_lag365"]  = cf.shift(365)

    if weather_aware:
        for c in WEATHER_COLS:
            X[c] = daily[c]                      # weather ON the target date
        tot = (daily["direct_radiation"] + daily["diffuse_radiation"]).replace(0, np.nan)
        X["beam_fraction"] = (daily["direct_radiation"] / tot).fillna(0.0)

    y = cf.copy()
    ok = X.notna().all(axis=1)
    return X[ok], y[ok], ok


def make_model(kind):
    if kind == "rf":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                     n_jobs=-1, random_state=SEED)
    return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=SEED, n_jobs=-1, verbosity=0)


def walk_forward(daily, kind, weather_aware):
    """Expanding window, refit every REFIT_EVERY origins, same origins as SARIMA.

    Predictions are made in blocks between refits (one .predict call per block
    rather than per origin) -- identical results, far less overhead.
    """
    dates = daily.index
    n = len(daily)
    rows = []

    for h in HORIZONS:
        X, y, _ = build_design(daily, h, weather_aware)
        origins = list(range(INITIAL_TRAIN, n - 1))

        for blk_start in range(0, len(origins), REFIT_EVERY):
            block = origins[blk_start:blk_start + REFIT_EVERY]
            fit_origin = block[0]

            # training rows: targets strictly before this block's first origin
            train_mask = X.index < dates[fit_origin]
            if train_mask.sum() < 100:
                continue
            model = make_model(kind)
            model.fit(X[train_mask], y[train_mask])

            # every target in this block, predicted in one call
            tgt_idx   = [o + h - 1 for o in block if o + h - 1 < n]
            tgt_dates = [dates[j] for j in tgt_idx if dates[j] in X.index]
            if not tgt_dates:
                continue
            preds = model.predict(X.loc[tgt_dates])

            for o, d, p in zip(block, tgt_dates, preds):
                rows.append({"origin": dates[o - 1], "target": d, "h": h,
                             "y_true": float(y.loc[d]), "y_pred": float(p)})

    return pd.DataFrame(rows)


def score(preds, y_full, scale_mae):
    out = []
    for h in HORIZONS:
        s = preds[preds["h"] == h]
        if s.empty:
            continue
        err = s["y_pred"] - s["y_true"]
        ss_res = (err ** 2).sum()
        ss_tot = ((s["y_true"] - s["y_true"].mean()) ** 2).sum()
        out.append({
            "h": h, "n": len(s),
            "MAE":  err.abs().mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "R2":   1 - ss_res / ss_tot,
            "MASE": err.abs().mean() / scale_mae,
            "annual_error_pct": 100 * (s["y_pred"].sum() - s["y_true"].sum()) / s["y_true"].sum(),
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    regions = (df[["pes_id", "region_name"]].drop_duplicates()
                 .set_index("pes_id")["region_name"].to_dict())

    all_rows = []
    for pes_id, name in regions.items():
        daily = load_daily(df, pes_id)
        y = daily["cf"]
        train = y.iloc[:INITIAL_TRAIN]
        scale_mae = (train - train.shift(365)).abs().mean()

        for kind in ["rf", "xgb"]:
            for weather_aware in [False, True]:
                t0 = time.time()
                preds = walk_forward(daily, kind, weather_aware)
                sc = score(preds, y, scale_mae)
                sc["pes_id"] = pes_id
                sc["region"] = name
                sc["model"]  = kind.upper()
                sc["info_set"] = "weather-aware" if weather_aware else "weather-blind"
                all_rows.append(sc)
                preds.to_csv(f"mlpred_{kind}_{'wx' if weather_aware else 'blind'}_pes{pes_id}.csv",
                             index=False)
                print(f"  {name:30s} {kind.upper():4s} "
                      f"{'weather-aware' if weather_aware else 'weather-blind'}  "
                      f"{time.time()-t0:5.1f}s")

    res = pd.concat(all_rows, ignore_index=True)
    res = res[["pes_id", "region", "model", "info_set", "h", "n",
               "MAE", "RMSE", "R2", "MASE", "annual_error_pct"]]
    res.to_csv("ml_backtest_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 96)
    print("MEAN ACROSS REGIONS — by model, information set and horizon")
    print("=" * 96)
    agg = (res.groupby(["model", "info_set", "h"])
              [["MAE", "RMSE", "R2", "MASE", "annual_error_pct"]].mean())
    print(agg.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nSaved: ml_backtest_summary.csv + per-region prediction files")
