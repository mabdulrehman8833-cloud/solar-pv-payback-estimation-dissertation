"""
================================================================================
Pooled deployment model — the one a postcode tool can actually use.
================================================================================

Two things make this DIFFERENT from the backtest models in ml_backtest.py, and
both reduce accuracy. Report it separately; do not quote backtest figures for
the deployed tool.

1. POOLED, not per-region. One model over all regions with lat/lon as features,
   so it accepts an arbitrary location instead of requiring a region-specific
   model.

2. NO LAG FEATURES. A new installation has no generation history, so
   cf_lag*/cf_roll* are unavailable at deployment. The model sees only weather,
   calendar and location.

Three evaluations, answering three different questions:

  A. Temporal holdout      train 2023-24, test 2025, same regions.
                           "How well does it predict an unseen YEAR?"
  B. Leave-one-region-out  train on 5 regions, test on the 6th.
                           "How well does it predict an unseen PLACE?"  <-- the
                           question that matters for a new postcode.
  C. Climatological run    feed the model a typical (multi-year mean) year
                           instead of actual weather, and compare its annual
                           estimate against actual.
                           "How accurate is the DEPLOYED tool?"  <-- the honest
                           number, since a 25-year payback cannot use a forecast.

C is the important one. The headline 3x advantage over PVGIS was measured with
KNOWN actual weather. A deployed tool runs on climatology, where that advantage
largely disappears. C measures what actually survives.
================================================================================
"""
import numpy as np, pandas as pd, warnings, joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

SEED = 42
WEATHER = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
           "temperature_2m", "relative_humidity_2m", "cloud_cover",
           "wind_speed_10m", "precipitation"]
FEATURES = WEATHER + ["beam_fraction", "day_sin", "day_cos",
                      "month_sin", "month_cos", "region_lat", "region_lon"]
TRAIN_YEARS, TEST_YEAR = [2023, 2024], 2025


def daily_panel(df):
    """Hourly -> one daily row per region, with weather means and location."""
    out = []
    for pes_id, g in df.groupby("pes_id"):
        g = g.sort_values("timestamp").set_index("timestamp")
        d = pd.DataFrame({
            "generation_mwh": g["generation_mw"].resample("D").sum(),
            "installed_mwp":  g["installed_mwp"].resample("D").mean(),
        })
        for c in WEATHER:
            d[c] = g[c].resample("D").mean()
        d["region_lat"] = g["region_lat"].iloc[0]
        d["region_lon"] = g["region_lon"].iloc[0]
        d["pes_id"] = pes_id
        d["region"] = g["region_name"].iloc[0]
        d["cf"] = (d["generation_mwh"] / (d["installed_mwp"] * 24)).clip(0, 1)
        # Interpolate short weather gaps rather than discarding the day. A day
        # missing one weather reading still has a valid capacity factor, and
        # dropping such days biases the annual mean upward whenever the gaps
        # fall in winter (this caused a 20% error in one region before the fix).
        d[WEATHER] = d[WEATHER].interpolate(limit_direction="both")
        out.append(d.dropna(subset=["cf"]))
    p = pd.concat(out)
    p.index = p.index.tz_localize(None)
    p = p.reset_index().rename(columns={"timestamp": "date"})

    tot = (p["direct_radiation"] + p["diffuse_radiation"]).replace(0, np.nan)
    p["beam_fraction"] = (p["direct_radiation"] / tot).fillna(0.0)
    doy = p["date"].dt.dayofyear
    p["day_sin"]   = np.sin(2*np.pi*doy/365.25);  p["day_cos"] = np.cos(2*np.pi*doy/365.25)
    p["month_sin"] = np.sin(2*np.pi*p["date"].dt.month/12)
    p["month_cos"] = np.cos(2*np.pi*p["date"].dt.month/12)
    p["year"] = p["date"].dt.year
    return p


def make(kind):
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     n_jobs=-1, random_state=SEED)
    return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=SEED, n_jobs=-1, verbosity=0)


def annual_yield(cf_series):
    """Mean daily capacity factor -> annual specific yield, kWh/kWp/yr."""
    return float(np.mean(cf_series)) * 8760


def metrics(y, yhat):
    err = yhat - y
    return dict(MAE=float(np.abs(err).mean()),
                RMSE=float(np.sqrt((err**2).mean())),
                R2=float(1 - (err**2).sum() / ((y - y.mean())**2).sum()))


if __name__ == "__main__":
    df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])
    p = daily_panel(df)
    print(f"Daily panel: {len(p):,} rows, {p['pes_id'].nunique()} regions, "
          f"{p['year'].min()}-{p['year'].max()}")

    tr = p[p["year"].isin(TRAIN_YEARS)]
    te = p[p["year"] == TEST_YEAR]

    # ---------------- A. temporal holdout ----------------------------------
    print("\n" + "="*78)
    print("A. TEMPORAL HOLDOUT — unseen year, same regions (pooled, no lags)")
    print("="*78)
    models = {}
    for kind in ["rf", "xgb"]:
        m = make(kind).fit(tr[FEATURES], tr["cf"])
        models[kind] = m
        s = metrics(te["cf"].values, m.predict(te[FEATURES]))
        print(f"  {kind.upper():4s}  MAE={s['MAE']:.4f}  RMSE={s['RMSE']:.4f}  R2={s['R2']:.4f}")

    # ---------------- B. leave-one-region-out ------------------------------
    print("\n" + "="*78)
    print("B. LEAVE-ONE-REGION-OUT — unseen PLACE (the new-postcode case)")
    print("="*78)
    loro = []
    for pes_id, name in p[["pes_id", "region"]].drop_duplicates().values:
        trn = p[p["pes_id"] != pes_id]
        tst = p[(p["pes_id"] == pes_id) & (p["year"] == TEST_YEAR)]
        m = make("rf").fit(trn[FEATURES], trn["cf"])
        pred = m.predict(tst[FEATURES])
        s = metrics(tst["cf"].values, pred)
        ann_p, ann_a = annual_yield(pred), annual_yield(tst["cf"])
        loro.append({"region": name, **s,
                     "annual_pred": ann_p, "annual_actual": ann_a,
                     "annual_err_pct": 100*(ann_p-ann_a)/ann_a})
    L = pd.DataFrame(loro)
    L.to_csv("leave_one_region_out.csv", index=False)
    print(L.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  mean |annual error| across held-out regions: "
          f"{L['annual_err_pct'].abs().mean():.2f}%")

    # ---------------- C. climatological deployment run ---------------------
    print("\n" + "="*78)
    print("C. CLIMATOLOGICAL RUN — what the DEPLOYED tool actually achieves")
    print("="*78)
    print(f"   typical year built from {TRAIN_YEARS} day-of-year means "
          f"({len(TRAIN_YEARS)} years — extend with more Open-Meteo history)")
    rows = []
    for pes_id, g in p.groupby("pes_id"):
        name = g["region"].iloc[0]
        hist = g[g["year"].isin(TRAIN_YEARS)].copy()
        hist["doy"] = hist["date"].dt.dayofyear
        clim = hist.groupby("doy")[FEATURES].mean().reset_index()   # typical year
        ann_clim = annual_yield(models["rf"].predict(clim[FEATURES]))

        act = g[g["year"] == TEST_YEAR]
        ann_act = annual_yield(act["cf"])
        ann_actual_wx = annual_yield(models["rf"].predict(act[FEATURES]))
        rows.append({"region": name,
                     "actual": ann_act,
                     "model_actual_weather": ann_actual_wx,
                     "model_climatology": ann_clim,
                     "err_actual_wx_pct": 100*(ann_actual_wx-ann_act)/ann_act,
                     "err_climatology_pct": 100*(ann_clim-ann_act)/ann_act})
    C = pd.DataFrame(rows)

    pvg = pd.read_csv("pvgis_estimates.csv").set_index("region")["pvgis_kwh_per_kwp"]
    C["pvgis"] = C["region"].map(pvg)
    C["err_pvgis_pct"] = 100*(C["pvgis"]-C["actual"])/C["actual"]
    C.to_csv("climatology_evaluation.csv", index=False)
    pd.set_option("display.width", 200)
    print(C.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print("\n  MEAN ABSOLUTE ANNUAL ERROR")
    print(f"    model, known actual weather (upper bound) : "
          f"{C['err_actual_wx_pct'].abs().mean():.2f}%")
    print(f"    model, climatological weather (DEPLOYED)  : "
          f"{C['err_climatology_pct'].abs().mean():.2f}%")
    print(f"    PVGIS                                     : "
          f"{C['err_pvgis_pct'].abs().mean():.2f}%")

    # ---------------- persist the deployable model -------------------------
    full = make("rf").fit(p[FEATURES], p["cf"])
    joblib.dump({"model": full, "features": FEATURES, "weather_cols": WEATHER},
                "pooled_model.joblib")
    print("\nSaved climatology_evaluation.csv, leave_one_region_out.csv, "
          "pooled_model.joblib")
