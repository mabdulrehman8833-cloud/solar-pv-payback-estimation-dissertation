"""Why does the deployed tool still under-predict by 5.77%?

The longer window was meant to remove a short-window artefact. It did not --
the under-prediction grew slightly, 5.28% -> 5.77%. This decomposes the gap by
feeding the SAME saved model three different weather inputs and holding
everything else fixed:

  A  actual 2023-25 daily weather        -- the model's own ceiling
  B  day-of-year mean of 2023-25         -- same period, smoothed: isolates
                                            the effect of averaging weather
  C  day-of-year mean of 2013-22         -- the deployed climatology

  B - A  = the smoothing effect (Jensen: a mean day is not an average day)
  C - B  = the window effect   (2013-22 was dimmer than 2023-25)

Caveat to state in the write-up: pooled_model.joblib was fitted on the full
panel, so A is in-sample. That is fine here -- this decomposes the effect of
the INPUT, not the accuracy of the model.
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# pooled_model.py writes this next to the scripts. Override with
#     POOLED_MODEL=/path/to/pooled_model.joblib python decompose_climatology.py
BUNDLE = os.environ.get("POOLED_MODEL", "pooled_model.joblib")
b = joblib.load(BUNDLE)
model, FEATURES = b["model"], b["features"]
WEATHER = b["weather_cols"]


def featurise(d):
    tot = (d["direct_radiation"] + d["diffuse_radiation"]).replace(0, np.nan)
    d["beam_fraction"] = (d["direct_radiation"] / tot).fillna(0.0)
    doy = d["date"].dt.dayofyear
    d["day_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["day_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["month_sin"] = np.sin(2 * np.pi * d["date"].dt.month / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["date"].dt.month / 12)
    d["doy"] = doy
    return d


# ---- recent period: daily weather and measured capacity factor -------------
raw = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])
recent = []
for pid, g in raw.groupby("pes_id"):
    g = g.sort_values("timestamp").set_index("timestamp")
    d = pd.DataFrame({"gen": g.generation_mw.resample("D").sum(),
                      "cap": g.installed_mwp.resample("D").mean()})
    for c in WEATHER:
        d[c] = g[c].resample("D").mean()
    d[WEATHER] = d[WEATHER].interpolate(limit_direction="both")
    d["cf"] = (d.gen / (d.cap * 24)).clip(0, 1)
    d["region_lat"] = g.region_lat.median()
    d["region_lon"] = g.region_lon.median()
    d["region"] = g.region_name.iloc[0]
    d["pes_id"] = pid
    d = d[d.cap > 0].dropna(subset=["cf"]).reset_index()
    d = d.rename(columns={"timestamp": "date"})
    d["date"] = d["date"].dt.tz_localize(None)
    recent.append(featurise(d))
R = pd.concat(recent, ignore_index=True)

# ---- long-run weather ------------------------------------------------------
L = pd.read_csv("longrun_weather.csv", parse_dates=["timestamp"])
L = L.rename(columns={"timestamp": "date", "region": "region_name"})
L = featurise(L)

ann = lambda cf: float(np.mean(cf)) * 8760
rows = []
for pid, g in R.groupby("pes_id"):
    name = g.region.iloc[0]
    lat, lon = g.region_lat.iloc[0], g.region_lon.iloc[0]

    A = ann(model.predict(g[FEATURES]))                       # actual weather

    clim_recent = g.groupby("doy")[FEATURES].mean().reset_index()
    B = ann(model.predict(clim_recent[FEATURES]))             # smoothed, same years

    lg = L[L.pes_id == pid].copy()
    clim_long = lg.groupby("doy")[FEATURES].mean().reset_index()
    clim_long["region_lat"], clim_long["region_lon"] = lat, lon
    Cc = ann(model.predict(clim_long[FEATURES]))              # deployed climatology

    M = ann(g.cf)                                             # measured
    rows.append({"region": name.split(",")[0], "measured": M,
                 "A_actual_wx": A, "B_smoothed_2325": B, "C_clim_1322": Cc,
                 "err_A": 100 * (A - M) / M,
                 "err_B": 100 * (B - M) / M,
                 "err_C": 100 * (Cc - M) / M,
                 "smoothing_pp": 100 * (B - A) / M,
                 "window_pp": 100 * (Cc - B) / M})

D = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(D.round(2).to_string(index=False))
print()
print("MEAN DECOMPOSITION OF THE DEPLOYED-TOOL ERROR")
print(f"  model on actual weather              {D.err_A.mean():+6.2f}%")
print(f"  + smoothing to a day-of-year mean    {D.smoothing_pp.mean():+6.2f} pp")
print(f"  + shifting window back to 2013-22    {D.window_pp.mean():+6.2f} pp")
print(f"  = deployed tool on climatology       {D.err_C.mean():+6.2f}%")
print()
print(f"  mean |error| deployed: {D.err_C.abs().mean():.2f}%")
D.to_csv("climatology_decomposition.csv", index=False)
print("\nSaved climatology_decomposition.csv")


# ---------------------------------------------------------------------------
# Does averaging the OUTPUT instead of the INPUT remove the smoothing bias?
# Run the model over each historical year separately, then average the annual
# yields. Same weather record, same model -- only the order of the averaging
# changes.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("AVERAGE-THE-OUTPUT vs AVERAGE-THE-INPUT (same 2013-22 record)")
print("=" * 78)
L["year"] = L["date"].dt.year
fix = []
for pid, g in R.groupby("pes_id"):
    name = g.region.iloc[0].split(",")[0]
    lat, lon = g.region_lat.iloc[0], g.region_lon.iloc[0]
    M = ann(g.cf)
    lg = L[L.pes_id == pid].copy()
    lg["region_lat"], lg["region_lon"] = lat, lon

    per_year = [ann(model.predict(y[FEATURES])) for _, y in lg.groupby("year")]
    out_avg = float(np.mean(per_year))

    clim = lg.groupby("doy")[FEATURES].mean().reset_index()
    clim["region_lat"], clim["region_lon"] = lat, lon
    in_avg = ann(model.predict(clim[FEATURES]))

    fix.append({"region": name, "measured": M,
                "average_the_input": in_avg, "average_the_output": out_avg,
                "err_input_pct": 100 * (in_avg - M) / M,
                "err_output_pct": 100 * (out_avg - M) / M,
                "recovered_pp": 100 * (out_avg - in_avg) / M})
F = pd.DataFrame(fix)
print(F.round(2).to_string(index=False))
print()
print(f"  mean |error|, average the INPUT  (typical year) : "
      f"{F.err_input_pct.abs().mean():.2f}%")
print(f"  mean |error|, average the OUTPUT (per-year runs): "
      f"{F.err_output_pct.abs().mean():.2f}%")
print(f"  mean recovered by re-ordering the averaging     : "
      f"{F.recovered_pp.mean():+.2f} pp")
F.to_csv("averaging_order.csv", index=False)
print("\nSaved averaging_order.csv")
