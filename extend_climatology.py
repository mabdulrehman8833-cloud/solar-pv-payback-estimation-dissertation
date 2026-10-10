"""
FIX 1 — build the climatology from a long weather history instead of two years.

RUN IN COLAB (needs Open-Meteo).

The problem
-----------
climatology_vs_3yr.py builds the "typical year" from 2023-24 only. Two years is
too short to be a climatology, and those are the same years the model trained on,
so the comparison is partly in-sample. The model's -5.3% under-prediction is
probably mostly an artefact of that.

The fix
-------
Pull CLIM_START..CLIM_END of weather (set in economics.py) for each region centroid.
Those years do NOT overlap the PV_Live actuals (2023-25), so the climatology is
fully out-of-sample: a genuine "typical year" for the location, exactly what a
vendor formula uses and what a 25-year payback needs.

Residual caveat to state: the model itself learned its weather->generation mapping
from 2023-24, which is unavoidable while PV_Live coverage is only three years. This
removes the climatology overlap, not the training overlap.

RATE LIMITS
-----------
Open-Meteo weights each call by (variables x days), so a 13-year 8-variable pull is
expensive and several in a row return HTTP 429. This version:

  * caches each region to wx_cache/<pes_id>.csv as soon as it downloads
  * SKIPS regions already cached, so a 429 costs you only the unfinished ones
  * retries with exponential backoff, honouring Retry-After when the server sends it
  * waits PAUSE_BETWEEN seconds between regions

If you still hit 429s: raise PAUSE_BETWEEN, or shorten the window (2013-2022 is
still a perfectly defensible 10-year climatology), or just re-run the script — it
resumes from the cache each time.

Requires: pooled_model.joblib, solar_weather_dataset.csv, pvgis_estimates.csv
Writes:   wx_cache/*.csv, longrun_weather.csv, climatology_longrun.csv
"""
import json, os, time, urllib.error, urllib.request
import numpy as np, pandas as pd, joblib
import pooled_model as pm

from economics import CLIM_START, CLIM_END   # single source of truth
CACHE_DIR      = "wx_cache"
PAUSE_BETWEEN  = 30      # seconds between regions
MAX_RETRIES    = 6
BACKOFF_START  = 60      # seconds, doubles each retry

REGIONS = {                       # pes_id: (name, lat, lon)
    17: ("Perth, Scotland",              56.42, -3.46),
    15: ("Cramlington, Northumberland",  55.09, -1.59),
    23: ("Castleford, West Yorkshire",   53.72, -1.37),
    11: ("Castle Donington, Derbyshire", 52.83, -1.31),
    10: ("Bury St Edmunds, Suffolk",     52.25,  0.73),
    19: ("Crawley, West Sussex",         51.12, -0.19),
}

WEATHER = ["temperature_2m", "relative_humidity_2m", "shortwave_radiation",
           "direct_radiation", "diffuse_radiation", "cloud_cover",
           "wind_speed_10m", "precipitation"]


def fetch_with_retry(lat, lon):
    """One region, with exponential backoff on 429/5xx."""
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}"
           f"&start_date={CLIM_START}&end_date={CLIM_END}"
           f"&hourly={','.join(WEATHER)}&timezone=UTC&wind_speed_unit=ms")

    wait = BACKOFF_START
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = urllib.request.urlopen(url, timeout=300).read().decode()
            h = json.loads(raw)["hourly"]
            d = pd.DataFrame(h)
            d["timestamp"] = pd.to_datetime(d.pop("time"))
            return d.set_index("timestamp").sort_index()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == MAX_RETRIES:
                raise
            retry_after = e.headers.get("Retry-After")
            sleep_for = int(retry_after) if (retry_after or "").isdigit() else wait
            print(f"    HTTP {e.code} — attempt {attempt}/{MAX_RETRIES}, "
                  f"waiting {sleep_for}s")
            time.sleep(sleep_for)
            wait *= 2
    raise RuntimeError("unreachable")


def get_region(pes_id, name, lat, lon):
    """Cached download. Returns the daily-mean frame for one region.

    The cache key includes the date range, so changing CLIM_START/CLIM_END
    forces a fresh download instead of silently serving the old window.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"{CLIM_START[:4]}_{CLIM_END[:4]}"
    path = os.path.join(CACHE_DIR, f"{pes_id}_{tag}.csv")

    if os.path.exists(path):
        d = pd.read_csv(path, parse_dates=["timestamp"])
        # belt and braces: verify the cached span really matches what we asked for
        want_lo, want_hi = pd.Timestamp(CLIM_START), pd.Timestamp(CLIM_END)
        got_lo, got_hi = d["timestamp"].min(), d["timestamp"].max()
        if abs((got_lo - want_lo).days) > 3 or abs((got_hi - want_hi).days) > 3:
            print(f"{name:32s} cache span {got_lo.date()}..{got_hi.date()} "
                  f"does not match {want_lo.date()}..{want_hi.date()} — refetching")
        else:
            print(f"{name:32s} cached ({len(d):,} daily rows, "
                  f"{got_lo.date()} to {got_hi.date()})")
            return d

    wx = fetch_with_retry(lat, lon)
    d = wx.resample("D").mean()
    d.index.name = "timestamp"          # so the cached CSV always has this column
    d["pes_id"], d["region"] = pes_id, name
    d["region_lat"], d["region_lon"] = lat, lon
    d = d.reset_index()
    d.to_csv(path, index=False)
    print(f"{name:32s} downloaded ({len(wx):,} hourly -> {len(d):,} daily) "
          f"-> {path}")
    return d


if __name__ == "__main__":
    bundle = joblib.load("pooled_model.joblib")
    model, FEATURES = bundle["model"], bundle["features"]

    frames = []
    for i, (pes_id, (name, lat, lon)) in enumerate(REGIONS.items()):
        was_cached = os.path.exists(os.path.join(CACHE_DIR, f"{pes_id}.csv"))
        frames.append(get_region(pes_id, name, lat, lon))
        if not was_cached and i < len(REGIONS) - 1:
            time.sleep(PAUSE_BETWEEN)

    W = pd.concat(frames, ignore_index=True)
    W.to_csv("longrun_weather.csv", index=False)

    # ---- typical year per region, then predict ----------------------------
    tot = (W["direct_radiation"] + W["diffuse_radiation"]).replace(0, np.nan)
    W["beam_fraction"] = (W["direct_radiation"] / tot).fillna(0.0)
    W["doy"] = W["timestamp"].dt.dayofyear

    out = []
    for pes_id, g in W.groupby("pes_id"):
        clim = g[g["doy"] <= 365].groupby("doy").mean(numeric_only=True).reset_index()
        clim["day_sin"]   = np.sin(2*np.pi*clim["doy"]/365.25)
        clim["day_cos"]   = np.cos(2*np.pi*clim["doy"]/365.25)
        month = pd.to_datetime(clim["doy"], format="%j").dt.month
        clim["month_sin"] = np.sin(2*np.pi*month/12)
        clim["month_cos"] = np.cos(2*np.pi*month/12)
        out.append({"pes_id": pes_id,
                    "region": REGIONS[pes_id][0],
                    "model_climatology_longrun":
                        pm.annual_yield(model.predict(clim[FEATURES]))})

    C = pd.DataFrame(out)

    # ---- compare against the 3-year mean actual ---------------------------
    df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])
    p = pm.daily_panel(df)
    act = p.groupby("pes_id")["cf"].mean() * 8760
    C["actual_3yr_mean"] = C["pes_id"].map(act)

    pv = pd.read_csv("pvgis_estimates.csv").set_index("pes_id")["pvgis_kwh_per_kwp"]
    C["pvgis"] = C["pes_id"].map(pv)
    C["model_err_pct"] = 100*(C.model_climatology_longrun - C.actual_3yr_mean)/C.actual_3yr_mean
    C["pvgis_err_pct"] = 100*(C.pvgis - C.actual_3yr_mean)/C.actual_3yr_mean
    C.to_csv("climatology_longrun.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "="*92)
    print(f"LONG-RUN CLIMATOLOGY ({CLIM_START[:4]}-{CLIM_END[:4]}) vs 3-year mean actual")
    print("="*92)
    print(C.drop(columns="pes_id").to_string(index=False,
                                             float_format=lambda v: f"{v:.1f}"))
    print(f"\n  model  mean abs error: {C.model_err_pct.abs().mean():5.2f}%"
          f"   (mean {C.model_err_pct.mean():+.2f}%)")
    print(f"  PVGIS  mean abs error: {C.pvgis_err_pct.abs().mean():5.2f}%"
          f"   (mean {C.pvgis_err_pct.mean():+.2f}%)")
    print("\nWindow sensitivity measured so far (mean abs error vs 3-yr actual):")
    print("    2023-2024 (2 yr)   5.28%   overlaps training, not usable")
    print("    2013-2022 (10 yr)  5.77%   <-- chosen")
    print("    2010-2022 (13 yr)  6.56%")
    print("    1995-2022 (28 yr)  7.09%")
    print("Longer windows read lower -- consistent with rising irradiance.")
    print("If you change the window, update DEPLOYED_ACCURACY_PCT in economics.py.")
    print("\nSaved longrun_weather.csv, climatology_longrun.csv")
