"""
================================================================================
STEP 00 — build solar_weather_dataset.csv
================================================================================

RUN IN COLAB. Needs two sources (no API keys):
    Sheffield Solar PV_Live   regional generation + installed capacity
    Open-Meteo archive        hourly weather at each region centroid

Setup cell:
    !pip install pvlive-api pandas numpy requests pytz tqdm

Output: solar_weather_dataset.csv  -- the input to every other script.
        ~157,700 rows (6 regions x 3 years x hourly), ~30 MB.
Runtime: roughly 8-12 minutes, dominated by the PV_Live pulls.

--------------------------------------------------------------------------------
WHAT PV_LIVE ACTUALLY IS
--------------------------------------------------------------------------------
PV_Live does not measure individual panels. It estimates total generation for each
UK planning region by sampling registered PV systems and scaling up using satellite
weather and installed-capacity registers. So the target is a REGIONAL FLEET
capacity factor -- mixed orientations, shading, ground-mount, ageing panels -- not
one optimally sited rooftop. Every downstream result inherits that.

gs_id = 0 is the whole of GB; never use it for location-specific modelling.

--------------------------------------------------------------------------------
TWO NON-OBVIOUS DETAILS
--------------------------------------------------------------------------------
1. extra_fields="installedcapacity_mwp" must be requested EXPLICITLY. It is not
   returned by default, and without it capacity factor cannot be computed. (This
   is why a CSV downloaded manually from the PV_Live website is missing it.)

2. Cyclical day-of-year encoding must account for leap years. Dividing by a fixed
   365 puts every date after 29 February in a leap year slightly out of phase.
   Handled below via is_leap_year.

--------------------------------------------------------------------------------
REGION LABELS
--------------------------------------------------------------------------------
The names below are labels for PES IDs and are used for grouping in the EDA. Two
regions once shared a name by copy-paste error, which silently merged them in every
groupby("region_name") -- blending North East and South East climates. Keep the
names distinct.
================================================================================
"""
import sys, time, warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import requests

warnings.filterwarnings("ignore")

try:
    from tqdm.notebook import tqdm
except ImportError:                      # plain script, not a notebook
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw): return x

# ------------------------------------------------------------------ config ---
START = "2023-01-01"
END   = "2025-12-31"

# PES = Planning Economic Sub-region. Centroids are the postcode locations used to
# fetch matching weather. Six of the fourteen PES regions, chosen for geographic
# spread (56.4 N to 51.1 N, -3.5 E to +0.7 E).
REGIONS = {
    17: {"name": "Perth, Scotland",              "lat": 56.42, "lon": -3.46},
    15: {"name": "Cramlington, Northumberland",  "lat": 55.09, "lon": -1.59},
    23: {"name": "Castleford, West Yorkshire",   "lat": 53.72, "lon": -1.37},
    11: {"name": "Castle Donington, Derbyshire", "lat": 52.83, "lon": -1.31},
    10: {"name": "Bury St Edmunds, Suffolk",     "lat": 52.25, "lon":  0.73},
    19: {"name": "Crawley, West Sussex",         "lat": 51.12, "lon": -0.19},
}

WEATHER_VARS = [
    "temperature_2m",        # C   -- higher temperature reduces panel efficiency
    "relative_humidity_2m",  # %
    "shortwave_radiation",   # W/m2 -- global horizontal irradiance, the dominant predictor
    "direct_radiation",      # W/m2 -- beam component
    "diffuse_radiation",     # W/m2 -- scattered component
    "cloud_cover",           # %
    "wind_speed_10m",        # m/s  -- cools panels slightly
    "precipitation",         # mm
]

OUTPUT_FILE = "solar_weather_dataset.csv"
CHUNK_DAYS  = 92        # days per PV_Live request
API_PAUSE   = 0.6       # seconds between requests
UTC         = pytz.UTC


# ----------------------------------------------------------------- PV_Live ---
def _capacity_col(df):
    for col in ("installedcapacity_mwp", "capacity_mwp"):
        if col in df.columns:
            return col
    return None


def fetch_region(pvl, pes_id, name):
    """All chunks for one region, resampled hourly, with capacity factor."""
    t0 = datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=UTC)
    t1 = datetime.strptime(END,   "%Y-%m-%d").replace(tzinfo=UTC)

    chunks, cur = [], t0
    while cur < t1:
        nxt = min(cur + timedelta(days=CHUNK_DAYS), t1)
        try:
            c = pvl.between(start=cur, end=nxt, entity_type="pes", entity_id=pes_id,
                            extra_fields="installedcapacity_mwp", dataframe=True)
            if c is not None and not c.empty:
                chunks.append(c)
        except Exception as exc:
            print(f"    warning [{name}] {cur.date()}..{nxt.date()}: {exc}")
        cur = nxt
        time.sleep(API_PAUSE)

    if not chunks:
        return None

    df = pd.concat(chunks)
    df["datetime_gmt"] = pd.to_datetime(df["datetime_gmt"], utc=True)
    df = df.set_index("datetime_gmt").sort_index().resample("1h").mean(numeric_only=True)

    cap = _capacity_col(df)
    if cap is None:
        print(f"  ERROR [{name}]: installed capacity not returned.")
        print(f"  Columns: {df.columns.tolist()}")
        print("  Try:  !pip install --upgrade pvlive-api")
        return None

    df["capacity_factor"] = (df["generation_mw"] / df[cap]).clip(0.0, 1.0)
    return (df.rename(columns={cap: "installed_mwp"})
              [["generation_mw", "installed_mwp", "capacity_factor"]])


def collect_pvlive():
    from pvlive_api import PVLive
    pvl = PVLive()
    parts, failed = [], []

    for pes_id, meta in tqdm(REGIONS.items(), desc="PV_Live"):
        d = fetch_region(pvl, pes_id, meta["name"])
        if d is None:
            failed.append(meta["name"])
            continue
        d["pes_id"]      = pes_id
        d["region_name"] = meta["name"]
        d["region_lat"]  = meta["lat"]
        d["region_lon"]  = meta["lon"]
        parts.append(d.reset_index().rename(columns={"datetime_gmt": "timestamp"}))

    if not parts:
        raise RuntimeError("No PV_Live data retrieved. Check the connection and "
                           "that pvlive-api is up to date.")
    if failed:
        print(f"\nRegions with no data: {failed}")
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------- Open-Meteo ---
def fetch_weather(lat, lon):
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={"latitude": lat, "longitude": lon,
                "start_date": START, "end_date": END,
                "hourly": ",".join(WEATHER_VARS),
                "timezone": "UTC", "wind_speed_unit": "ms"},
        timeout=120)
    r.raise_for_status()
    hourly = r.json().get("hourly", {})
    if not hourly:
        raise ValueError(f"empty weather response for ({lat}, {lon})")
    w = pd.DataFrame(hourly)
    w["timestamp"] = pd.to_datetime(w.pop("time"), utc=True)
    return w.set_index("timestamp").sort_index()


def collect_weather():
    parts = []
    for pes_id, meta in tqdm(REGIONS.items(), desc="Open-Meteo"):
        try:
            w = fetch_weather(meta["lat"], meta["lon"])
            w["pes_id"] = pes_id
            parts.append(w.reset_index())
        except Exception as exc:
            print(f"  warning [{meta['name']}]: {exc}")
        time.sleep(API_PAUSE)
    if not parts:
        raise RuntimeError("No weather data retrieved.")
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------ merge and features ---
def add_features(df):
    ts = df["timestamp"]
    df["hour"]        = ts.dt.hour
    df["month"]       = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear

    # Cyclical encodings. Day-of-year uses the correct year length so that dates
    # after 29 February in a leap year are not shifted out of phase.
    days_in_year = np.where(ts.dt.is_leap_year, 366.0, 365.0)
    df["hour_sin"]  = np.sin(2*np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2*np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2*np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2*np.pi * df["month"] / 12)
    df["day_sin"]   = np.sin(2*np.pi * df["day_of_year"] / days_in_year)
    df["day_cos"]   = np.cos(2*np.pi * df["day_of_year"] / days_in_year)

    total = (df["direct_radiation"] + df["diffuse_radiation"]).replace(0, np.nan)
    df["beam_fraction"] = (df["direct_radiation"] / total).fillna(0.0)
    df["is_daylight"]   = (df["shortwave_radiation"] > 5).astype(int)
    return df


def validate(df):
    """Fail loudly on the problems that have actually bitten this pipeline."""
    problems = []

    per_region = df.groupby("region_name").size()
    if per_region.nunique() != 1:
        problems.append(f"uneven rows per region:\n{per_region.to_string()}")

    n_names = df["region_name"].nunique()
    n_ids   = df["pes_id"].nunique()
    if n_names != n_ids:
        problems.append(f"{n_ids} PES ids but only {n_names} distinct names -- "
                        "duplicate labels will silently merge regions in groupby")

    nulls = df.isna().sum()
    if nulls.any():
        problems.append(f"nulls present:\n{nulls[nulls > 0].to_string()}")

    if not df["capacity_factor"].between(0, 1).all():
        problems.append("capacity_factor outside [0, 1]")

    daylight_pct = df["is_daylight"].mean() * 100
    if not 25 < daylight_pct < 60:
        problems.append(f"daylight share {daylight_pct:.1f}% (expected 30-55%)")

    print("\n" + "=" * 72)
    if problems:
        print("VALIDATION PROBLEMS")
        print("=" * 72)
        for p in problems:
            print(f"  - {p}")
        raise SystemExit("Fix the above before using this dataset.")
    print("VALIDATION PASSED")
    print("=" * 72)
    print(f"  rows            : {len(df):,}")
    print(f"  regions         : {n_ids}")
    print(f"  period          : {df['timestamp'].min().date()} to "
          f"{df['timestamp'].max().date()}")
    print(f"  daylight share  : {daylight_pct:.1f}%")
    cf = df["capacity_factor"]
    print(f"  capacity factor : {cf.min():.4f} to {cf.max():.4f}, mean {cf.mean():.4f}")
    print("\n  mean daylight capacity factor by region:")
    for name, g in df[df["is_daylight"] == 1].groupby("region_name"):
        print(f"    {name:32s} {g['capacity_factor'].mean():.4f}")


if __name__ == "__main__":
    print(f"Building {OUTPUT_FILE}: {START} to {END}, {len(REGIONS)} regions\n")

    pv = collect_pvlive()
    print(f"\nPV_Live : {len(pv):,} rows, {pv['region_name'].nunique()} regions")

    wx = collect_weather()
    print(f"Weather : {len(wx):,} rows")

    df = pd.merge(
        pv.assign(timestamp=pd.to_datetime(pv["timestamp"], utc=True).dt.round("1h")),
        wx.assign(timestamp=pd.to_datetime(wx["timestamp"], utc=True)),
        on=["timestamp", "pes_id"], how="inner")

    df = add_features(df)
    df = (df.dropna(subset=["capacity_factor"])
            .drop_duplicates(subset=["timestamp", "pes_id"])
            .sort_values(["pes_id", "timestamp"])
            .reset_index(drop=True))

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {OUTPUT_FILE}  ({len(df):,} rows x {len(df.columns)} columns)")

    validate(df)
