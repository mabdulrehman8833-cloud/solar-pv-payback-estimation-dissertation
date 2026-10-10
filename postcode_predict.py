"""
================================================================================
Postcode -> annual generation -> payback

RUN IN COLAB. Needs two public APIs (no keys):
    postcodes.io           postcode  -> latitude / longitude
    Open-Meteo archive     lat/lon   -> historical hourly weather

Requires pooled_model.joblib from pooled_model.py in the same directory.

--------------------------------------------------------------------------------
HOW IT WORKS
--------------------------------------------------------------------------------
1. Geocode the postcode.
2. Pull weather for CLIM_START..CLIM_END (set in economics.py) and average it by
   day-of-year into a "typical year". A 25-year payback cannot use a forecast,
   so the model is fed climatology -- the same basis PVGIS and MCS use, but with
   a weather-response function learned from measured UK generation rather than a
   physical model plus a fixed derate.
3. Predict daily capacity factor, convert to annual specific yield (kWh/kWp/yr).
4. Scale by system size and run the cash flows.

--------------------------------------------------------------------------------
ACCURACY -- QUOTE THESE, NOT THE BACKTEST FIGURES
--------------------------------------------------------------------------------
    Known actual weather, per-region models (upper bound)    1.4%   NOT achievable here
    Unseen region, actual weather (leave-one-region-out)      3.3%
    Climatology 2013-2022 vs 3-year mean actual              5.8%   <- THIS TOOL
    PVGIS on the same basis                                 16.6%

Quote the 5.8% figure, and state the window it came from. It is tied to
CLIM_START/CLIM_END in economics.py -- change the window and it must be
re-derived with extend_climatology.py.

The tool predicts REGIONAL FLEET capacity factor -- mixed orientations, shading,
ground-mount, ageing panels. A specific well-sited roof will typically beat it;
a shaded or east-west roof will not reach it. Present a range, not a point.

--------------------------------------------------------------------------------
FINANCIAL ASSUMPTIONS
--------------------------------------------------------------------------------
All economics live in economics.py. Fill in every 'source' comment there before
submission and reproduce the assumption table in the method chapter -- every
payback number depends on them.
================================================================================
"""
import json, time, urllib.parse, urllib.request
import numpy as np, pandas as pd, joblib

# ------------------------------------------------------------------ config ---
SYSTEM_KWP   = 4.0

# ---- assumptions live in economics.py so both scripts stay in sync ----------
from economics import (IMPORT_PRICE, SEG_RATE, SELF_CONSUMPTION,
                       PRICE_ESCALATION, DEGRADATION, INVERTER_REPLACE,
                       DISCOUNT_RATE, HORIZON_YEARS, capital_cost,
                       assumptions_table, CLIM_START, CLIM_END,
                       DEPLOYED_ACCURACY_PCT)

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "shortwave_radiation",
                "direct_radiation", "diffuse_radiation", "cloud_cover",
                "wind_speed_10m", "precipitation"]


# ------------------------------------------------------------------- data ---
def geocode(postcode):
    url = "https://api.postcodes.io/postcodes/" + urllib.parse.quote(postcode.strip())
    r = json.loads(urllib.request.urlopen(url, timeout=30).read().decode())
    if r.get("status") != 200:
        raise ValueError(f"postcode lookup failed for {postcode!r}: {r}")
    d = r["result"]
    return d["latitude"], d["longitude"], d.get("admin_district"), d.get("region")


def fetch_weather(lat, lon, start=CLIM_START, end=CLIM_END):
    """Hourly archive weather over the SAME fixed window the tool was evaluated on.

    A rolling 'last N years' window would drift with the run date and would not
    match any measured accuracy figure. Fixed dates keep the two in step.
    """
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}"
           f"&start_date={start}&end_date={end}"
           f"&hourly={','.join(WEATHER_VARS)}&timezone=UTC&wind_speed_unit=ms")
    h = json.loads(urllib.request.urlopen(url, timeout=180).read().decode())["hourly"]
    w = pd.DataFrame(h)
    w["timestamp"] = pd.to_datetime(w.pop("time"))
    return w.set_index("timestamp").sort_index()


def typical_year(wx, lat, lon, features):
    """Daily means -> average across years by day-of-year -> model features."""
    d = wx.resample("D").mean()
    tot = (d["direct_radiation"] + d["diffuse_radiation"]).replace(0, np.nan)
    d["beam_fraction"] = (d["direct_radiation"] / tot).fillna(0.0)
    d["doy"] = d.index.dayofyear
    clim = d.groupby("doy").mean(numeric_only=True).reset_index()
    clim = clim[clim["doy"] <= 365]

    clim["day_sin"]   = np.sin(2*np.pi*clim["doy"]/365.25)
    clim["day_cos"]   = np.cos(2*np.pi*clim["doy"]/365.25)
    month = pd.to_datetime(clim["doy"], format="%j").dt.month
    clim["month_sin"] = np.sin(2*np.pi*month/12)
    clim["month_cos"] = np.cos(2*np.pi*month/12)
    clim["region_lat"], clim["region_lon"] = lat, lon
    return clim[features]


# -------------------------------------------------------------- financial ---
def cash_flows(annual_kwh_y1):
    capital = capital_cost(SYSTEM_KWP)
    rows = []
    cum = -capital
    for t in range(1, HORIZON_YEARS + 1):
        gen   = annual_kwh_y1 * (1 - DEGRADATION) ** (t - 1)
        price = IMPORT_PRICE * (1 + PRICE_ESCALATION) ** (t - 1)
        saving = gen * SELF_CONSUMPTION * price + gen * (1 - SELF_CONSUMPTION) * SEG_RATE
        cost   = INVERTER_REPLACE[1] if t == INVERTER_REPLACE[0] else 0.0
        net    = saving - cost
        cum   += net
        rows.append({"year": t, "generation_kwh": gen, "net_benefit": net,
                     "cumulative": cum,
                     "discounted": net / (1 + DISCOUNT_RATE) ** t})
    cf = pd.DataFrame(rows)
    cf["cum_discounted"] = cf["discounted"].cumsum() - capital

    def crossing(col):
        hit = cf[cf[col] >= 0]
        if hit.empty:
            return None
        i = hit.index[0]
        if i == 0:
            return 1.0
        prev, cur = cf[col].iloc[i-1], cf[col].iloc[i]
        return float(cf["year"].iloc[i-1] + (0 - prev) / (cur - prev))

    return cf, capital, crossing("cumulative"), crossing("cum_discounted")


def report(postcode, system_kwp=SYSTEM_KWP):
    bundle = joblib.load("pooled_model.joblib")
    model, features = bundle["model"], bundle["features"]

    lat, lon, district, region = geocode(postcode)
    print(f"{postcode.upper()}  ->  {lat:.4f}, {lon:.4f}   {district}, {region}")

    wx = fetch_weather(lat, lon)
    print(f"weather: {wx.index.min().date()} to {wx.index.max().date()} "
          f"({len(wx):,} hourly rows)")

    X = typical_year(wx, lat, lon, features)
    cf_daily = model.predict(X)
    specific_yield = float(np.mean(cf_daily)) * 8760
    annual_kwh = specific_yield * system_kwp

    print(f"\nspecific yield : {specific_yield:,.0f} kWh/kWp/yr")
    print(f"system         : {system_kwp:.1f} kWp")
    print(f"annual output  : {annual_kwh:,.0f} kWh/yr")
    if DEPLOYED_ACCURACY_PCT is None:
        print("  accuracy band: UNKNOWN — run extend_climatology.py on this window"
              "\n                 and set DEPLOYED_ACCURACY_PCT in economics.py")
    else:
        b = DEPLOYED_ACCURACY_PCT / 100
        print(f"  fleet-model range (+/-{DEPLOYED_ACCURACY_PCT:.1f}%, "
              f"window {CLIM_START[:4]}-{CLIM_END[:4]}): "
              f"{annual_kwh*(1-b):,.0f} to {annual_kwh*(1+b):,.0f} kWh/yr")

    cf, capital, simple, discounted = cash_flows(annual_kwh)
    print(f"\ncapital        : GBP {capital:,.0f}")
    print(f"year-1 benefit : GBP {cf['net_benefit'].iloc[0]:,.0f}")
    print(f"simple payback : {simple:.1f} years" if simple else
          "simple payback : not reached within horizon")
    print(f"discounted payback ({DISCOUNT_RATE:.0%}): "
          + (f"{discounted:.1f} years" if discounted else "not reached"))
    print(f"NPV over {HORIZON_YEARS} yr: GBP {cf['cum_discounted'].iloc[-1]:,.0f}")
    return dict(postcode=postcode, lat=lat, lon=lon,
                specific_yield=specific_yield, annual_kwh=annual_kwh,
                capital=capital, simple_payback=simple,
                discounted_payback=discounted,
                npv=float(cf["cum_discounted"].iloc[-1])), cf


def sensitivity(annual_kwh,
                prices=(0.20, 0.25, 0.30, 0.35),
                shares=(0.25, 0.35, 0.45, 0.55)):
    """Payback is dominated by these two assumptions -- report the grid, not a point."""
    global IMPORT_PRICE, SELF_CONSUMPTION
    p0, s0 = IMPORT_PRICE, SELF_CONSUMPTION
    out = pd.DataFrame(index=[f"{p:.2f}" for p in prices],
                       columns=[f"{s:.0%}" for s in shares], dtype=float)
    for p in prices:
        for s in shares:
            IMPORT_PRICE, SELF_CONSUMPTION = p, s
            _, _, simple, _ = cash_flows(annual_kwh)
            out.loc[f"{p:.2f}", f"{s:.0%}"] = simple if simple else np.nan
    IMPORT_PRICE, SELF_CONSUMPTION = p0, s0
    out.index.name = "import GBP/kWh"; out.columns.name = "self-consumption"
    return out


if __name__ == "__main__":
    res, cf = report("RH10 1HA")          # Three Bridges, Crawley
    print("\nSIMPLE PAYBACK (years) — sensitivity")
    print(sensitivity(res["annual_kwh"]).to_string(float_format=lambda v: f"{v:.1f}"))
