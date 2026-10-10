"""
Why did the LONGER climatology make the model WORSE?

Expected: a 13-year climatology (2010-2022) would be more representative than a
2-year one and shrink the model's under-prediction.
Observed: the under-prediction grew, -5.28% -> -6.56%.

The obvious candidate is that 2010-2022 was less sunny than 2023-25, so a
backward-looking climatology under-states a recent, sunnier period. This script
tests that directly:

  1. mean irradiance 2010-2022 (long-run) vs 2023-2025 (the PV_Live period)
  2. whether annual irradiance TRENDS upward within 2010-2022
  3. what each implies for which climatology period a payback tool should use

Requires: longrun_weather.csv (from extend_climatology.py)
          solar_weather_dataset.csv
"""
import numpy as np, pandas as pd
pd.set_option("display.width", 200)

W = pd.read_csv("longrun_weather.csv", parse_dates=["timestamp"])
D = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])

W["year"] = W["timestamp"].dt.year
D["year"] = D["timestamp"].dt.year

# ---- 1. long-run vs recent -------------------------------------------------
lr = W.groupby("region")["shortwave_radiation"].mean()
rc = D.groupby("region_name")["shortwave_radiation"].mean()
cmp = pd.DataFrame({"mean_2010_2022": lr, "mean_2023_2025": rc})
cmp["recent_vs_longrun_pct"] = 100 * (cmp.mean_2023_2025 / cmp.mean_2010_2022 - 1)

print("=" * 84)
print("MEAN IRRADIANCE — long-run climatology period vs the PV_Live period")
print("=" * 84)
print(cmp.to_string(float_format=lambda v: f"{v:.2f}"))
print(f"\n  mean across regions: {cmp.recent_vs_longrun_pct.mean():+.2f}%")

# ---- 2. trend within the long-run window -----------------------------------
print("\n" + "=" * 84)
print("ANNUAL MEAN IRRADIANCE TREND, 2010-2022  (OLS slope, W/m2 per year)")
print("=" * 84)
rows = []
for region, g in W.groupby("region"):
    a = g.groupby("year")["shortwave_radiation"].mean()
    slope, intercept = np.polyfit(a.index.values, a.values, 1)
    pct_per_decade = 100 * slope * 10 / a.mean()
    rows.append({"region": region, "slope_w_m2_per_yr": slope,
                 "pct_per_decade": pct_per_decade,
                 "first_year": a.iloc[0], "last_year": a.iloc[-1]})
T = pd.DataFrame(rows)
print(T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\n  mean trend: {T.pct_per_decade.mean():+.2f}% per decade")

# ---- 3. rolling-window climatologies ---------------------------------------
print("\n" + "=" * 84)
print("WHAT THE ANSWER DEPENDS ON — mean irradiance by window")
print("=" * 84)
for lo, hi in [(2010, 2022), (2013, 2022), (2018, 2022), (2020, 2022)]:
    m = W[(W.year >= lo) & (W.year <= hi)]["shortwave_radiation"].mean()
    diff = 100 * (D["shortwave_radiation"].mean() / m - 1)
    print(f"  {lo}-{hi}: {m:6.2f} W/m2   "
          f"2023-25 is {diff:+5.2f}% relative to it")

print("""
INTERPRETATION

If recent years are systematically sunnier, a backward-looking climatology
under-states future generation, and the longer the window the worse it gets --
which is what happened. Two readings, and the data above should indicate which:

  a) a real brightening trend (declining aerosols is the usual mechanism) --
     then a RECENT window is the better basis for a forward-looking payback, and
     the 13-year climatology is over-conservative;

  b) 2023-25 was simply an anomalously sunny run -- then the long-run window is
     right and the 3-year PV_Live mean is the unrepresentative quantity.

With only three years of generation data you cannot fully separate these. State
the climatology window as an explicit modelling choice, report the sensitivity of
annual yield to it, and note that PVGIS faces exactly the same problem -- its
typical-meteorological-year database is also backward-looking, which is part of
why both methods err in the direction they do.
""")
