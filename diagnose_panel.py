"""
Why is the pooled panel short of 6 x 1096 = 6576 rows?

daily_panel() calls .dropna() per region, which silently discards any day with a
missing value. If the discarded days are winter days, the surviving mean capacity
factor is biased UPWARD -- which is what has happened to Castleford.

Run this against your solar_weather_dataset.csv and paste the output.
"""
import numpy as np, pandas as pd
pd.set_option("display.width", 200)

WEATHER = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
           "temperature_2m", "relative_humidity_2m", "cloud_cover",
           "wind_speed_10m", "precipitation"]

df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])
print(f"raw hourly rows: {len(df):,}")
print("\nhourly rows per region (expect 26,281 each for 3 full years):")
print(df.groupby("region_name").size().to_string())

print("\nNaNs in the raw hourly file:")
n = df.isna().sum()
print(n[n > 0].to_string() if (n > 0).any() else "  none")

print("\n" + "="*78)
print("DAILY AGGREGATION — where rows are lost")
print("="*78)
for pes_id, g in df.groupby("pes_id"):
    name = g["region_name"].iloc[0]
    g = g.sort_values("timestamp").set_index("timestamp")
    d = pd.DataFrame({
        "generation_mwh": g["generation_mw"].resample("D").sum(),
        "installed_mwp":  g["installed_mwp"].resample("D").mean(),
    })
    for c in WEATHER:
        d[c] = g[c].resample("D").mean()
    d["cf"] = (d["generation_mwh"] / (d["installed_mwp"] * 24)).clip(0, 1)

    before = len(d)
    bad = d[d.isna().any(axis=1)]
    after = before - len(bad)

    print(f"\n{name}  ({before} days -> {after} kept, {len(bad)} dropped)")
    if len(bad):
        cols = bad.columns[bad.isna().any()].tolist()
        print(f"   columns causing drops : {cols}")
        print(f"   dropped date range    : {bad.index.min().date()} to {bad.index.max().date()}")
        print(f"   dropped by month      : "
              f"{bad.groupby(bad.index.month).size().to_dict()}")
        kept = d.dropna()
        print(f"   mean CF  all days     : {d['cf'].mean():.5f}"
              f"  -> annual {d['cf'].mean()*8760:.1f} kWh/kWp")
        print(f"   mean CF  kept only    : {kept['cf'].mean():.5f}"
              f"  -> annual {kept['cf'].mean()*8760:.1f} kWh/kWp")
        print(f"   BIAS FROM DROPPING    : "
              f"{100*(kept['cf'].mean()/d['cf'].mean()-1):+.1f}%")

print("\n" + "="*78)
print("FIX")
print("="*78)
print("""Do not drop days. In daily_panel(), replace the per-region

    out.append(d.dropna())

with an interpolation of the weather columns and a drop only where the TARGET
is missing:

    d[WEATHER] = d[WEATHER].interpolate(limit_direction='both')
    out.append(d.dropna(subset=['cf']))

A day missing one weather reading still has a valid capacity factor; discarding
it removes real generation data and biases the annual mean.""")
