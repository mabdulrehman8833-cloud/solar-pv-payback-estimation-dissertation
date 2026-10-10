"""Merge the PVGIS (vendor-analogue) estimates into the method comparison, and
check whether they are actually comparable in levels to fleet-wide PV_Live."""
import numpy as np, pandas as pd
pd.set_option("display.width", 200)

HOURS = 8760
pv = pd.read_csv("pvgis_estimates.csv")
est = pd.read_csv("annual_generation_estimates.csv")

actual = (est[est["method"] == "Actual (PV_Live)"]
            .set_index("pes_id")[["region", "specific_yield_kwh_per_kwp"]]
            .rename(columns={"specific_yield_kwh_per_kwp": "actual"}))
p = pv.set_index("pes_id")[["lat", "pvgis_kwh_per_kwp"]].join(actual)
p["pvgis_error_pct"] = 100 * (p["pvgis_kwh_per_kwp"] - p["actual"]) / p["actual"]
p = p.sort_values("lat", ascending=False)

print("=" * 92)
print("PVGIS (4 kWp, 35 deg, due south) vs PV_Live fleet actual — sorted north to south")
print("=" * 92)
print(p[["lat", "region", "pvgis_kwh_per_kwp", "actual", "pvgis_error_pct"]]
        .to_string(float_format=lambda v: f"{v:.1f}"))

print(f"\nPVGIS  north-south spread: {p['pvgis_kwh_per_kwp'].min():.0f} -> "
      f"{p['pvgis_kwh_per_kwp'].max():.0f} kWh/kWp "
      f"({100*(p['pvgis_kwh_per_kwp'].max()/p['pvgis_kwh_per_kwp'].min()-1):.1f}% range)")
print(f"Actual north-south spread: {p['actual'].min():.0f} -> {p['actual'].max():.0f} kWh/kWp "
      f"({100*(p['actual'].max()/p['actual'].min()-1):.1f}% range)")
print(f"\nCorrelation of PVGIS vs actual across regions: "
      f"r = {p['pvgis_kwh_per_kwp'].corr(p['actual']):.3f}")
print(f"Correlation of latitude vs PVGIS : r = {p['lat'].corr(p['pvgis_kwh_per_kwp']):.3f}")
print(f"Correlation of latitude vs actual: r = {p['lat'].corr(p['actual']):.3f}")

# ---- monthly shape: PVGIS vs observed -------------------------------------
df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])
mrows = []
for pes_id in p.index:
    reg = df[df["pes_id"] == pes_id].copy()
    reg["month"] = reg["timestamp"].dt.month
    # observed monthly specific yield, averaged over the 3 years
    g = reg.groupby([reg["timestamp"].dt.year, "month"]).apply(
        lambda d: (d["generation_mw"].sum() / d["installed_mwp"].mean()),
        include_groups=False)
    obs = g.groupby(level=1).mean()
    pvg = pv.set_index("pes_id").loc[pes_id, [f"m{m:02d}" for m in range(1, 13)]].astype(float)
    pvg.index = range(1, 13)
    mrows.append({"pes_id": pes_id, "region": p.loc[pes_id, "region"],
                  "shape_corr": np.corrcoef(obs.values, pvg.values / 4)[0, 1],
                  "obs_summer_share": obs[[5,6,7,8]].sum() / obs.sum(),
                  "pvgis_summer_share": pvg[[5,6,7,8]].sum() / pvg.sum()})

m = pd.DataFrame(mrows)
print("\n" + "=" * 92)
print("MONTHLY SHAPE — does PVGIS get the seasonal profile right, ignoring level?")
print("=" * 92)
print(m.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
