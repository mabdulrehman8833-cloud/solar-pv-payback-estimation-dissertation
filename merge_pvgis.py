"""
Merge the PVGIS (vendor-analogue) estimates into annual_generation_estimates.csv
so every method sits in one table, scored against the same PV_Live ground truth.

Inputs (same directory):
    annual_generation_estimates.csv   from annual_estimates.py
    pvgis_estimates.csv               from fetch_pvgis.py
Output:
    method_ladder.csv                 all methods incl. PVGIS, per region
    method_ladder_summary.csv         mean / mean-abs / range per method

Comparability note (state this in the dissertation):
PVGIS models ONE optimally-oriented 4 kWp system (35 deg, due south). PV_Live
measures the WHOLE regional fleet -- mixed orientations, shading, ground-mount,
older panels -- normalised by installed capacity. An optimal system should beat
the fleet average, so part of any positive PVGIS error is orientation mismatch
rather than formula inaccuracy. Comparing ERRORS against a common ground truth
is still valid; interpreting the level difference as pure formula error is not.
"""
import numpy as np, pandas as pd
pd.set_option("display.width", 200)

SOUTH = [10, 19, 11, 23]      # Suffolk, W Sussex, Derbyshire, W Yorkshire
NORTH = [17, 15]              # Scotland, Northumberland

est = pd.read_csv("annual_generation_estimates.csv")
pv  = pd.read_csv("pvgis_estimates.csv")

actual = (est[est["method"] == "Actual (PV_Live)"]
            .set_index("pes_id")["specific_yield_kwh_per_kwp"])

# ---- PVGIS rows in the same schema ----------------------------------------
pv_rows = pd.DataFrame({
    "pes_id":  pv["pes_id"],
    "region":  pv["region"],
    "method":  "PVGIS (vendor analogue)",
    "mean_cf": pv["pvgis_kwh_per_kwp"] / 8760,
    "n_days":  np.nan,
    "specific_yield_kwh_per_kwp": pv["pvgis_kwh_per_kwp"],
})
pv_rows["actual_yield"] = pv_rows["pes_id"].map(actual)
pv_rows["error_pct"] = (100 * (pv_rows["specific_yield_kwh_per_kwp"] - pv_rows["actual_yield"])
                        / pv_rows["actual_yield"])

ladder = pd.concat([est, pv_rows[est.columns.intersection(pv_rows.columns)]],
                   ignore_index=True)
ladder.to_csv("method_ladder.csv", index=False)

# ---- summary ---------------------------------------------------------------
s = (ladder[ladder["method"] != "Actual (PV_Live)"]
       .groupby("method")["error_pct"]
       .agg(mean="mean", mean_abs=lambda x: x.abs().mean(), min="min", max="max")
       .sort_values("mean_abs"))
s.to_csv("method_ladder_summary.csv")

print("=" * 90)
print("ANNUAL GENERATION ERROR vs PV_Live actual — all methods, common ground truth")
print("=" * 90)
print(s.to_string(float_format=lambda v: f"{v:+.2f}"))

# ---- regional split for the vendor formula ---------------------------------
p = ladder[ladder["method"] == "PVGIS (vendor analogue)"].set_index("pes_id")
print(f"\nPVGIS mean error — 4 southern regions : {p.loc[SOUTH, 'error_pct'].mean():+.2f}%")
print(f"PVGIS mean error — 2 northern regions : {p.loc[NORTH, 'error_pct'].mean():+.2f}%")

# ---- latitude-gradient diagnostic ------------------------------------------
lat = pv.set_index("pes_id")["lat"]
g = pd.DataFrame({"lat": lat, "pvgis": pv.set_index("pes_id")["pvgis_kwh_per_kwp"],
                  "actual": actual})
rng = lambda c: 100 * (g[c].max() / g[c].min() - 1)
print("\n" + "=" * 90)
print("LATITUDE-GRADIENT CHECK — is the PV_Live target physically plausible?")
print("=" * 90)
print(f"  PVGIS  range across regions: {rng('pvgis'):5.1f}%   corr with latitude: {g['lat'].corr(g['pvgis']):+.3f}")
print(f"  Actual range across regions: {rng('actual'):5.1f}%   corr with latitude: {g['lat'].corr(g['actual']):+.3f}")
print("  A weak actual gradient suggests the PV_Live installed-capacity denominator")
print("  may be understated where deployment is sparse (Scotland / North East),")
print("  inflating the apparent capacity factor there. Report as a limitation.")

print("\nSaved method_ladder.csv and method_ladder_summary.csv")
