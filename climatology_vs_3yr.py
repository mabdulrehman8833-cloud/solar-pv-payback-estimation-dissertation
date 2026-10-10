"""
The deployed-tool headline number.

Script 09 compares the climatological estimate against a SINGLE year (2025).
That is the wrong basis: a climatological estimate cannot track one year's
weather anomaly by construction, and 2025 was 8-11% sunnier than the 2023-24
climatology the model was given.

A payback calculation depends on LONG-RUN AVERAGE generation, so the right
comparison is against the multi-year mean. This script does that.

Caveat to state in the write-up: the model was trained on 2023-24, which overlaps
the 3-year mean, so this is partly in-sample. Extend the weather history and hold
out a later year before the final version.

Requires: solar_weather_dataset.csv, pooled_model.joblib, climatology_evaluation.csv,
          pvgis_estimates.csv, and pooled_model.py importable alongside.
"""
import numpy as np, pandas as pd, joblib
import pooled_model as pm

pd.set_option("display.width", 200)

df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])
p = pm.daily_panel(df)
C = pd.read_csv("climatology_evaluation.csv").set_index("region")

rows = []
for pes_id, g in p.groupby("pes_id"):
    name = g["region"].iloc[0]
    rows.append({
        "region": name,
        "actual_3yr_mean":   pm.annual_yield(g["cf"]),
        "model_climatology": C.loc[name, "model_climatology"],
        "pvgis":             C.loc[name, "pvgis"],
    })

T = pd.DataFrame(rows)
T["model_err_pct"] = 100 * (T.model_climatology - T.actual_3yr_mean) / T.actual_3yr_mean
T["pvgis_err_pct"] = 100 * (T.pvgis           - T.actual_3yr_mean) / T.actual_3yr_mean
T.to_csv("climatology_vs_3yr_mean.csv", index=False)

print("="*95)
print("DEPLOYED TOOL vs VENDOR FORMULA — climatological estimate against 3-year mean actual")
print("="*95)
print(T.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
print()
print(f"  model climatology  mean abs error: {T.model_err_pct.abs().mean():5.2f}%"
      f"   (mean {T.model_err_pct.mean():+.2f}%)")
print(f"  PVGIS              mean abs error: {T.pvgis_err_pct.abs().mean():5.2f}%"
      f"   (mean {T.pvgis_err_pct.mean():+.2f}%)")
print()
print("The two errors run in OPPOSITE directions:")
print("  model under-predicts  -> short climatology, artefact a longer history should reduce")
print("  PVGIS over-predicts   -> optimal orientation vs mixed real fleet, structural")
print("\nSaved climatology_vs_3yr_mean.csv")
