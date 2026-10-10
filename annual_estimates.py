"""
Collapse every model's daily backtest predictions into ONE annual generation
estimate per region per method — the quantity that feeds the payback calculator.

Output: annual_generation_estimates.csv, with a specific_yield column
(kWh per kWp per year) so it is directly comparable to a PVGIS/MCS figure
and to a vendor quote, independent of system size.

Capacity factor -> specific yield:   kWh/kWp/yr = mean_CF * 8760
"""
import numpy as np, pandas as pd

HOURS_PER_YEAR = 8760
H = 30   # the horizon at which the statistical model has decayed to climatology

df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])
regions = (df[["pes_id", "region_name"]].drop_duplicates()
             .set_index("pes_id")["region_name"].to_dict())

rows = []
for pes_id, name in regions.items():
    # ---- observed truth over the test year -------------------------------
    sar = pd.read_csv(f"backtest_preds_pes{pes_id}.csv")
    truth = sar[sar["h"] == H]
    n_days = len(truth)
    actual_cf = truth["y_true"].mean()
    rows.append({"pes_id": pes_id, "region": name, "method": "Actual (PV_Live)",
                 "mean_cf": actual_cf, "n_days": n_days})

    rows.append({"pes_id": pes_id, "region": name, "method": "SARIMA/Fourier",
                 "mean_cf": truth["y_pred"].mean(), "n_days": n_days})

    for kind, tag in [("rf", "RF"), ("xgb", "XGB")]:
        for wx, info in [("blind", "weather-blind"), ("wx", "weather-aware")]:
            p = pd.read_csv(f"mlpred_{kind}_{wx}_pes{pes_id}.csv")
            s = p[p["h"] == H]
            rows.append({"pes_id": pes_id, "region": name,
                         "method": f"{tag} ({info})",
                         "mean_cf": s["y_pred"].mean(), "n_days": len(s)})

est = pd.DataFrame(rows)
est["specific_yield_kwh_per_kwp"] = est["mean_cf"] * HOURS_PER_YEAR

# error vs actual, per region
actual = (est[est["method"] == "Actual (PV_Live)"]
            .set_index("pes_id")["specific_yield_kwh_per_kwp"])
est["actual_yield"] = est["pes_id"].map(actual)
est["error_pct"] = 100 * (est["specific_yield_kwh_per_kwp"] - est["actual_yield"]) / est["actual_yield"]
est.to_csv("annual_generation_estimates.csv", index=False)

pd.set_option("display.width", 150)
print("=" * 88)
print(f"ANNUAL GENERATION ESTIMATES — specific yield (kWh/kWp/yr), h={H} forecasts")
print("=" * 88)
piv = est.pivot_table(index="method", columns="region",
                      values="specific_yield_kwh_per_kwp")
order = ["Actual (PV_Live)", "SARIMA/Fourier",
         "RF (weather-aware)", "XGB (weather-aware)",
         "RF (weather-blind)", "XGB (weather-blind)"]
print(piv.reindex(order).to_string(float_format=lambda v: f"{v:.0f}"))

print("\n" + "=" * 60)
print("MEAN ERROR vs ACTUAL (%)")
print("=" * 60)
print(est[est["method"] != "Actual (PV_Live)"]
        .groupby("method")["error_pct"].agg(["mean", "min", "max"])
        .reindex(order[1:]).to_string(float_format=lambda v: f"{v:+.2f}"))
print("\nSaved annual_generation_estimates.csv")
print("\nAdd PVGIS as another row per region once pvgis_estimates.csv exists:")
print("  its pvgis_kwh_per_kwp column is directly comparable to specific_yield_kwh_per_kwp.")
