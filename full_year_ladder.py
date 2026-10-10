"""
FIX 2 — recompute the method ladder on a COMPLETE YEAR.

The problem
-----------
The original ladder used h=30 predictions, which cover only 337 days
(2025-01-29 to 2025-12-31). The missing days are late January and, critically,
the darkest part of December is present while the compensating days are not --
net effect, the window OVERSTATES true annual yield by 5.8-6.2%.

Model estimates and the "actual" both came from that window, so their errors were
internally consistent. PVGIS supplies a genuine full-calendar-year figure, so
PVGIS alone was scored against an inflated baseline and its error was UNDERSTATED.

The fix
-------
h=1 predictions cover 365 consecutive days (2024-12-31 to 2025-12-30) -- a
complete year. Recompute every method on that window.

Trade-off to state in the write-up: at h=1 the models have access to yesterday's
actual generation, which flatters them relative to h=30. So the model-vs-PVGIS gap
computed here is an UPPER bound on model advantage, just as the h=30 version was a
lower bound on PVGIS's error. Report both, and be explicit about which is which:

    h=30 ladder  -> like-for-like comparison BETWEEN MODELS (all same horizon)
    h=1  ladder  -> fair-window comparison AGAINST PVGIS (complete year)

Requires: backtest_preds_pes*.csv, mlpred_*_pes*.csv, pvgis_estimates.csv
"""
import numpy as np, pandas as pd
pd.set_option("display.width", 220)

HOURS = 8760
H = 1                       # the horizon with complete 365-day coverage
PES = [10, 11, 15, 17, 19, 23]

METHODS = [("rf",  "wx",    "RF (weather-aware)"),
           ("xgb", "wx",    "XGB (weather-aware)"),
           ("rf",  "blind", "RF (weather-blind)"),
           ("xgb", "blind", "XGB (weather-blind)")]

rows = []
for pes_id in PES:
    sar = pd.read_csv(f"backtest_preds_pes{pes_id}.csv", parse_dates=["target"])
    s = sar[sar["h"] == H]
    if len(s) < 365:
        print(f"WARNING pes {pes_id}: only {len(s)} days at h={H}")
    span = f"{s['target'].min().date()} to {s['target'].max().date()}"

    rows.append({"pes_id": pes_id, "method": "Actual (PV_Live)",
                 "yield": s["y_true"].mean()*HOURS, "n_days": len(s), "span": span})
    rows.append({"pes_id": pes_id, "method": "SARIMA/Fourier",
                 "yield": s["y_pred"].mean()*HOURS, "n_days": len(s), "span": span})

    for kind, wx, label in METHODS:
        m = pd.read_csv(f"mlpred_{kind}_{wx}_pes{pes_id}.csv", parse_dates=["target"])
        m = m[m["h"] == H]
        rows.append({"pes_id": pes_id, "method": label,
                     "yield": m["y_pred"].mean()*HOURS, "n_days": len(m), "span": span})

L = pd.DataFrame(rows)

pv = pd.read_csv("pvgis_estimates.csv")
name = pv.set_index("pes_id")["region"].to_dict()
for _, r in pv.iterrows():
    L.loc[len(L)] = {"pes_id": r["pes_id"], "method": "PVGIS (vendor analogue)",
                     "yield": r["pvgis_kwh_per_kwp"], "n_days": 365,
                     "span": "full year (TMY)"}

L["region"] = L["pes_id"].map(name)
actual = L[L["method"] == "Actual (PV_Live)"].set_index("pes_id")["yield"]
L["actual"] = L["pes_id"].map(actual)
L["error_pct"] = 100*(L["yield"] - L["actual"]) / L["actual"]
L.to_csv("method_ladder_fullyear.csv", index=False)

print("="*92)
print(f"FULL-YEAR METHOD LADDER (h={H}, {L['n_days'].max()} days per region)")
print("="*92)
summ = (L[L["method"] != "Actual (PV_Live)"]
          .groupby("method")["error_pct"]
          .agg(MAE=lambda x: x.abs().mean(),
               SD="std",                      # spread across the six regions
               mean_error="mean",
               min="min", max="max")
          .sort_values("MAE"))
print(summ.to_string(float_format=lambda v: f"{v:+.2f}"))
summ.to_csv("method_ladder_fullyear_summary.csv")

print("\n" + "="*92)
print("WINDOW COMPARISON — how much the old 337-day window inflated 'actual'")
print("="*92)
old = pd.read_csv("method_ladder.csv")
oa = old[old["method"] == "Actual (PV_Live)"].set_index("region")["specific_yield_kwh_per_kwp"]
na = L[L["method"] == "Actual (PV_Live)"].set_index("region")["yield"]
cmp = pd.DataFrame({"actual_337d": oa, "actual_365d": na})
cmp["inflation_pct"] = 100*(cmp.actual_337d/cmp.actual_365d - 1)
print(cmp.to_string(float_format=lambda v: f"{v:.1f}"))

print("\nSaved method_ladder_fullyear.csv and method_ladder_fullyear_summary.csv")
