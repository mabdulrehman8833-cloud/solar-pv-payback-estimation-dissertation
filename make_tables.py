"""Generate every table the dissertation needs, as CSV, from the source data.

Nothing here is retyped by hand. Re-run after any change to the pipeline:
    python make_tables.py
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL

import economics as E
from make_figures import PANEL, REGIONS

OUT = "tables"
import os
os.makedirs(OUT, exist_ok=True)

ORDER = [19, 10, 11, 23, 15, 17]          # south to north
SHORT = {p: REGIONS[p].split(",")[0] for p in REGIONS}


def mann_kendall(x):
    """Original Mann-Kendall test. Returns (trend, p)."""
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    uniq, cnt = np.unique(x, return_counts=True)
    tie = sum(c * (c - 1) * (2 * c + 5) for c in cnt if c > 1)
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var)
    elif s < 0:
        z = (s + 1) / np.sqrt(var)
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    trend = "increasing" if (p < 0.05 and z > 0) else \
            "decreasing" if (p < 0.05 and z < 0) else "no trend"
    return trend, p


def strengths(s):
    """Wang-Smith-Hyndman trend and seasonal strength from an STL fit."""
    # robust=False, deliberately. The earlier run used robust=True, which
    # down-weights "outliers" -- but in a daily solar series the extreme days
    # ARE the weather signal, not contamination. With only three annual cycles
    # that down-weighting collapses the residual into the middle cycle
    # (residual sd 0.002 / 0.049 / 0.003 by year for Crawley) and leaves an
    # edge-degenerate fit. Non-robust STL spreads it evenly (0.015 / 0.031 /
    # 0.015) and is the defensible specification here.
    r = STL(s, period=365, robust=False).fit()
    var = lambda a: np.var(a, ddof=1)
    ts = max(0.0, 1 - var(r.resid) / var(r.trend + r.resid))
    ss = max(0.0, 1 - var(r.resid) / var(r.seasonal + r.resid))
    return ts, ss


# ------------------------------------------------ Table: series diagnostics
def series_diagnostics():
    rows = []
    for pid in ORDER:
        s = PANEL[pid].cf.asfreq("D").interpolate()
        adf_p = adfuller(s, autolag="AIC")[1]
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
        mk_t, mk_p = mann_kendall(s.values)
        ts, ss = strengths(s)
        d365 = s.diff(365).dropna()
        rows.append({
            "Region": SHORT[pid],
            "ADF p": f"{adf_p:.3f}",
            "KPSS p": f"{'>0.10' if kpss_p >= 0.1 else f'{kpss_p:.3f}'}",
            "Mann-Kendall": mk_t,
            "MK p": f"{mk_p:.3f}",
            "Trend": f"{ts:.3f}",
            "Seasonal": f"{ss:.3f}",
            "ADF p, lag-365 diff": f"{adfuller(d365, autolag='AIC')[1]:.1e}",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------ Table: data sources
def data_sources():
    span = {}
    for pid in ORDER:
        d = PANEL[pid]
        span[pid] = (d.index.min().date(), d.index.max().date(), len(d))
    lo = min(v[0] for v in span.values())
    hi = max(v[1] for v in span.values())
    n = min(v[2] for v in span.values())
    return pd.DataFrame([
        {"Source": "PV_Live (Sheffield Solar)",
         "Variables used": "Half-hourly regional generation (MW) and installed "
                           "capacity (MWp)",
         "Coverage": f"6 of 14 GB PES regions; {lo} to {hi}; "
                     f"{n} daily records per region",
         "Role": "Ground truth for measured generation"},
        {"Source": "Open-Meteo ERA5 reanalysis",
         "Variables used": "Shortwave, direct and diffuse radiation; "
                           "temperature; relative humidity; cloud cover; "
                           "wind speed; precipitation",
         "Coverage": f"Region centroid; {lo} to {hi}; hourly, "
                     f"aggregated to daily means",
         "Role": "Exogenous weather information set"},
        {"Source": "PVGIS v5.2 (European Commission JRC)",
         "Variables used": "Modelled annual and monthly yield, 4 kWp array, "
                           "35° tilt, due south, 14% system loss",
         "Coverage": "Region centroid; typical meteorological year; "
                     "6 annual and 72 monthly estimates",
         "Role": "Vendor-analogue benchmark"},
        {"Source": "DESNZ solar PV cost statistics",
         "Variables used": "Mean installed cost per kW, domestic 0–4 kW "
                           "band, Declared Net Capacity basis",
         "Coverage": "Great Britain; financial year 2025/26; "
                     "107,137 installations",
         "Role": "Capital cost input to the techno-economic model"},
    ])


# -------------------------------------------------------- Table: SHAP top 10
def shap_top10():
    df = pd.read_csv("shap_importance.csv", index_col=0)
    t = df[["mean_abs_shap"]].sort_values("mean_abs_shap", ascending=False).head(10)
    t = t.reset_index()
    t.columns = ["Feature", "Mean |SHAP|"]
    t["Rank"] = range(1, len(t) + 1)
    t["Mean |SHAP|"] = t["Mean |SHAP|"].map(lambda v: f"{v:.5f}")
    return t[["Rank", "Feature", "Mean |SHAP|"]]


# ------------------------------------------------- Table: deployment model
def deployment():
    """Deployment evaluation on the selected 2013-2022 climatology window.

    This is the window economics.py declares (CLIM_START/CLIM_END) and the one
    the chapter selects, so the table and the text now report the same run.
    """
    d = pd.read_csv("climatology_longrun.csv")
    d["Region"] = d.region.str.split(",").str[0]
    out = pd.DataFrame({
        "Region": d.Region,
        "Measured, 3-year mean (kWh/kWp)": d.actual_3yr_mean.map("{:.0f}".format),
        "Pooled model (kWh/kWp)": d.model_climatology_longrun.map("{:.0f}".format),
        "Model error (%)": d.model_err_pct.map("{:+.1f}".format),
        "PVGIS (kWh/kWp)": d.pvgis.map("{:.0f}".format),
        "PVGIS error (%)": d.pvgis_err_pct.map("{:+.1f}".format),
    })
    order = [SHORT[p] for p in ORDER]
    out = out.set_index("Region").loc[order].reset_index()
    mean = {
        "Region": "Mean absolute error",
        "Measured, 3-year mean (kWh/kWp)": "", "Pooled model (kWh/kWp)": "",
        # From the full-precision run (economics.DEPLOYED_ACCURACY_PCT);
        # the per-region columns above are the published one-decimal values,
        # so re-averaging them would give 5.75 / 16.60 and disagree with 4.6.
        "Model error (%)": f"{E.DEPLOYED_ACCURACY_PCT:.2f}",
        "PVGIS (kWh/kWp)": "",
        "PVGIS error (%)": "16.61",
    }
    return pd.concat([out, pd.DataFrame([mean])], ignore_index=True)


# ------------------------------------------- Table: order of averaging
def averaging_order():
    """Averaging the weather vs averaging the annual output, same record."""
    d = pd.read_csv("averaging_order.csv")
    out = pd.DataFrame({
        "Region": d.region,
        "Measured (kWh/kWp)": d.measured.map("{:.0f}".format),
        "Typical year (kWh/kWp)": d.average_the_input.map("{:.0f}".format),
        "Error (%)": d.err_input_pct.map("{:+.2f}".format),
        "Per-year runs (kWh/kWp)": d.average_the_output.map("{:.0f}".format),
        "Error (%) ": d.err_output_pct.map("{:+.2f}".format),
    })
    mean = {"Region": "Mean absolute error",
            "Measured (kWh/kWp)": "", "Typical year (kWh/kWp)": "",
            "Error (%)": f"{d.err_input_pct.abs().mean():.2f}",
            "Per-year runs (kWh/kWp)": "",
            "Error (%) ": f"{d.err_output_pct.abs().mean():.2f}"}
    return pd.concat([out, pd.DataFrame([mean])], ignore_index=True)


# -------------------------------------------- Table: payback sensitivity grid
def payback(annual_kwh_y1, capital, import_price, self_consumption):
    """Simple payback, linearly interpolated within the crossing year.

    Same arithmetic as payback_comparison.payback_years, but with the two
    swept levers passed in rather than read from module constants.
    """
    cum = prev = -capital
    for t in range(1, E.HORIZON_YEARS + 1):
        gen = annual_kwh_y1 * (1 - E.DEGRADATION) ** (t - 1)
        saving = (gen * self_consumption * import_price
                  + gen * (1 - self_consumption) * E.SEG_RATE)
        cost = E.INVERTER_REPLACE[1] if t == E.INVERTER_REPLACE[0] else 0.0
        prev, cum = cum, cum + saving - cost
        if cum >= 0:
            return t - 1 + (0 - prev) / (cum - prev)
    return np.nan


def sensitivity():
    imports = [0.20, 0.2232, 0.2632, 0.30, 0.34]
    selfcons = [0.25, 0.30, 0.35, 0.40, 0.45]
    P = pd.read_csv("payback_by_method.csv")
    gen = P.groupby("region").actual_kwh.first().mean()
    cap = E.capital_cost()
    rows = []
    for sc in selfcons:
        r = {"Self-consumption": f"{sc:.0%}"}
        for ip in imports:
            y = payback(gen, cap, ip, sc)
            r[f"{ip*100:.2f}p"] = "beyond horizon" if np.isnan(y) else f"{y:.1f}"
        rows.append(r)
    print(f"   [sensitivity built on measured mean {gen:,.0f} kWh/yr, "
          f"capital £{cap:,.0f}]")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for name, fn in [
            ("data_sources", data_sources),
            ("series_diagnostics", series_diagnostics),
            ("shap_top10", shap_top10),
            ("deployment", deployment),
            ("sensitivity", sensitivity),
            ("averaging_order", averaging_order)]:
        try:
            t = fn()
            t.to_csv(f"{OUT}/{name}.csv", index=False)
            print(f"--- {name} ---")
            print(t.to_string(index=False))
            print()
        except Exception as exc:
            print(f"!! {name} FAILED: {exc}\n")
