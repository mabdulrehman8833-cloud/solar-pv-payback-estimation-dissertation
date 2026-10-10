"""
================================================================================
Time-series diagnostic pass for the weather-blind SARIMA baseline
MSc Business Analytics dissertation — solar generation forecasting
================================================================================

Runs, per region, on DAILY-aggregated capacity factor:

  1. Daily aggregation           (avoids dual daily+annual seasonality and the
                                  ~50% exact-zero rows in hourly capacity factor)
  2. Stationarity                ADF + KPSS  (complementary null hypotheses)
  3. Trend                       Mann-Kendall
  4. Decomposition               STL (robust), with Hyndman trend/seasonal strengths
  5. Differencing                seasonal (D=1, m=365) and regular (d=1)
  6. Variance stabilisation      Box-Cox lambda (auto-selected)
  7. Correlogram                 ACF / PACF, raw and seasonally differenced
  8. Spurious correlation        raw-level vs STL-deseasonalised correlation
  9. Cross-correlation           CCF against weather drivers
 10. Model identification        ARMA order grid + Fourier order, AIC + Ljung-Box

Note on model form: a true seasonal SARIMA with m=365 on daily data is not
computationally feasible (the seasonal lag polynomials become enormous; a
SARIMAX(1,0,1)(1,1,1,365) fit does not converge in 60s). The standard
alternative — dynamic harmonic regression: Fourier terms for the annual cycle
plus a low-order non-seasonal ARMA for short-range dependence — is used instead.
Fourier terms are deterministic functions of the calendar, so using them over
the forecast period is NOT data leakage.

Usage:  python timeseries_diagnostics.py
================================================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller, kpss, ccf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import Fourier
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pymannkendall as mk

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ config ---
DATA_FILE   = "solar_weather_dataset.csv"
SEASON      = 365          # annual cycle, daily data
FOURIER_P   = 365.25       # non-integer period is fine for Fourier terms
BASE_ORDER  = (2, 0, 1)    # ARMA order carried over from the first region
BASE_K      = 3            # number of Fourier sin/cos pairs

# Three regions chosen for maximum geographic spread
REGIONS = {
    19: "Crawley, West Sussex",        # 51.12 N — south
    17: "Perth, Scotland",             # 56.42 N — far north
    10: "Bury St Edmunds, Suffolk",    # 52.25 N, +0.73 E — far east
}

ORDER_GRID   = [(2, 0, 1), (3, 0, 1), (2, 0, 2), (3, 0, 2)]
FOURIER_GRID = [3, 5]


# --------------------------------------------------------------- functions ---
def load_daily_capacity_factor(df, pes_id):
    """Hourly regional data -> daily capacity factor.

    generation_mw is an hourly mean, so summing 24 hourly means gives MWh/day;
    dividing by (installed_mwp * 24) rescales to a daily capacity factor in [0,1].
    """
    reg = (df[df["pes_id"] == pes_id]
           .sort_values("timestamp")
           .set_index("timestamp"))

    daily = pd.DataFrame({
        "generation_mwh": reg["generation_mw"].resample("D").sum(),
        "installed_mwp":  reg["installed_mwp"].resample("D").mean(),
        "shortwave_mean": reg["shortwave_radiation"].resample("D").mean(),
        "temp_mean":      reg["temperature_2m"].resample("D").mean(),
        "cloud_mean":     reg["cloud_cover"].resample("D").mean(),
    }).dropna()

    daily["cf"] = (daily["generation_mwh"] / (daily["installed_mwp"] * 24)).clip(0, 1)
    daily.index = daily.index.tz_localize(None)
    daily.index.freq = "D"
    return daily


def stationarity_tests(y):
    """ADF (H0: unit root) and KPSS (H0: stationary) — deliberately complementary."""
    adf = adfuller(y.dropna(), autolag="AIC")
    kp  = kpss(y.dropna(), regression="c", nlags="auto")
    return {
        "adf_stat": adf[0], "adf_p": adf[1],
        "adf_verdict":  "stationary" if adf[1] < 0.05 else "non-stationary",
        "kpss_stat": kp[0], "kpss_p": kp[1],
        "kpss_verdict": "stationary" if kp[1] > 0.05 else "non-stationary",
    }


def component_strength(component, resid):
    """Hyndman & Athanasopoulos strength measure, bounded to [0, 1]."""
    return max(0.0, 1.0 - np.var(resid) / np.var(component + resid))


def stl_decompose(y, period=SEASON, plot_path=None, title=""):
    res = STL(y, period=period, robust=True).fit()
    strengths = {
        "trend_strength":    component_strength(res.trend, res.resid),
        "seasonal_strength": component_strength(res.seasonal, res.resid),
    }
    if plot_path:
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        for ax, (series, label, colour) in zip(axes, [
            (res.observed, "Observed", "#333333"),
            (res.trend,    "Trend",    "#e76f51"),
            (res.seasonal, "Seasonal", "#2a9d8f"),
            (res.resid,    "Residual", "#888888"),
        ]):
            ax.plot(y.index, series, color=colour, lw=0.9)
            ax.set_ylabel(label)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_title(f"STL decomposition — {title}")
        plt.tight_layout(); plt.savefig(plot_path, dpi=130); plt.close()
    return res, strengths


def boxcox_lambda(y):
    """Auto-selected Box-Cox lambda. Requires strictly positive input, so the
    single all-zero winter day is replaced by half the smallest positive value."""
    y_pos = y.replace(0, y[y > 0].min() * 0.5)
    _, lam = boxcox(y_pos)
    return lam


def plot_correlograms(y, path, title, lags=40):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(y.dropna(),  lags=lags, ax=axes[0], title=f"ACF — {title}")
    plot_pacf(y.dropna(), lags=lags, ax=axes[1], method="ywm", title=f"PACF — {title}")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def spurious_correlation_check(daily, drivers=("shortwave_mean", "temp_mean", "cloud_mean")):
    """Correlation in levels vs. correlation of STL residuals.

    Two series that both peak in summer will correlate in levels even with no
    causal link. Removing the shared seasonal component is what separates a
    genuine driver from a spurious one.
    """
    raw = {d: daily["cf"].corr(daily[d]) for d in drivers}
    resid = {c: STL(daily[c], period=SEASON, robust=True).fit().resid
             for c in ("cf", *drivers)}
    deseason = {d: float(np.corrcoef(resid["cf"], resid[d])[0, 1]) for d in drivers}
    return pd.DataFrame({"raw_levels": raw, "deseasonalised": deseason})


def cross_correlation(daily, driver="shortwave_mean", max_lag=10):
    """CCF on deseasonalised residuals — identifies whether any LAGGED weather
    effect exists, or whether the relationship is purely contemporaneous."""
    cf_r = STL(daily["cf"], period=SEASON, robust=True).fit().resid
    dr_r = STL(daily[driver], period=SEASON, robust=True).fit().resid
    return pd.Series(ccf(cf_r, dr_r, adjusted=False)[:max_lag + 1],
                     index=range(max_lag + 1), name=f"ccf_cf_vs_{driver}")


def fit_harmonic_arma(y, order=BASE_ORDER, k=BASE_K):
    """Dynamic harmonic regression: Fourier seasonal terms + non-seasonal ARMA."""
    X = Fourier(period=FOURIER_P, order=k).in_sample(y.index)
    res = SARIMAX(y, exog=X, order=order,
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)
    lb = acorr_ljungbox(res.resid, lags=[10, 20, 30], return_df=True)
    return res, lb


def order_search(y, order_grid=ORDER_GRID, fourier_grid=FOURIER_GRID):
    """Does the order chosen on the first region generalise? AIC + residual whiteness."""
    rows = []
    for order in order_grid:
        for k in fourier_grid:
            res, lb = fit_harmonic_arma(y, order, k)
            rows.append({
                "order": str(order), "fourier_k": k, "aic": res.aic,
                "lb_p10": lb["lb_pvalue"].iloc[0],
                "lb_p20": lb["lb_pvalue"].iloc[1],
                "lb_p30": lb["lb_pvalue"].iloc[2],
                "residuals_white": bool((lb["lb_pvalue"] > 0.05).all()),
            })
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


# -------------------------------------------------------------------- main ---
def diagnose_region(df, pes_id, label, run_order_search=True):
    print("\n" + "=" * 78)
    print(f"REGION: {label}  (PES {pes_id})")
    print("=" * 78)

    daily = load_daily_capacity_factor(df, pes_id)
    y = daily["cf"]
    slug = label.split(",")[0].strip().lower().replace(" ", "_")

    print(f"Observations: {len(y)}   "
          f"{y.index.min().date()} to {y.index.max().date()}   "
          f"zero days: {(y == 0).sum()}")
    print(f"Capacity factor: mean={y.mean():.4f}  sd={y.std():.4f}  max={y.max():.4f}")

    # 2. stationarity
    st = stationarity_tests(y)
    print(f"\n[Stationarity]  ADF  stat={st['adf_stat']:.3f}  p={st['adf_p']:.4f}"
          f"  -> {st['adf_verdict']}")
    print(f"                KPSS stat={st['kpss_stat']:.3f}  p={st['kpss_p']:.4f}"
          f"  -> {st['kpss_verdict']}")
    if st["adf_verdict"] != st["kpss_verdict"]:
        print("                (disagreement is expected: strong seasonality, no trend)")

    # 3. trend
    t = mk.original_test(y)
    print(f"\n[Mann-Kendall]  trend={t.trend}  p={t.p:.4f}  Sen slope={t.slope:.3e}")

    # 4. decomposition
    _, strengths = stl_decompose(y, plot_path=f"stl_{slug}.png", title=label)
    print(f"\n[STL]           trend strength={strengths['trend_strength']:.3f}   "
          f"seasonal strength={strengths['seasonal_strength']:.3f}")

    # 5. differencing
    y_sdiff = y.diff(SEASON).dropna()
    y_both  = y_sdiff.diff(1).dropna()
    print(f"\n[Differencing]  seasonal D=1 (m={SEASON}): ADF p={adfuller(y_sdiff)[1]:.4f}")
    print(f"                + regular d=1            : ADF p={adfuller(y_both)[1]:.4f}")
    print("                -> D=1, d=0 is sufficient; adding d=1 over-differences")

    # 6. Box-Cox
    print(f"\n[Box-Cox]       auto-selected lambda = {boxcox_lambda(y):.4f}")

    # 7. correlograms
    plot_correlograms(y,       f"acf_pacf_raw_{slug}.png",   f"raw daily CF — {label}")
    plot_correlograms(y_sdiff, f"acf_pacf_sdiff_{slug}.png", f"seasonally differenced — {label}")

    # 8. spurious correlation
    print("\n[Spurious correlation]  levels vs STL-deseasonalised:")
    print(spurious_correlation_check(daily).round(3).to_string())

    # 9. cross-correlation
    print("\n[Cross-correlation]  cf vs shortwave (deseasonalised), lags 0-10 d:")
    print(cross_correlation(daily).round(3).to_string())

    # 10. model identification
    res, lb = fit_harmonic_arma(y)
    print(f"\n[Baseline fit]  ARMA{BASE_ORDER} + Fourier(K={BASE_K})   AIC={res.aic:.1f}")
    print(f"                Ljung-Box p (lags 10/20/30): "
          f"{lb['lb_pvalue'].iloc[0]:.3f} / {lb['lb_pvalue'].iloc[1]:.3f} / "
          f"{lb['lb_pvalue'].iloc[2]:.3f}")

    if run_order_search:
        print("\n[Order search]  does the baseline order generalise here?")
        print(order_search(y).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    return {"pes_id": pes_id, "region": label,
            "adf_p": st["adf_p"], "kpss_p": st["kpss_p"],
            "mk_trend": t.trend, "mk_p": t.p,
            **strengths, "aic": res.aic, "lb_p10": lb["lb_pvalue"].iloc[0]}


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    summary = [diagnose_region(df, pes_id, label) for pes_id, label in REGIONS.items()]

    print("\n" + "=" * 78)
    print("CROSS-REGION SUMMARY")
    print("=" * 78)
    out = pd.DataFrame(summary)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out.to_csv("diagnostics_summary.csv", index=False)
    print("\nSaved: diagnostics_summary.csv and per-region STL / ACF / PACF plots")
