"""
Is the irradiance trend statistically real, or is 13 years of decadal variability
being read as a trend?

The OLS slopes in irradiance_trend.py came with no uncertainty attached, and two
things about them look suspicious:

  * +5.5%/decade on average is LARGE relative to published European brightening
    rates -- check the literature for the accepted figure and compare.
  * the spatial spread is incoherent: Perth +8.2%/decade vs Crawley +1.9%/decade.
    Aerosol-driven brightening should be fairly uniform across a country the size
    of the UK. A 4x spread across six sites points to noise, not signal.

This script attaches proper uncertainty:

  1. OLS slope with p-value and 95% CI          (parametric)
  2. Mann-Kendall test + Sen's slope            (non-parametric, robust to outliers)
  3. Leave-one-year-out slope stability         (is one sunny year driving it?)

With n=13 annual points, expect wide intervals. If the CI spans zero, you cannot
claim a trend -- only an upward tendency consistent with, but not demonstrating,
the documented brightening literature.

Requires: longrun_weather.csv
"""
import numpy as np, pandas as pd
from scipy import stats
import pymannkendall as mk

pd.set_option("display.width", 200)

W = pd.read_csv("longrun_weather.csv", parse_dates=["timestamp"])
W["year"] = W["timestamp"].dt.year

rows = []
for region, g in W.groupby("region"):
    a = g.groupby("year")["shortwave_radiation"].mean()
    x, y = a.index.values.astype(float), a.values

    lr = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, len(x) - 2)
    lo, hi = lr.slope - tcrit*lr.stderr, lr.slope + tcrit*lr.stderr

    m = mk.original_test(y)

    # leave-one-year-out: how much does any single year move the slope?
    slopes = [stats.linregress(np.delete(x, i), np.delete(y, i)).slope
              for i in range(len(x))]
    worst_i = int(np.argmax(np.abs(np.array(slopes) - lr.slope)))

    rows.append({
        "region": region.split(",")[0],
        "n_years": len(x),
        "slope": lr.slope,
        "ci_lo": lo, "ci_hi": hi,
        "p_ols": lr.pvalue,
        "r2": lr.rvalue**2,
        "mk_trend": m.trend, "p_mk": m.p, "sen_slope": m.slope,
        "pct_per_decade": 100*lr.slope*10/y.mean(),
        "loo_slope_range": max(slopes) - min(slopes),
        "most_influential_year": int(x[worst_i]),
    })

T = pd.DataFrame(rows)

print("="*104)
print("TREND SIGNIFICANCE — annual mean irradiance, 2010-2022 (n=13 per region)")
print("="*104)
print(T[["region","slope","ci_lo","ci_hi","p_ols","r2","pct_per_decade"]]
        .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n" + "="*104)
print("NON-PARAMETRIC CHECK — Mann-Kendall + Sen's slope")
print("="*104)
print(T[["region","mk_trend","p_mk","sen_slope","slope"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n" + "="*104)
print("ROBUSTNESS — how much one year can move the slope")
print("="*104)
print(T[["region","slope","loo_slope_range","most_influential_year"]]
        .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

sig_ols = (T.p_ols < 0.05).sum()
sig_mk  = (T.p_mk  < 0.05).sum()
ci_excl = ((T.ci_lo > 0) | (T.ci_hi < 0)).sum()

print(f"""
{'='*104}
VERDICT
{'='*104}
  regions with OLS p < 0.05           : {sig_ols}/6
  regions with Mann-Kendall p < 0.05  : {sig_mk}/6
  regions whose 95% CI excludes zero  : {ci_excl}/6

If most regions fail these tests, write it as an upward TENDENCY that is
consistent with the documented European brightening literature but not
independently demonstrated by 13 years of reanalysis at six points. That is the
defensible claim, and it still fully supports the practical finding: the choice
of climatology window materially changes the estimate, so it must be stated and
its sensitivity reported.

To strengthen it, extend the window. ERA5 runs back to 1940; set
CLIM_START='1995-01-01' in extend_climatology.py, delete wx_cache/, and re-run.
Roughly 28 annual points would separate trend from decadal variability far better
than 13 -- and would let you compare directly against published brightening rates.
""")
