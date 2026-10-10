"""Efficiency drift versus weather-aware model error, and where -0.85 came from.

The claim under test (§5.2 of the draft): the residual error of the
weather-aware models is not noise — it tracks a drift in the fleet's realised
conversion efficiency that the models cannot see, because their predictors are
meteorological and the drift is not.

Drift is measured as the change in **capacity factor per unit of shortwave
irradiance** between the first and last full calendar year in the dataset. That
ratio is the fleet's realised conversion efficiency: how much of the available
resource it actually turns into output. Normalising by irradiance is what makes
the measure a statement about the fleet rather than about the weather — a dull
year lowers capacity factor without changing efficiency, and dividing the one by
the other removes that. It is computed on daylight hours only, as a ratio of
annual sums rather than a mean of hourly ratios; the two are not the same thing
and only the former is a fleet-level efficiency.


THE PROVENANCE OF -0.85 AND -0.87
---------------------------------

The draft reports r = -0.85 for Random Forest and -0.87 for XGBoost. Those
figures are real and this script reproduces them — but at the **30-day forecast
horizon**, which is not the horizon the draft's accuracy figures come from:

    horizon    RF Pearson    XGB Pearson
    h = 1        -0.780        -0.780
    h = 7        -0.884        -0.842
    h = 30       -0.851        -0.872     <- the draft's -0.85 / -0.87

The accuracy the same paragraph is explaining — "the remaining 1.35% to 1.42%
annual error" — is the mean absolute annual error at **h = 1** (1.345% for
Random Forest, 1.423% for XGBoost). So the draft quotes the error at one horizon
and the correlation that explains it at another. Nothing was miscalculated;
two horizons were mixed. The correlation reported in §5.2 should be the h = 1
one, to match the error it is explaining.

That the relationship holds at every horizon, and strengthens at longer ones, is
a point in favour of the drift interpretation rather than against it, and is
worth stating as a robustness check instead of quietly selecting the strongest.


WHICH COEFFICIENT TO REPORT
---------------------------

At the matching horizon:

    Random Forest   Spearman rho = -0.829  p = 0.042    Pearson r ~ -0.78
    XGBoost         Spearman rho = -0.886  p = 0.019    Pearson r ~ -0.78

Report the Spearman coefficient, for a reason this script makes visible. The
drift column is deterministic: it comes from the observed dataset and is
identical to every decimal place on every run. The model error column is not.
The ensembles are fitted with a fixed seed, but parallel fitting and library
versions still move the annual errors by hundredths of a percentage point
between runs, and that is enough to move Pearson r from -0.776 to -0.808 —
across the conventional threshold. XGBoost came out at p = 0.052 on one run and
p = 0.067 on another.

The ranks did not move at all. Spearman was identical in both runs.

So on six points the rank correlation is the defensible statistic: it is stable
under exactly the perturbation that destabilises Pearson, it assumes no
linearity, and no single region can drag it. Give Pearson beside it as
approximately -0.78 and attach no significance claim to a coefficient whose
p-value depends on thread scheduling.

Even with Spearman, n = 6. The result is consistent with the drift explanation;
it does not demonstrate it. "Suggestive", with the coefficient and p-value
stated — not "almost perfectly predicts".

Run from the scripts directory with the data files alongside:

    cd scripts && cp ../data/* . && python drift_correlation.py
"""
import numpy as np
import pandas as pd
from scipy import stats

DATA = "solar_weather_dataset.csv"
BACKTEST = "ml_backtest_summary.csv"
OUT = "drift_correlation.csv"

FIRST_YEAR, LAST_YEAR = 2023, 2025
HORIZONS = [1, 7, 30]
REPORT_H = 1          # the horizon the draft's accuracy figures come from
MODELS = ["RF", "XGB"]


# ------------------------------------------------- 1  efficiency drift by region
def efficiency_by_year(df):
    """Capacity factor per unit shortwave irradiance, daylight hours, by year."""
    d = df[df.is_daylight == 1]
    g = d.groupby(d.timestamp.dt.year)
    return g.capacity_factor.sum() / g.shortwave_radiation.sum()


raw = pd.read_csv(DATA, parse_dates=["timestamp"],
                  usecols=["timestamp", "pes_id", "region_name",
                           "capacity_factor", "shortwave_radiation",
                           "is_daylight"])
print(f"loaded {len(raw):,} hourly rows, "
      f"{raw.timestamp.min().date()} to {raw.timestamp.max().date()}")

rows = []
for (pes, name), df in raw.groupby(["pes_id", "region_name"]):
    eff = efficiency_by_year(df)
    missing = [y for y in (FIRST_YEAR, LAST_YEAR) if y not in eff.index]
    if missing:
        raise SystemExit(f"{name}: no data for {missing}")
    rows.append(dict(pes_id=pes, region=name,
                     eff_first=eff[FIRST_YEAR], eff_last=eff[LAST_YEAR],
                     drift_pct=100.0 * (eff[LAST_YEAR] / eff[FIRST_YEAR] - 1.0)))
drift = pd.DataFrame(rows).sort_values("pes_id").reset_index(drop=True)


# ------------------------------------------------- 2  model error by horizon
ml = pd.read_csv(BACKTEST)
ml = ml[ml.info_set == "weather-aware"]


def merged(h, model):
    g = ml[(ml.h == h) & (ml.model == model)][["pes_id", "annual_error_pct"]]
    m = drift.merge(g, on="pes_id")
    if len(m) != len(drift):
        raise SystemExit(f"h={h} {model}: expected {len(drift)} regions, got {len(m)}")
    return m


# ------------------------------------------------- 3  the horizon comparison
print("\ndrift correlation at every horizon "
      "(the draft's -0.85/-0.87 is the h=30 row)\n")
print(f"  {'horizon':>8}   {'RF Pearson':>11}  {'RF Spearman':>12}   "
      f"{'XGB Pearson':>12}  {'XGB Spearman':>13}")
for h in HORIZONS:
    cells = []
    for model in MODELS:
        m = merged(h, model)
        r, _ = stats.pearsonr(m.drift_pct, m.annual_error_pct)
        rs, _ = stats.spearmanr(m.drift_pct, m.annual_error_pct)
        cells += [f"{r:+.3f}", f"{rs:+.3f}"]
    flag = "   <- reported in §5.2" if h == REPORT_H else ""
    print(f"  {'h = ' + str(h):>8}   {cells[0]:>11}  {cells[1]:>12}   "
          f"{cells[2]:>12}  {cells[3]:>13}{flag}")

print("\n  The relationship holds at every horizon and strengthens at longer")
print("  ones. Reporting h=1 keeps the correlation on the same horizon as the")
print("  accuracy figures it explains, rather than selecting the strongest.")


# ------------------------------------------------- 4  the reported horizon
print(f"\n\nefficiency drift {FIRST_YEAR}->{LAST_YEAR} against h={REPORT_H} "
      f"annual error\n")
tab = drift.copy()
for model in MODELS:
    tab[f"{model} error %"] = merged(REPORT_H, model).annual_error_pct.values

show = tab[["region", "drift_pct"] + [f"{m} error %" for m in MODELS]].copy()
show.columns = ["region", "drift %", "RF error %", "XGB error %"]
print(show.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

print(f"\ncorrelation, n = {len(tab)}\n")
for model in MODELS:
    m = merged(REPORT_H, model)
    rs, ps = stats.spearmanr(m.drift_pct, m.annual_error_pct)
    r, p = stats.pearsonr(m.drift_pct, m.annual_error_pct)
    # Fisher z interval, to make the width of the uncertainty explicit
    z, se = np.arctanh(r), 1.0 / np.sqrt(len(m) - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    print(f"  {model:4s}  Spearman rho = {rs:+.3f}  p = {ps:.3f}"
          f"       <- report this")
    print(f"  {'':4s}  Pearson  r   = {r:+.3f}  p = {p:.3f}   "
          f"95% CI [{lo:+.2f}, {hi:+.2f}]")
    tab[f"rho_{model}"], tab[f"p_rho_{model}"] = rs, ps
    tab[f"r_{model}"], tab[f"p_r_{model}"] = r, p

print("""
  Report Spearman. The drift column is deterministic, but the model error
  column moves by hundredths of a point between fitting runs, and that is
  enough to push Pearson r anywhere from -0.776 to -0.808 -- across the 0.05
  threshold. The ranks do not move, so Spearman does not either.

  n = 6 regardless. State the coefficient and p-value and call the result
  suggestive. Do not write "almost perfectly predicts".""")

tab.to_csv(OUT, index=False)
print(f"\nwrote {OUT}")
