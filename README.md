# Solar generation estimates and the payback error borne by UK households

Analysis code and data for an MSc Business Analytics dissertation (AL4IIP,
Aston University, 2026).

The project tests whether the generation estimates UK households are given when
they consider rooftop solar are systematically wrong, and whether the error is
large enough to change the payback period. Three estimation methods are
compared against measured regional output for six Great Britain regions over
2023 to 2025, and the differences between them are translated into payback
years.

Every figure, table and number in the dissertation is produced by the scripts
here. Nothing was typed by hand.

---

## Data sources

All inputs are publicly available. The merged dataset is included in this
repository so the analysis can be re-run without re-downloading anything.

| Source | Used for | Link |
|---|---|---|
| PV_Live (Sheffield Solar) | Half-hourly regional generation and installed capacity | https://www.solar.sheffield.ac.uk/pvlive/ |
| Open-Meteo historical weather API (ERA5 reanalysis) | Irradiance, temperature, cloud cover, wind, precipitation | https://open-meteo.com/en/docs/historical-weather-api |
| PVGIS v5.2 (European Commission JRC) | Vendor-equivalent generation estimates | https://re.jrc.ec.europa.eu/pvg_tools/en/ |
| DESNZ solar PV cost data | Installed cost per kW | https://www.gov.uk/government/statistics/solar-pv-cost-data |
| Ofgem energy price cap | Electricity import price | https://www.ofgem.gov.uk/information-consumers/energy-advice-households/energy-price-cap-unit-rates-and-standing-charges |

Regions are six of the fourteen GB Public Electricity Supplier areas: Crawley,
Bury St Edmunds, Castle Donington, Castleford, Cramlington and Perth.

---

## Setup

```
pip install -r requirements.txt
gunzip solar_weather_dataset.csv.gz
```

The merged dataset is stored compressed (4.6 MB packed, 29 MB unpacked) so that
every file in this repository stays within GitHub's browser upload limit. Unzip
it once before running anything.

Python 3.11. The scripts read and write CSV files in the working directory, so
run them from the repository root rather than from elsewhere.

`pooled_model.joblib` is not committed (78 MB). Regenerate it with
`python pooled_model.py`, or point the scripts at an existing copy:

```
POOLED_MODEL=/path/to/pooled_model.joblib python decompose_climatology.py
```

---

## Headline results

Re-running the pipeline reproduces the figures reported in the dissertation:

| | |
|---|---|
| Vendor-equivalent estimate (PVGIS) | 10.13% mean absolute annual error |
| Statistical baseline, no weather | 2.91% |
| Random Forest / XGBoost, no weather | 7.68% / 9.18% |
| Random Forest / XGBoost, with weather | 1.35% / 1.42% |
| Payback, vendor estimate vs measured | 22.1 years vs 24.2 years |
| Order-of-averaging correction | recovers 3.85 percentage points |

Financial assumptions: £2,095.51/kW installed (£8,382 total for 4 kWp), 26.32
p/kWh import, 5 p/kWh export, 35% self-consumption, 0.5%/yr degradation, 25
year horizon, 5% discount rate. All are defined once in `economics.py` and
imported everywhere else, so the tables cannot drift from the code.

---

## Run order

**1. Build the dataset** (needs network access; the outputs are already
included, so this step can be skipped)

```
python build_dataset.py     # PV_Live + Open-Meteo -> solar_weather_dataset.csv
python fetch_pvgis.py       # -> pvgis_estimates.csv
python merge_pvgis.py
```

**2. Diagnostics and the controlled experiment**

```
python diagnose_panel.py             # missing-data audit
python timeseries_diagnostics.py     # ADF, KPSS, Mann-Kendall, STL
python backtest.py                   # statistical baseline, walk-forward
python ml_backtest.py                # Random Forest and XGBoost, both information sets
python annual_estimates.py           # daily predictions -> one annual figure per region
python full_year_ladder.py           # all six methods on a complete calendar year
python compare_pvgis.py
```

**3. Robustness checks**

```
python tuning_full.py                # 34 hyperparameter configurations
python finish_xgb_wx.py              # completes the weather-aware XGBoost arm
python shap_rf.py                    # predictor importance, Random Forest
python hourly_dual_seasonality.py    # tests the daily-aggregation decision
python drift_correlation.py          # residual error against fleet efficiency drift
```

**4. Deployment model and the climatology question**

```
python pooled_model.py          # pooled model without lag features -> .joblib
python extend_climatology.py    # long-run weather record (needs network)
python irradiance_trend.py
python trend_significance.py
python decompose_climatology.py # smoothing vs window; the order-of-averaging result
python postcode_predict.py      # the deployed tool, postcode to annual estimate
```

**5. Outputs for the dissertation**

```
python economics.py --csv > assumptions_table.csv
python payback_comparison.py
python make_figures.py          # -> figures/
python make_tables.py           # -> tables/
```

---

## Reproducibility

Tree models use a fixed random seed of 42. Even so, parallel fitting means
annual errors can move by a few hundredths of a percentage point between
machines or library versions. This matters in one place: the correlation
between fleet efficiency drift and residual model error moves between -0.776
and -0.808 across runs, which straddles the conventional significance
threshold, while the rank correlation is unchanged. `drift_correlation.py`
demonstrates this and explains why the rank coefficient is the one to report.

`economics.py` stamps its assumptions with the date they were checked
(`AS_OF = 2026-09-15`) and prints a warning if they are more than 90 days old,
since the Ofgem price cap changes quarterly. The values are deliberately frozen
rather than refreshed, so that the numbers in the dissertation stay
reproducible.

---

## Scope notes

**Some scripts produce analyses that are not in the final dissertation.** The
hourly dual-seasonality test, the full SHAP attribution, the efficiency drift
correlation and the irradiance trend analysis were all carried out and then cut
when the dissertation was shortened and simplified. The code and outputs are
kept here because the decisions they informed are still described in the text,
and because the dual-seasonality result is cited as the justification for
aggregating to daily totals.

`build_document.py` assembles the Word file from the figures and tables. It is
document production rather than analysis, and is included only for
completeness.

`archive/` holds earlier versions of three figure scripts, superseded by
`make_figures.py`. They are kept for transparency and are not part of the
pipeline.

---

## Declaration

Generative AI tools were used in developing the analysis code, as declared in
the dissertation. All results reported were produced by executing these scripts
on the data in this repository.
