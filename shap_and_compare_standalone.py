"""
SHAP attribution for the weather-aware XGBoost + the combined
SARIMA vs RF vs XGB comparison table.

STANDALONE — no import of ml_backtest.py. Paste straight into a Colab cell.
Requires in the working directory:
    solar_weather_dataset.csv     (from the dataset builder)
    backtest_summary.csv          (from backtest.py)
    ml_backtest_summary.csv       (from ml_backtest.py)

Colab setup cell (note: NO `pip install ml_backtest` — that is an unrelated
PyPI package, not this project):
    !pip install statsmodels==0.14.6 pandas==2.2.3 shap xgboost scikit-learn
"""
import numpy as np, pandas as pd, shap, warnings
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

INITIAL_TRAIN = 730
SEED = 42
WEATHER_COLS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
                "temperature_2m", "relative_humidity_2m", "cloud_cover",
                "wind_speed_10m", "precipitation"]


def load_daily(df, pes_id):
    reg = df[df["pes_id"] == pes_id].sort_values("timestamp").set_index("timestamp")
    daily = pd.DataFrame({
        "generation_mwh": reg["generation_mw"].resample("D").sum(),
        "installed_mwp":  reg["installed_mwp"].resample("D").mean(),
    })
    for c in WEATHER_COLS:
        daily[c] = reg[c].resample("D").mean()
    daily["cf"] = (daily["generation_mwh"] / (daily["installed_mwp"] * 24)).clip(0, 1)
    daily = daily.dropna()
    daily.index = daily.index.tz_localize(None)
    return daily


def build_design(daily, h, weather_aware):
    cf  = daily["cf"]
    doy = daily.index.dayofyear.values
    X = pd.DataFrame(index=daily.index)
    X["day_sin"]   = np.sin(2 * np.pi * doy / 365.25)
    X["day_cos"]   = np.cos(2 * np.pi * doy / 365.25)
    X["month_sin"] = np.sin(2 * np.pi * daily.index.month.values / 12)
    X["month_cos"] = np.cos(2 * np.pi * daily.index.month.values / 12)
    for k in range(0, 7):
        X[f"cf_lag{h + k}"] = cf.shift(h + k)
    X["cf_roll7"]  = cf.shift(h).rolling(7).mean()
    X["cf_roll30"] = cf.shift(h).rolling(30).mean()
    X["cf_lag365"] = cf.shift(365)
    if weather_aware:
        for c in WEATHER_COLS:
            X[c] = daily[c]
        tot = (daily["direct_radiation"] + daily["diffuse_radiation"]).replace(0, np.nan)
        X["beam_fraction"] = (daily["direct_radiation"] / tot).fillna(0.0)
    y = cf.copy()
    ok = X.notna().all(axis=1)
    return X[ok], y[ok]


def make_xgb():
    return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=SEED, n_jobs=-1, verbosity=0)


if __name__ == "__main__":
    df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])
    regions = (df[["pes_id", "region_name"]].drop_duplicates()
                 .set_index("pes_id")["region_name"].to_dict())

    # ---------------------------------------------------------------- SHAP ---
    frames = []
    for pes_id, name in regions.items():
        daily = load_daily(df, pes_id)
        X, y = build_design(daily, h=1, weather_aware=True)
        Xtr, Xte, ytr = X.iloc[:INITIAL_TRAIN], X.iloc[INITIAL_TRAIN:], y.iloc[:INITIAL_TRAIN]
        mod = make_xgb().fit(Xtr, ytr)
        sv = shap.TreeExplainer(mod).shap_values(Xte)      # held-out data only
        frames.append(pd.Series(np.abs(sv).mean(axis=0), index=X.columns, name=name))

    shap_df = pd.concat(frames, axis=1)
    shap_df["mean_abs_shap"] = shap_df.mean(axis=1)
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv("shap_importance.csv")

    print("=" * 74)
    print("SHAP — mean |value|, weather-aware XGBoost, averaged over regions")
    print("(computed on held-out test window only)")
    print("=" * 74)
    print(shap_df[["mean_abs_shap"]].head(14).to_string(float_format=lambda v: f"{v:.5f}"))

    # ----------------------------------------------- combined comparison -----
    sar = pd.read_csv("backtest_summary.csv")
    sar_agg = (sar.groupby("h")[["MAE", "RMSE", "R2", "MASE"]].mean().reset_index())
    sar_agg["model"] = "SARIMA/Fourier"
    sar_agg["info_set"] = "weather-blind"
    # annual error recomputed from the SARIMA prediction files (not hardcoded)
    ann = []
    for h in sorted(sar_agg["h"]):
        tot_t = tot_p = 0.0
        for pes_id in regions:
            p = pd.read_csv(f"backtest_preds_pes{pes_id}.csv")
            s = p[p["h"] == h]
            tot_t += s["y_true"].sum(); tot_p += s["y_pred"].sum()
        ann.append(100 * (tot_p - tot_t) / tot_t)
    sar_agg["annual_error_pct"] = ann

    ml = pd.read_csv("ml_backtest_summary.csv")
    ml_agg = (ml.groupby(["model", "info_set", "h"])
                [["MAE", "RMSE", "R2", "MASE", "annual_error_pct"]].mean().reset_index())

    cols = ["model", "info_set", "h", "MAE", "RMSE", "R2", "MASE", "annual_error_pct"]
    combined = (pd.concat([sar_agg[cols], ml_agg[cols]], ignore_index=True)
                  .sort_values(["h", "R2"], ascending=[True, False]))
    combined.to_csv("model_comparison.csv", index=False)

    print("\n" + "=" * 96)
    print("COMBINED COMPARISON — mean across regions, identical forecast origins")
    print("=" * 96)
    for h in sorted(combined["h"].unique()):
        print(f"\n--- horizon h = {h} days ---")
        print(combined[combined["h"] == h][cols[:2] + cols[3:]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
