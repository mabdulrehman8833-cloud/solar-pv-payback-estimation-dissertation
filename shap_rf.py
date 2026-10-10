"""SHAP attribution on the weather-aware RANDOM FOREST.

Supervisor's instruction: use Random Forest for the SHAP analysis. The original
pass used XGBoost, whose attribution below the top two positions was unstable
between library versions.

Two reasons RF is the better choice here, both worth saying in the write-up:
bagged trees average attribution over many independently grown trees rather
than a boosted sequence where later trees explain the residual of earlier ones;
and RF was the better-performing weather-aware model anyway (MAE 1.35% against
1.42%), so the attribution describes the model that is actually being reported.

Identical protocol to shap_and_compare_standalone.py otherwise: same design
matrix, same train/test split, SHAP computed on held-out data only.

Run:  python shap_rf.py
Out:  shap_importance_rf.csv
"""
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

from ml_backtest import load_daily, build_design, INITIAL_TRAIN

SEED = 42


def make_rf():
    """Same settings as the reported weather-aware Random Forest."""
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                 n_jobs=-1, random_state=SEED)


if __name__ == "__main__":
    df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])
    regions = (df[["pes_id", "region_name"]].drop_duplicates()
                 .set_index("pes_id")["region_name"].to_dict())

    frames = []
    for pes_id, name in regions.items():
        daily = load_daily(df, pes_id)
        X, y, _ = build_design(daily, h=1, weather_aware=True)
        Xtr, ytr = X.iloc[:INITIAL_TRAIN], y.iloc[:INITIAL_TRAIN]
        Xte = X.iloc[INITIAL_TRAIN:]
        mod = make_rf().fit(Xtr, ytr)
        sv = shap.TreeExplainer(mod).shap_values(Xte, check_additivity=False)
        frames.append(pd.Series(np.abs(sv).mean(axis=0), index=X.columns,
                                name=name.split(",")[0]))
        print(f"  {name:32s} done", flush=True)

    S = pd.concat(frames, axis=1)
    S["mean_abs_shap"] = S.mean(axis=1)
    # Spread across regions -- the supervisor wants variability shown, and for
    # SHAP it is the direct evidence for whether a ranking is stable.
    S["sd_across_regions"] = S.drop(columns="mean_abs_shap").std(axis=1, ddof=1)
    S["cv"] = S.sd_across_regions / S.mean_abs_shap
    S = S.sort_values("mean_abs_shap", ascending=False)
    S.to_csv("shap_importance_rf.csv")

    print("\n" + "=" * 86)
    print("SHAP — mean |value|, weather-aware RANDOM FOREST, held-out data only")
    print("=" * 86)
    print(S[["mean_abs_shap", "sd_across_regions", "cv"]].head(12)
           .to_string(float_format=lambda v: f"{v:.5f}"))
    print("\n  cv = coefficient of variation across the six regions.")
    print("  A high cv means the feature's importance is region-dependent, so")
    print("  its position in the ranking should not be read as stable.")
