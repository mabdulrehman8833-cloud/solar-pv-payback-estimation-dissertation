"""Complete the weather-aware XGBoost arm of the tuning grid."""
import itertools, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from tuning_full import annual_error, DEFAULTS
from ml_backtest import load_daily

GRID = [dict(n_estimators=500, learning_rate=lr, max_depth=d,
             subsample=0.8, colsample_bytree=0.8)
        for lr, d in itertools.product([0.05, 0.02], [2, 3, 5, 8])]

df = pd.read_csv("solar_weather_dataset.csv", parse_dates=["timestamp"])
regions = df[["pes_id", "region_name"]].drop_duplicates().values
panels = {int(p): load_daily(df, p) for p, _ in regions}

rows = []
for tag, params in [("default", DEFAULTS["XGB"])] + [("grid", g) for g in GRID]:
    e = np.array([annual_error(panels[p], "XGB", params, True) for p, _ in regions])
    rows.append({"model": "XGB", "info_set": "weather-aware", "source": tag,
                 "params": str(params), "MAE": np.abs(e).mean(),
                 "SD": e.std(ddof=1), "mean_error": e.mean()})
    print(f"  weather-aware  XGB  {tag:7s} MAE {rows[-1]['MAE']:6.2f}%  "
          f"SD {rows[-1]['SD']:5.2f}  {str(params)[:60]}", flush=True)
pd.DataFrame(rows).to_csv("tuning_xgb_wx.csv", index=False)
print("done")
